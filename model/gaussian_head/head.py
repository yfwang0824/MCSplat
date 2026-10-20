import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
try:
    import torch_scatter  # optional; we fall back to a torch-only implementation if unavailable
except Exception:
    torch_scatter = None

from typing import List, Dict, Tuple, Union, Optional

from model.ppisp_wrapper import CHROMABilateralGridHead

from ..layers.patch_embed import PatchEmbed
from ..layers.gaussian_dyn import GaussianRenderer_dyn
from ..layers.spconv_unet import get_voxel_centers, project_world_points_to_images
from ..layers.vision_transformer import vit_small
from .utils import create_uv_grid, position_grid_to_embed, is_point_in_frustum_batch
from .unet import UNet
from .head_layers import _make_scratch, _make_fusion_block_custom, custom_interpolate
from simple_knn_v2._C import distCUDACross
from pi3.models.layers.transformer_head import TransformerDecoder
from pi3.models.layers.pos_embed import RoPE2D, PositionGetter

def quaternion_to_matrix(quaternions: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.view(*quaternions.shape[:-1], 3, 3)

def gaussian_orientation_from_scales(
    rotation_wxyz: torch.Tensor,
    scale: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    rotation_wxyz = rotation_wxyz / (rotation_wxyz.norm(dim=-1, keepdim=True) + eps)
    rotation = quaternion_to_matrix(rotation_wxyz, eps=eps)
    column_idx = scale.argmin(dim=-1).to(torch.long)
    gather_idx = column_idx.unsqueeze(-1).unsqueeze(-2).expand(*rotation.shape[:-1], 1)
    direction = torch.gather(rotation, dim=-1, index=gather_idx).squeeze(-1)
    return direction / (direction.norm(dim=-1, keepdim=True) + eps)


class LidarImageEncoder(nn.Module):
    """Trainable per-pixel appearance encoder for LiDAR intensity.

    Extracts reflectance-relevant features from the GT camera image using the
    early stages of EfficientNetV2-S (~1/8 resolution), projects to ``out_dim``
    and upsamples to a target resolution. The output is sampled per source pixel
    and concatenated into the LiDAR intensity head input.
    """

    def __init__(self, out_dim: int = 64, pretrained: bool = True):
        super().__init__()
        from torchvision.models import (
            EfficientNet_V2_S_Weights,
            efficientnet_v2_s,
        )

        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_v2_s(weights=weights)
        # features[:4] -> [N, 64, H/8, W/8] (early/mid, keeps material detail).
        self.features = backbone.features[:4]
        self.proj = nn.Sequential(
            nn.Conv2d(64, out_dim, kernel_size=1),
            nn.GELU(),
        )
        self.register_buffer(
            "_imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "_imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, images: torch.Tensor, target_hw) -> torch.Tensor:
        # images: [N, 3, H, W] in [0, 1]; target_hw: (H_out, W_out)
        x = (images - self._imagenet_mean) / self._imagenet_std
        feat = self.features(x)
        feat = self.proj(feat)
        feat = F.interpolate(
            feat, size=tuple(target_hw), mode="bilinear", align_corners=False
        )
        return feat


class _LidarUNetDoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2 for LiDAR UNet head."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels, out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _LidarUNetDown(nn.Module):
    """Downscaling with maxpool then double conv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            _LidarUNetDoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _LidarUNetUp(nn.Module):
    """Upscaling then double conv."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
            self.conv = _LidarUNetDoubleConv(
                in_channels, out_channels, in_channels // 2
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = _LidarUNetDoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class LidarConv2dHead(nn.Module):
    """Image-space 2D conv head for per-pixel LiDAR intensity.

    Operates on the per-pixel grid of the to_lidar input features. Dilated convs
    give a large receptive field so each pixel's intensity can use neighborhood
    (material) context. Output: per-pixel intensity logit.
    """

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=4, dilation=4),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(
        self, x: torch.Tensor, return_multiscale: bool = False
    ) -> torch.Tensor:
        # x: [S, in_dim, H, W] -> [S, 1, H, W]
        return self.net(x)


class LidarUNetHead(nn.Module):
    """NeRF-LiDAR style UNet head for per-pixel LiDAR intensity logits."""

    def __init__(
        self,
        in_dim: int,
        base_channels: int = 64,
        bilinear: bool = True,
        enable_multiscale_outputs: bool = True,
    ) -> None:
        super().__init__()
        self.bilinear = bilinear
        self.enable_multiscale_outputs = bool(enable_multiscale_outputs)
        c1 = int(base_channels)
        c2, c3, c4 = c1 * 2, c1 * 4, c1 * 8
        factor = 2 if bilinear else 1
        c5 = (c1 * 16) // factor

        self.inc = _LidarUNetDoubleConv(in_dim, c1)
        self.down1 = _LidarUNetDown(c1, c2)
        self.down2 = _LidarUNetDown(c2, c3)
        self.down3 = _LidarUNetDown(c3, c4)
        self.down4 = _LidarUNetDown(c4, c5)
        self.up1 = _LidarUNetUp(c1 * 16, c4 // factor, bilinear=bilinear)
        self.up2 = _LidarUNetUp(c1 * 8, c3 // factor, bilinear=bilinear)
        self.up3 = _LidarUNetUp(c1 * 4, c2 // factor, bilinear=bilinear)
        self.up4 = _LidarUNetUp(c1 * 2, c1, bilinear=bilinear)
        self.outc = nn.Conv2d(c1, 1, kernel_size=1)
        self.outc_up1 = nn.Conv2d(c4 // factor, 1, kernel_size=1)
        self.outc_up2 = nn.Conv2d(c3 // factor, 1, kernel_size=1)
        self.outc_up3 = nn.Conv2d(c2 // factor, 1, kernel_size=1)

    def forward(
        self, x: torch.Tensor, return_multiscale: bool = False
    ) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        u1 = self.up1(x5, x4)
        u2 = self.up2(u1, x3)
        u3 = self.up3(u2, x2)
        u4 = self.up4(u3, x1)
        main = self.outc(u4)
        want_ms = bool(return_multiscale and self.enable_multiscale_outputs)
        if not want_ms:
            return main
        return {
            "main": main,
            # Coarse -> fine auxiliary supervision maps.
            "up1": self.outc_up1(u1),
            "up2": self.outc_up2(u2),
            "up3": self.outc_up3(u3),
        }


class _LidarFourierFeatures(nn.Module):
    """Fourier feature mapping for normalized UV coordinates (Open-d4rt style)."""

    def __init__(self, input_dim: int = 2, num_bands: int = 6, include_input: bool = True):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_bands = int(num_bands)
        self.include_input = bool(include_input)
        self.register_buffer(
            "frequencies",
            2.0 ** torch.arange(self.num_bands, dtype=torch.float32) * math.pi,
            persistent=False,
        )

    @property
    def output_dim(self) -> int:
        base = self.input_dim if self.include_input else 0
        return base + self.input_dim * self.num_bands * 2

    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        # uv: [..., input_dim] normalized to [0, 1]
        uv = uv.to(dtype=torch.float32)
        x = uv.unsqueeze(-1) * self.frequencies.view(*([1] * uv.dim()), -1)
        sin = torch.sin(x)
        cos = torch.cos(x)
        out = [sin, cos]
        if self.include_input:
            out.insert(0, uv.unsqueeze(-1))
        merged = torch.cat(out, dim=-1)
        return merged.flatten(start_dim=uv.dim() - 1)


class _LidarQueryCrossAttnLayer(nn.Module):
    """Cross-attention only block (no query self-attention), Open-d4rt style.

    Each query token (shape [P, D]) attends over its own set of local memory
    tokens (shape [P, L, D]). A learnable per-head relative-position bias over
    the L window slots is added to the attention logits.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float,
        window_len: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.rel_bias = nn.Parameter(torch.zeros(self.num_heads, int(window_len)))

        ff_dim = int(round(hidden_dim * mlp_ratio))
        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        # x: [P, D]; mem: [P, L, D]
        P = x.shape[0]
        L = mem.shape[1]
        q = self.norm_q(x)
        m = self.norm_kv(mem)
        qh = self.q_proj(q).view(P, self.num_heads, self.head_dim)
        kh = self.k_proj(m).view(P, L, self.num_heads, self.head_dim)
        vh = self.v_proj(m).view(P, L, self.num_heads, self.head_dim)
        # attn logits: [P, num_heads, L]
        attn = torch.einsum("phd,plhd->phl", qh, kh) * self.scale
        attn = attn + self.rel_bias.unsqueeze(0)  # [1, H, L]
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("phl,plhd->phd", attn, vh).reshape(P, self.hidden_dim)
        x = x + self.out_proj(out)
        x = x + self.ff(self.norm_ff(x))
        return x


class LidarQueryHead(nn.Module):
    """Open-d4rt style independent-query cross-attention head for intensity.

    - Query  = per-point geometry token: ray direction, surface normal,
      incidence cosine, range features/embedding, ray embedding, plus a Fourier
      encoding of the (image-plane) UV location.
    - Key/Value = local ``window_size x window_size`` neighborhood of appearance/
      context memory (sample_features + reflectance + image features) scattered
      on the source-pixel grid around each query's own pixel.
    - Cross-attention only (no query self-attention): every point independently
      reads its local image/gaussian context and regresses one intensity logit.

    Operates on the source1 (pixel) Gaussians only; voxel Gaussians are handled
    by the parallel MLP in the caller (mirrors the conv2d/unet heads).
    """

    def __init__(
        self,
        q_in_dim: int,
        kv_component_dims,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        window_size: int = 5,
        uv_num_bands: int = 6,
        mlp_ratio: float = 2.0,
        query_chunk_size: int = 65536,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        num_heads = int(num_heads)
        if hidden_dim % num_heads != 0:
            hidden_dim = (hidden_dim // num_heads) * num_heads
            hidden_dim = max(hidden_dim, num_heads)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.window_size = max(1, int(window_size))
        self.window_len = self.window_size * self.window_size
        self.query_chunk_size = max(1, int(query_chunk_size))

        self.uv_encoder = _LidarFourierFeatures(
            input_dim=2, num_bands=int(uv_num_bands), include_input=True
        )
        self.q_proj_in = nn.Linear(int(q_in_dim), hidden_dim)
        self.uv_proj = nn.Linear(self.uv_encoder.output_dim, hidden_dim)
        # Additive K/V input projection over separate components (sample_features
        # / reflectance / image feats) so we never materialize a big concatenated
        # [n1, sum(dims)] tensor (memory-tight training).
        self.kv_component_dims = [int(d) for d in kv_component_dims if int(d) > 0]
        self.kv_in_projs = nn.ModuleList(
            [nn.Linear(d, hidden_dim) for d in self.kv_component_dims]
        )
        self.kv_mix = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.q_norm_in = nn.LayerNorm(hidden_dim)

        self.layers = nn.ModuleList(
            [
                _LidarQueryCrossAttnLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    window_len=self.window_len,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Window offset grid (row/col), registered for device movement.
        k = self.window_size
        half = k // 2
        offs = torch.arange(k) - half
        dr, dc = torch.meshgrid(offs, offs, indexing="ij")
        self.register_buffer("_win_dr", dr.reshape(-1).long(), persistent=False)
        self.register_buffer("_win_dc", dc.reshape(-1).long(), persistent=False)

    def _neighbor_index(self, source1_pix, S, H, W):
        """Compute [n1, L] flat neighbor indices into the [S*H*W] grid."""
        device = source1_pix.device
        hw = int(H) * int(W)
        s = torch.div(source1_pix, hw, rounding_mode="floor")
        rem = source1_pix - s * hw
        r = torch.div(rem, int(W), rounding_mode="floor")
        c = rem - r * int(W)
        dr = self._win_dr.to(device).view(1, -1)
        dc = self._win_dc.to(device).view(1, -1)
        rr = (r.view(-1, 1) + dr).clamp_(0, int(H) - 1)
        cc = (c.view(-1, 1) + dc).clamp_(0, int(W) - 1)
        nbr = s.view(-1, 1) * hw + rr * int(W) + cc
        # UV normalized to [0, 1] for Fourier encoding.
        denom_w = max(1, int(W) - 1)
        denom_h = max(1, int(H) - 1)
        u = c.float() / float(denom_w)
        v = r.float() / float(denom_h)
        uv = torch.stack([u, v], dim=-1)
        return nbr, uv

    def _chunk_decode(self, mem_flat, q_chunk, idx_chunk):
        """Gather local memory windows (inside checkpoint) and run the decoder.

        Gathering is done here (not before checkpointing) so autograd retains only
        ``mem_flat`` [num_pix, D] and the per-chunk query token, recomputing the
        [c, L, D] neighbor window in backward instead of storing it for every chunk.
        """
        c = q_chunk.shape[0]
        mem = mem_flat[idx_chunk.reshape(-1)].view(
            c, self.window_len, self.hidden_dim
        )  # [c, L, D]
        x = q_chunk
        for layer in self.layers:
            x = layer(x, mem)
        return self.out_head(self.out_norm(x))

    def forward(self, q_s1, kv_list, source1_pix, S, H, W):
        """q_s1: [n1, Dq]; kv_list: list of [n1, d_i] context components;
        source1_pix: [n1] flat grid idx.

        Returns per-source1 intensity logit [n1, 1].
        """
        n1 = int(q_s1.shape[0])
        if n1 <= 0:
            return q_s1.new_zeros((0, 1))
        num_pix = int(S) * int(H) * int(W)
        compute_dtype = torch.float32
        q_s1 = q_s1.to(compute_dtype)

        nbr_idx, uv = self._neighbor_index(source1_pix, S, H, W)  # [n1, L], [n1, 2]

        # Additive K/V projection (no big concat): sum per-component Linears.
        mem_sum = None
        for proj, comp in zip(self.kv_in_projs, kv_list):
            h = proj(comp.to(compute_dtype))
            mem_sum = h if mem_sum is None else mem_sum + h
        if mem_sum is None:
            mem_sum = q_s1.new_zeros((n1, self.hidden_dim))
        mem_hidden_s1 = self.kv_mix(mem_sum)  # [n1, D]

        # Scatter source1 memory features to the full pixel grid (zeros elsewhere).
        mem_flat = q_s1.new_zeros((num_pix, self.hidden_dim))
        mem_flat[source1_pix] = mem_hidden_s1

        # Query tokens: geometry projection + UV Fourier.
        uv_tok = self.uv_proj(self.uv_encoder(uv))
        q_tok = self.q_norm_in(self.q_proj_in(q_s1) + uv_tok)  # [n1, D]

        # Chunk over queries. Under autograd, checkpoint each chunk so the large
        # per-window activations are recomputed in backward (bounds memory to a
        # single chunk instead of the whole [n1, L, D] set).
        use_ckpt = torch.is_grad_enabled() and (
            q_tok.requires_grad or mem_flat.requires_grad
        )
        out_chunks = []
        chunk = self.query_chunk_size
        for start in range(0, n1, chunk):
            end = min(start + chunk, n1)
            idx = nbr_idx[start:end]  # [c, L]
            q_chunk = q_tok[start:end]  # [c, D]
            if use_ckpt:
                out = checkpoint(
                    self._chunk_decode,
                    mem_flat,
                    q_chunk,
                    idx,
                    use_reentrant=False,
                )
            else:
                out = self._chunk_decode(mem_flat, q_chunk, idx)
            out_chunks.append(out)
        return torch.cat(out_chunks, dim=0)


class GuassianHead(nn.Module):
    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        out_channels: List[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: List[int] = [4, 11, 17, 23],
        pos_embed: bool = True,
        cfg: Dict = None,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.pos_embed = pos_embed
        self._freeze_gaussian_rgb = False
        render_w, render_h = cfg.get("resolution")
        self.intermediate_layer_idx = [3, 8, 12, 16]

        self.norm = nn.LayerNorm(dim_in)
        # Projection layers for each output channel from tokens.
        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=dim_in,
                    out_channels=oc,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for oc in out_channels
            ]
        )

        # Resize layers for upsampling feature maps.
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0],
                    out_channels=out_channels[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1],
                    out_channels=out_channels[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3],
                    out_channels=out_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )
        self.scratch = _make_scratch(
            out_channels,
            256,
            expand=False,
        )
        self.scratch.refinenet1 = _make_fusion_block_custom(256)
        self.scratch.refinenet2 = _make_fusion_block_custom(256)
        self.scratch.refinenet3 = _make_fusion_block_custom(256)
        self.scratch.refinenet4 = _make_fusion_block_custom(256, has_residual=False)
        self.scratch.output_conv1 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1
        )

        self.use_fisher_loss_mask = False

        self.voxel_proj = nn.Conv2d(
            in_channels=256, out_channels=32, kernel_size=1, stride=1, padding=0
        )
        self.plucker_to_embed = PatchEmbed(
            img_size=(render_h, render_w),
            patch_size=14,
            in_chans=6,
            embed_dim=1024,
            norm_layer=nn.LayerNorm,
        )
        self.depth_embeds = PatchEmbed(
            img_size=(render_h, render_w),
            patch_size=14,
            in_chans=5,
            embed_dim=1024,
            norm_layer=nn.LayerNorm,
        )
        self.embed_proj = nn.Sequential(
            nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Linear(1024, 2048)
        )

        # Gaussian layers used by stage1/stage2 configs.
        self._use_exposure_integrated = False
        out_dim = 15
        point_feature_dim = 320
        self.to_gaussians = nn.Sequential(
            nn.Linear(point_feature_dim, 256),
            nn.GELU(),
            nn.Linear(256, out_dim),
        )
        self.to_gaussians_sky = nn.Sequential(
            nn.Linear(point_feature_dim, 256),
            nn.GELU(),
            nn.Linear(256, out_dim),
        )

        # Optional small-scale initialization for the Gaussian scale channels
        # (raw[..., 4:7]). Default (None) keeps PyTorch's Linear init, which
        # after scale_act = softplus(x, beta=2) yields ~0.35m (then clamped by
        # max_scale). Setting cfg `scale_init` (point branch) / `sky_scale_init`
        # (sky branch, defaults to scale_init) makes the initial output start
        # near the target value instead.
        def _init_scale_channels(head, target_scale, beta=2.0, weight_gain=0.01):
            # inverse softplus: raw such that softplus(raw, beta) == target
            raw = math.log(math.expm1(beta * float(target_scale))) / beta
            last = head[-1]
            with torch.no_grad():
                # Shrink the scale-channel weight rows so the initial output is
                # dominated by the bias (still trainable, just starts tight).
                last.weight[4:7].mul_(weight_gain)
                last.bias[4:7].fill_(raw)

        _scale_init = cfg.get("scale_init", None) if cfg is not None else None
        _sky_scale_init = (
            cfg.get("sky_scale_init", _scale_init) if cfg is not None else None
        )
        if _scale_init is not None:
            _init_scale_channels(self.to_gaussians, _scale_init)
        if _sky_scale_init is not None:
            _init_scale_channels(self.to_gaussians_sky, _sky_scale_init)

        # LiDAR intensity head. Input per Gaussian:
        #   [sample_features(optional), LiDAR ray dir(3), reflectance embedding,
        #    surface normal in LiDAR frame(3), incidence cosine |d.n|(1),
        #    optional LiDAR-ray embedding].
        # The reflectance embedding is produced by a small learnable encoder on
        # the per-Gaussian RGB (a content/material proxy), letting the model learn
        # an appearance -> reflectance mapping. The normal + incidence cosine give
        # the physical cos(incidence) term of the LiDAR radiometric model. The
        # output scalar logit is alpha-blended into the main camera render as an
        # extra channel.
        _lidar_refl_dim = int(cfg.get("lidar_reflectance_embed_dim", 16))
        self.lidar_reflectance_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
            nn.Linear(32, _lidar_refl_dim),
        )
        # Optional per-pixel image inputs, both aligned to each source pixel:
        #  - trainable EfficientNetV2 encoder features (lidar_use_image_encoder)
        #  - raw GT RGB (lidar_use_rgb_input)
        self.lidar_use_image_encoder = bool(
            cfg.get("lidar_use_image_encoder", False)
        )
        self.lidar_use_rgb_input = bool(cfg.get("lidar_use_rgb_input", False))
        self.lidar_use_sample_features = bool(
            cfg.get("lidar_use_sample_features", True)
        )
        # Debug/ablation switch: replace per-Gaussian LiDAR ray direction with
        # ray directions reconstructed from GT LiDAR-image depth whenever valid.
        # Invalid pixels (and voxel branch) fall back to geometry-derived rays.
        self.lidar_use_gt_ray_dir = bool(
            cfg.get("lidar_use_gt_ray_dir", False)
        )
        # Optional learnable embedding per LiDAR-plane ray (row/col bin).
        self.lidar_use_ray_embedding = bool(
            cfg.get("lidar_use_ray_embedding", False)
        )
        self.lidar_ray_embed_dim = int(cfg.get("lidar_ray_embed_dim", 8))
        self.lidar_ray_embed_height = int(cfg.get("lidar_ray_embed_height", 64))
        self.lidar_ray_embed_width = int(cfg.get("lidar_ray_embed_width", 2650))
        self.lidar_ray_embed_sensor_index = int(
            cfg.get("lidar_ray_embed_sensor_index", 0)
        )
        if (
            self.lidar_use_ray_embedding
            and self.lidar_ray_embed_dim > 0
            and self.lidar_ray_embed_height > 0
            and self.lidar_ray_embed_width > 0
        ):
            self.lidar_ray_embedding = nn.Embedding(
                self.lidar_ray_embed_height * self.lidar_ray_embed_width,
                self.lidar_ray_embed_dim,
            )
        else:
            self.lidar_ray_embedding = None
            self.lidar_use_ray_embedding = False
            self.lidar_ray_embed_dim = 0
        # Distance-to-LiDAR inputs for intensity attenuation/material response.
        self.lidar_use_range_features = bool(
            cfg.get("lidar_use_range_features", False)
        )
        # Debug/ablation switch: replace distance-to-LiDAR with GT LiDAR-image
        # range where available. Invalid pixels and voxel branch fall back.
        self.lidar_use_gt_range = bool(cfg.get("lidar_use_gt_range", False))
        self.lidar_use_range_embedding = bool(
            cfg.get("lidar_use_range_embedding", False)
        )
        self.lidar_range_embed_dim = int(cfg.get("lidar_range_embed_dim", 8))
        self.lidar_range_embed_bins = int(cfg.get("lidar_range_embed_bins", 128))
        self.lidar_range_min = float(cfg.get("lidar_range_min", 0.0))
        self.lidar_range_max = float(cfg.get("lidar_range_max", 120.0))
        if self.lidar_range_max <= self.lidar_range_min:
            self.lidar_range_max = self.lidar_range_min + 1.0
        if (
            self.lidar_use_range_embedding
            and self.lidar_range_embed_dim > 0
            and self.lidar_range_embed_bins > 1
        ):
            self.lidar_range_embedding = nn.Embedding(
                self.lidar_range_embed_bins,
                self.lidar_range_embed_dim,
            )
        else:
            self.lidar_range_embedding = None
            self.lidar_use_range_embedding = False
            self.lidar_range_embed_dim = 0
        _lidar_img_embed_dim = (
            int(cfg.get("lidar_image_embed_dim", 64))
            if self.lidar_use_image_encoder
            else 0
        )
        self.lidar_image_encoder = (
            LidarImageEncoder(
                out_dim=_lidar_img_embed_dim,
                pretrained=bool(cfg.get("lidar_image_encoder_pretrained", True)),
            )
            if self.lidar_use_image_encoder
            else None
        )
        # Per-point extra image-derived feature dim = encoder embed (+3 raw RGB).
        _lidar_img_dim = _lidar_img_embed_dim + (3 if self.lidar_use_rgb_input else 0)
        _lidar_sample_feat_dim = point_feature_dim if self.lidar_use_sample_features else 0
        _lidar_ray_embed_dim = self.lidar_ray_embed_dim if self.lidar_use_ray_embedding else 0
        _lidar_range_feat_dim = 3 if self.lidar_use_range_features else 0
        _lidar_range_embed_dim = (
            self.lidar_range_embed_dim if self.lidar_use_range_embedding else 0
        )
        lidar_in_dim = (
            _lidar_sample_feat_dim
            + 3
            + _lidar_refl_dim
            + 3
            + 1
            + _lidar_img_dim
            + _lidar_ray_embed_dim
            + _lidar_range_feat_dim
            + _lidar_range_embed_dim
        )
        self._lidar_in_dim = lidar_in_dim
        # Query-head (Open-d4rt style) input split:
        #  - Q (geometry): ray dir(3) + normal(3) + incidence(1) + range feats +
        #    range embed + ray embed  (+ UV Fourier added inside the head).
        #  - K/V (appearance/context): sample_features + reflectance embed +
        #    image features.
        self._lidar_query_q_dim = (
            3
            + 3
            + 1
            + _lidar_range_feat_dim
            + _lidar_range_embed_dim
            + _lidar_ray_embed_dim
        )
        self._lidar_query_kv_dim = (
            _lidar_sample_feat_dim + _lidar_refl_dim + _lidar_img_dim
        )
        # K/V component dims, in the exact order the forward assembles kv_list:
        # [sample_features(optional), reflectance embed, image feats(optional)].
        self._lidar_query_kv_component_dims = []
        if _lidar_sample_feat_dim > 0:
            self._lidar_query_kv_component_dims.append(_lidar_sample_feat_dim)
        self._lidar_query_kv_component_dims.append(_lidar_refl_dim)
        if _lidar_img_dim > 0:
            self._lidar_query_kv_component_dims.append(_lidar_img_dim)
        self.to_lidar = nn.Sequential(
            nn.Linear(lidar_in_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )
        # Intensity head type:
        #  - "mlp": per-Gaussian MLP (default)
        #  - "conv2d": image-space dilated-conv head
        #  - "unet": image-space UNet head
        #  - "query": Open-d4rt style independent-query cross-attention head
        # 2D/query heads use scatter->head->gather for pixel/sky gaussians; voxel
        # gaussians fall back to a parallel MLP `to_lidar_voxel`.
        self.lidar_head_type = str(cfg.get("lidar_head_type", "mlp")).lower()
        self.lidar_unet_multiscale_supervision = bool(
            cfg.get("lidar_unet_multiscale_supervision", True)
        )
        self.lidar_query_head = None
        if self.lidar_head_type in {"conv2d", "unet"}:
            if self.lidar_head_type == "unet":
                self.lidar_conv2d_head = LidarUNetHead(
                    lidar_in_dim,
                    base_channels=int(cfg.get("lidar_unet_base_channels", 64)),
                    bilinear=bool(cfg.get("lidar_unet_bilinear", True)),
                    enable_multiscale_outputs=self.lidar_unet_multiscale_supervision,
                )
            else:
                self.lidar_conv2d_head = LidarConv2dHead(
                    lidar_in_dim,
                    hidden=int(cfg.get("lidar_conv2d_hidden", 64)),
                )
            self.to_lidar_voxel = nn.Sequential(
                nn.Linear(lidar_in_dim, 256),
                nn.GELU(),
                nn.Linear(256, 1),
            )
        elif self.lidar_head_type == "query":
            self.lidar_conv2d_head = None
            self.lidar_query_head = LidarQueryHead(
                q_in_dim=max(1, self._lidar_query_q_dim),
                kv_component_dims=self._lidar_query_kv_component_dims,
                hidden_dim=int(cfg.get("lidar_query_hidden_dim", 128)),
                num_heads=int(cfg.get("lidar_query_num_heads", 4)),
                num_layers=int(cfg.get("lidar_query_num_layers", 2)),
                window_size=int(cfg.get("lidar_query_window_size", 5)),
                uv_num_bands=int(cfg.get("lidar_query_uv_bands", 6)),
                mlp_ratio=float(cfg.get("lidar_query_mlp_ratio", 2.0)),
                query_chunk_size=int(cfg.get("lidar_query_chunk_size", 65536)),
            )
            self.to_lidar_voxel = nn.Sequential(
                nn.Linear(lidar_in_dim, 256),
                nn.GELU(),
                nn.Linear(256, 1),
            )
        else:
            self.lidar_conv2d_head = None
            self.to_lidar_voxel = None
        self.use_velocity = False
        self._velocity_max = float(cfg.get("velocity_max", 10.0))
        self._velocity_as_dynamic = False
        self._velocity_dynamic_scale = 0.0
        self.voxel_exposure_head = None
        self.feature_norm = nn.Identity()
        self.offset_act = lambda x: F.tanh(x) * cfg.get("offset_scale", 1.0)
        self.opt_act = torch.sigmoid
        self.scale_act = lambda x: F.softplus(x, beta=2.0)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = torch.sigmoid
        self.renderer = GaussianRenderer_dyn(
            resolution=[render_h, render_w], znear=0.1, zfar=1000.0
        )
        self.cfg = cfg
        self._sky_geometry_log_once = set()

        self.pts_range = torch.tensor(
            cfg.pts_range, dtype=torch.float
        )  # x_min, y_min, z_min, x_max, y_max, z_max
        self.voxel_size = torch.tensor(
            cfg.voxel_size, dtype=torch.float
        )  # voxel size in meters
        self.grid_size = (self.pts_range[3:6] - self.pts_range[:3]) / self.voxel_size[
            :3
        ]  # grid size in voxels
        self.grid_size = self.grid_size.int()
        self.scale_xyz = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.scale_yz = self.grid_size[1] * self.grid_size[2]
        self.scale_z = self.grid_size[2]

        # UNet for Scaffold feature extraction
        self.unet = UNet(
            self.grid_size.numpy(),
            self.voxel_size.numpy(),
            self.pts_range.numpy(),
            cfg.voxel_gs_num,
        )
        self.image_backbone = vit_small(
            img_size=render_w,
            patch_size=14,
            num_register_tokens=4,
            interpolate_antialias=True,
            interpolate_offset=0.0,
            block_chunks=0,
            init_values=1.0,
        )
        self._resnet_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        self._resnet_mean = self._resnet_mean[None, None, :, None, None]
        self._resnet_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        self._resnet_std = self._resnet_std[None, None, :, None, None]
        self.dinov2_proj = nn.Sequential(nn.Linear(384, 2048), nn.LayerNorm(2048))
        self.history_queue = Gaussians_Queue_v2()

        self.proj_pt = None
        self.proj_vox = None
        self.proj_view_embed = None
        self.default_view_embed = None

        # scale prediction
        self.scale_head = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )
        self.shift_head = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )
        self.rope = RoPE2D(freq=100.0)
        self.position_getter = PositionGetter()
        self.point_decoder = TransformerDecoder(
            in_dim=2048,
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
        )

        # Bilateral-grid transformer head.
        self._isp_output_mode = "disabled"
        self.use_bilateral_grid_transformer = bool(
            cfg.get("use_bilateral_grid_transformer", False)
        )
        self._bilateral_grid_chroma_enabled = bool(
            cfg.get("bilateral_grid_chroma_enabled", True)
        )
        self.bilateral_grid_transformer = None
        if self.use_bilateral_grid_transformer:
            try:
                if not self._bilateral_grid_chroma_enabled:
                    raise ValueError(
                        "Legacy non-CHROMA bilateral-grid path has been removed; "
                        "please set bilateral_grid_chroma_enabled=true."
                    )

                def _parse_grid_sizes(raw_value):
                    if isinstance(raw_value, (list, tuple)):
                        out_sizes = []
                        for item in raw_value:
                            if isinstance(item, (list, tuple)) and len(item) == 3:
                                out_sizes.append(tuple(int(v) for v in item))
                        return out_sizes
                    text = str(raw_value)
                    out_sizes = []
                    for part in text.split(";"):
                        part = part.strip()
                        if not part:
                            continue
                        vals = [int(x.strip()) for x in part.split(",") if x.strip()]
                        if len(vals) == 3:
                            out_sizes.append((vals[0], vals[1], vals[2]))
                    return out_sizes

                def _parse_guidance_factors(raw_value):
                    if isinstance(raw_value, (list, tuple)):
                        return [int(v) for v in raw_value]
                    text = str(raw_value)
                    return [int(x.strip()) for x in text.split(",") if x.strip()]

                _grid_sizes = _parse_grid_sizes(
                    cfg.get(
                        "bilateral_grid_sizes",
                        "2,2,1;4,4,2;8,8,4",
                    )
                )
                if len(_grid_sizes) == 0:
                    _grid_sizes = [(2, 2, 1), (4, 4, 2), (8, 8, 4)]
                _guidance_factors = _parse_guidance_factors(
                    cfg.get(
                        "bilateral_grid_guidance_factors",
                        "4,4,2",
                    )
                )
                if len(_guidance_factors) != len(_grid_sizes):
                    _guidance_factors = [4] * len(_grid_sizes)

                _chroma_kwargs = {
                    "enabled": True,
                    "token_dim": int(
                        cfg.get("bilateral_grid_chroma_token_dim", 512)
                    ),
                    "num_decoder_layers": int(
                        cfg.get("bilateral_grid_chroma_num_layers", 3)
                    ),
                    "num_heads": int(
                        cfg.get("bilateral_grid_chroma_num_heads", 8)
                    ),
                    "ff_dim": int(cfg.get("bilateral_grid_chroma_ff_dim", 2048)),
                    "patch_feat_dim": int(
                        cfg.get("bilateral_grid_appearance_feature_dim", 384)
                    ),
                    "ref_cam_ids": list(
                        cfg.get("bilateral_grid_chroma_ref_cameras", []) or []
                    ),
                    "kv_mode": str(
                        cfg.get("bilateral_grid_chroma_kv_mode", "residual")
                    ),
                    "use_cam_id_embed": bool(
                        cfg.get("bilateral_grid_chroma_use_cam_id_embed", True)
                    ),
                }

                self.bilateral_grid_transformer = CHROMABilateralGridHead(
                    num_cameras=int(cfg.get("num_cameras", 5)),
                    token_dim=_chroma_kwargs["token_dim"],
                    num_decoder_layers=_chroma_kwargs["num_decoder_layers"],
                    num_heads=_chroma_kwargs["num_heads"],
                    ff_dim=_chroma_kwargs["ff_dim"],
                    patch_feat_dim=_chroma_kwargs["patch_feat_dim"],
                    grid_sizes=_grid_sizes,
                    guidance_factors=_guidance_factors,
                    ref_cam_ids=_chroma_kwargs["ref_cam_ids"],
                    kv_mode=_chroma_kwargs["kv_mode"],
                    use_cam_id_embed=_chroma_kwargs["use_cam_id_embed"],
                    device="cuda",
                )
                self._isp_output_mode = "bilateral_grid"
            except ImportError:
                self.bilateral_grid_transformer = None
                self._isp_output_mode = "disabled"

        self.use_lidar_supervision = bool(cfg.get("use_lidar_supervision", False))

    def _lidar_conv2d_forward(
        self,
        to_lidar_in,
        n1,
        pt_sky_mask,
        valid_mask_i,
        S,
        H,
        W,
        collect_multiscale: bool = False,
    ):
        """Image-space 2D intensity head (conv2d/unet).

        Scatters the per-point to_lidar input (in gaussians_all order: source1 =
        pixel Gaussians [non-sky, sky], then voxel) onto the per-pixel grid, runs
        the 2D head, and gathers back. Voxel Gaussians use the parallel MLP.
        Returns per-Gaussian intensity logit [N, 1] (gaussians_all order), or None
        if the pixel mapping cannot be established (caller falls back to MLP).
        """
        num_pix = int(S) * int(H) * int(W)
        device = to_lidar_in.device
        if pt_sky_mask.numel() == num_pix:
            base_idx = torch.arange(num_pix, device=device)
        elif (
            valid_mask_i is not None
            and int(valid_mask_i.reshape(-1).sum()) == pt_sky_mask.numel()
        ):
            base_idx = torch.nonzero(
                valid_mask_i.reshape(-1).bool(), as_tuple=False
            ).reshape(-1).to(device)
        else:
            return None

        source1_pix = torch.cat(
            [base_idx[~pt_sky_mask], base_idx[pt_sky_mask]], dim=0
        )  # [n1], matches gaussians_source1 order
        if source1_pix.shape[0] != n1:
            return None

        D = to_lidar_in.shape[1]
        feat_s1 = to_lidar_in[:n1]
        grid = to_lidar_in.new_zeros(num_pix, D)
        grid[source1_pix] = feat_s1
        grid = grid.view(int(S), int(H), int(W), D).permute(0, 3, 1, 2).contiguous()
        head_out = self.lidar_conv2d_head(
            grid,
            return_multiscale=(
                collect_multiscale and self.lidar_head_type == "unet"
            ),
        )
        inten_multiscale = None
        if isinstance(head_out, dict):
            inten = head_out.get("main", None)
            if not isinstance(inten, torch.Tensor):
                return None
            inten_multiscale = {
                f"intensity_logits_{k}": v
                for k, v in head_out.items()
                if k != "main" and isinstance(v, torch.Tensor)
            }
        else:
            inten = head_out
        # inten: [S, 1, H, W]
        inten_flat = inten.permute(0, 2, 3, 1).reshape(num_pix, 1)
        inten_s1 = inten_flat[source1_pix]  # [n1, 1]

        feat_voxel = to_lidar_in[n1:]
        if feat_voxel.shape[0] > 0:
            inten_voxel = self.to_lidar_voxel(feat_voxel)
            logits = torch.cat([inten_s1, inten_voxel], dim=0)
        else:
            logits = inten_s1
        if isinstance(inten_multiscale, dict) and len(inten_multiscale) > 0:
            return {
                "gaussian_logits": logits,
                "image_logits_multiscale": inten_multiscale,
            }
        return logits

    def _lidar_source1_pixel_index(self, n1, pt_sky_mask, valid_mask_i, S, H, W, device):
        """Flat [n1] grid index for source1 (pixel) Gaussians in gaussians order.

        Mirrors the mapping used by the 2D conv/unet heads. Returns None if the
        pixel mapping cannot be established.
        """
        num_pix = int(S) * int(H) * int(W)
        if pt_sky_mask.numel() == num_pix:
            base_idx = torch.arange(num_pix, device=device)
        elif (
            valid_mask_i is not None
            and int(valid_mask_i.reshape(-1).sum()) == pt_sky_mask.numel()
        ):
            base_idx = torch.nonzero(
                valid_mask_i.reshape(-1).bool(), as_tuple=False
            ).reshape(-1).to(device)
        else:
            return None
        source1_pix = torch.cat(
            [base_idx[~pt_sky_mask], base_idx[pt_sky_mask]], dim=0
        )
        if source1_pix.shape[0] != n1:
            return None
        return source1_pix

    def _lidar_query_forward(
        self,
        q_s1,
        kv_list,
        voxel_feat,
        source1_pix,
        S,
        H,
        W,
    ):
        """Open-d4rt style query cross-attention intensity head.

        q_s1 / kv_list are the source1 (pixel) Gaussian features (kv_list is the
        list of context components); voxel_feat holds the full-dim to_lidar input
        for voxel Gaussians (or None). source1 Gaussians go through the query head;
        voxel Gaussians use the parallel MLP `to_lidar_voxel`. Returns per-Gaussian
        intensity logit [N, 1] or None.
        """
        if self.lidar_query_head is None or not isinstance(source1_pix, torch.Tensor):
            return None
        inten_s1 = self.lidar_query_head(q_s1, kv_list, source1_pix, S, H, W)
        inten_s1 = inten_s1.to(q_s1.dtype)
        if isinstance(voxel_feat, torch.Tensor) and voxel_feat.shape[0] > 0:
            inten_voxel = self.to_lidar_voxel(voxel_feat).to(inten_s1.dtype)
            logits = torch.cat([inten_s1, inten_voxel], dim=0)
        else:
            logits = inten_s1
        return logits

    def _select_lidar_supervision_for_batch(
        self,
        lidar_supervision_batch,
        batch_idx: int,
        batch_size: int,
    ):
        if not isinstance(lidar_supervision_batch, dict):
            return None
        lidar_supervision_i = {}
        for key, value in lidar_supervision_batch.items():
            if isinstance(value, torch.Tensor):
                if value.dim() > 0 and value.shape[0] == batch_size:
                    lidar_supervision_i[key] = value[batch_idx]
                else:
                    lidar_supervision_i[key] = value
            else:
                lidar_supervision_i[key] = value
        return lidar_supervision_i

    def _get_lidar_ray_embedding(
        self,
        lidar_ray_dir: torch.Tensor,
        lidar_supervision_i: Optional[Dict] = None,
    ) -> Optional[torch.Tensor]:
        """Lookup learnable embedding for each LiDAR ray direction."""
        if (
            not self.lidar_use_ray_embedding
            or self.lidar_ray_embedding is None
            or not isinstance(lidar_ray_dir, torch.Tensor)
            or lidar_ray_dir.dim() != 2
            or lidar_ray_dir.shape[-1] != 3
        ):
            return None
        device = lidar_ray_dir.device
        dtype = lidar_ray_dir.dtype
        n_pts = int(lidar_ray_dir.shape[0])
        if n_pts <= 0:
            return lidar_ray_dir.new_zeros((0, self.lidar_ray_embed_dim))

        # Fallback: uniform spherical quantization.
        xy_norm = torch.sqrt((lidar_ray_dir[:, :2] ** 2).sum(dim=-1).clamp_min(1e-12))
        az = torch.atan2(lidar_ray_dir[:, 1], lidar_ray_dir[:, 0])  # [-pi, pi]
        el = torch.atan2(lidar_ray_dir[:, 2], xy_norm)  # [-pi/2, pi/2]
        row_idx = torch.floor(
            (el + math.pi / 2.0) / math.pi * float(self.lidar_ray_embed_height)
        ).long()
        col_idx = torch.floor(
            (az + math.pi) / (2.0 * math.pi) * float(self.lidar_ray_embed_width)
        ).long()

        # Prefer dataset-defined LiDAR raster bins when available.
        if isinstance(lidar_supervision_i, dict):
            raster_pts = lidar_supervision_i.get("raster_pts", None)
            if isinstance(raster_pts, torch.Tensor) and raster_pts.dim() == 4:
                sensor_idx = int(
                    max(
                        0,
                        min(
                            self.lidar_ray_embed_sensor_index,
                            int(raster_pts.shape[0]) - 1,
                        ),
                    )
                )
                raster_sensor = raster_pts[sensor_idx].to(device=device, dtype=dtype)
                if raster_sensor.shape[-1] >= 2:
                    h_l, w_l = int(raster_sensor.shape[0]), int(raster_sensor.shape[1])
                    if h_l > 0 and w_l > 0:
                        az_cols = raster_sensor[0, :, 0]
                        row_elev = raster_sensor[:, 0, 1]
                        az_is_rad = bool(az_cols.abs().max().item() <= (math.pi + 0.5))
                        el_is_rad = bool(
                            row_elev.abs().max().item() <= (math.pi / 2.0 + 0.5)
                        )
                        az_use = az if az_is_rad else torch.rad2deg(az)
                        el_use = el if el_is_rad else torch.rad2deg(el)

                        if w_l > 1:
                            step = (az_cols[-1] - az_cols[0]) / float(max(1, w_l - 1))
                            if float(step.abs().item()) > 1e-9:
                                col_l = torch.floor((az_use - az_cols[0]) / step + 0.5).long()
                            else:
                                col_l = col_idx
                        else:
                            col_l = torch.zeros_like(col_idx)
                        col_l = col_l.clamp(0, max(0, w_l - 1))

                        if h_l > 1:
                            elev_vals = row_elev
                            ascending = bool(elev_vals[0] <= elev_vals[-1])
                            sorted_vals = elev_vals if ascending else torch.flip(elev_vals, dims=[0])
                            boundaries = 0.5 * (sorted_vals[:-1] + sorted_vals[1:])
                            row_sorted = torch.bucketize(el_use, boundaries)
                            row_l = row_sorted if ascending else (h_l - 1 - row_sorted)
                        else:
                            row_l = torch.zeros_like(row_idx)
                        row_l = row_l.clamp(0, max(0, h_l - 1))

                        row_idx = torch.floor(
                            (row_l.float() + 0.5)
                            * float(self.lidar_ray_embed_height)
                            / float(max(1, h_l))
                        ).long()
                        col_idx = torch.floor(
                            (col_l.float() + 0.5)
                            * float(self.lidar_ray_embed_width)
                            / float(max(1, w_l))
                        ).long()

        row_idx = row_idx.clamp(0, self.lidar_ray_embed_height - 1)
        col_idx = col_idx.clamp(0, self.lidar_ray_embed_width - 1)
        ray_ids = row_idx * self.lidar_ray_embed_width + col_idx
        ray_embed = self.lidar_ray_embedding(ray_ids)
        return ray_embed.to(dtype=dtype)

    def _get_lidar_range_embedding(
        self, lidar_range: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Continuous learnable embedding for distance to the LiDAR origin."""
        if (
            not self.lidar_use_range_embedding
            or self.lidar_range_embedding is None
            or not isinstance(lidar_range, torch.Tensor)
            or lidar_range.numel() <= 0
        ):
            return None
        range_flat = lidar_range.reshape(-1).to(dtype=torch.float32)
        denom = max(self.lidar_range_max - self.lidar_range_min, 1e-6)
        range_norm = (
            (range_flat - self.lidar_range_min) / denom
        ).clamp(0.0, 1.0)
        pos = range_norm * float(self.lidar_range_embed_bins - 1)
        idx0 = torch.floor(pos).long().clamp(0, self.lidar_range_embed_bins - 1)
        idx1 = (idx0 + 1).clamp(0, self.lidar_range_embed_bins - 1)
        w = (pos - idx0.to(pos.dtype)).unsqueeze(-1)
        emb0 = self.lidar_range_embedding(idx0)
        emb1 = self.lidar_range_embedding(idx1)
        emb = emb0 * (1.0 - w) + emb1 * w
        return emb.to(device=lidar_range.device, dtype=lidar_range.dtype)

    def forward(
        self,
        aggregated_tokens_list,
        images,
        patch_start_idx,
        input_dict_gs,
        output_dict_gs,
    ):
        return self.test(
            aggregated_tokens_list,
            images,
            patch_start_idx,
            input_dict_gs=input_dict_gs,
            output_dict_gs=output_dict_gs,
        )

    def test(
        self,
        aggregated_tokens_list,
        images,
        patch_start_idx,
        input_dict_gs,
        output_dict_gs,
    ):

        B, S, _, H, W = images.shape

        input_dict_gs["scene"] = aggregated_tokens_list["scene"]
        input_dict_gs["frame"] = aggregated_tokens_list["frame"]

        images_norm = (
            images - self._resnet_mean.to(images.device)
        ) / self._resnet_std.to(images.device)
        images_norm = images_norm.view(B * S, 3, H, W)
        dinov2_dict = self.image_backbone(images_norm)
        dinov2_patchtokens_raw = dinov2_dict["x_norm_patchtokens"]
        dinov2_features = self.dinov2_proj(dinov2_patchtokens_raw)

        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        # plucker_embedder
        device_id = images.device
        dtype = images.dtype
        rays_o = input_dict_gs["rays_o"].to(device_id, dtype)
        rays_d = input_dict_gs["rays_d"].to(device_id, dtype)
        pluckers = self.plucker_embedder(rays_o, rays_d)
        pluckers = pluckers.reshape(B * S, -1, H, W)
        plucker_embeds = self.plucker_to_embed(pluckers)

        ##### compute unscaled local depths #####
        local_points = aggregated_tokens_list["local_points"]
        intrinsics = input_dict_gs["intrinsics"].clone()
        local_depth = local_points[..., 2]

        # pred scale and shift
        pos = self.position_getter(
            B * S, H // self.patch_size, W // self.patch_size, images.device
        )
        out = []
        for layer_idx in self.intermediate_layer_idx:
            x = aggregated_tokens_list["intermidiate_output"][layer_idx][
                :, patch_start_idx:
            ]
            out.append(x)
        out = torch.stack(out, dim=-1)
        out = out.mean(dim=-1)
        out_hidden_patches = self.point_decoder(out, xpos=pos)
        out_hidden = out_hidden_patches.mean(dim=1)
        # Scale/shift prediction for depth adjustment
        # This is needed for pretrained models that learned these parameters
        pred_scale = self.scale_head(out_hidden).squeeze(-1).exp().reshape(B, S)
        pred_shift = self.shift_head(out_hidden).squeeze(-1).reshape(B, S)

        predicted_bilateral_grids = None
        if self.bilateral_grid_transformer is not None:
            bilateral_grid_active = bool(
                input_dict_gs.get("_bilateral_grid_active", True)
            )
            if bilateral_grid_active:
                view_cam_ids = output_dict_gs.get("view_cam_ids", None)
                view_cam_ids_2d = None
                if isinstance(view_cam_ids, torch.Tensor):
                    view_cam_ids = view_cam_ids.to(out_hidden.device)
                    if view_cam_ids.dim() == 2:
                        view_cam_ids_2d = view_cam_ids[:, -S:]
                    elif view_cam_ids.dim() == 1 and view_cam_ids.shape[0] >= B * S:
                        view_cam_ids_2d = view_cam_ids[-B * S :].view(B, S)
                if view_cam_ids_2d is None:
                    view_cam_ids_2d = (
                        torch.arange(S, device=out_hidden.device, dtype=torch.long)
                        .view(1, S)
                        .expand(B, -1)
                    )

                bilateral_raw_gt_patches = None
                bilateral_canon_patches = None
                raw_rgb_5d = output_dict_gs.get("rgb", None)
                if (
                    isinstance(raw_rgb_5d, torch.Tensor)
                    and raw_rgb_5d.dim() == 5
                    and raw_rgb_5d.shape[1] >= S
                ):
                    raw_curr = raw_rgb_5d[:B, -S:].to(
                        device=device_id, dtype=images.dtype
                    )
                    if tuple(raw_curr.shape[-2:]) != (H, W):
                        raw_curr = F.interpolate(
                            raw_curr.reshape(B * S, 3, *raw_curr.shape[-2:]),
                            size=(H, W),
                            mode="bilinear",
                            align_corners=False,
                        ).reshape(B, S, 3, H, W)
                    raw_norm = (
                        raw_curr - self._resnet_mean.to(device_id)
                    ) / self._resnet_std.to(device_id)
                    with torch.no_grad():
                        raw_dino = self.image_backbone(raw_norm.view(B * S, 3, H, W))
                    raw_patches = raw_dino["x_norm_patchtokens"]
                    bilateral_raw_gt_patches = raw_patches.view(
                        B, S, raw_patches.shape[1], raw_patches.shape[2]
                    )
                    bilateral_canon_patches = dinov2_patchtokens_raw.view(
                        B,
                        S,
                        dinov2_patchtokens_raw.shape[1],
                        dinov2_patchtokens_raw.shape[2],
                    )

                if bilateral_raw_gt_patches is None or bilateral_canon_patches is None:
                    raise RuntimeError(
                        "CHROMA bilateral-grid head requires current-frame raw RGB "
                        "and canonical patch tokens, but required inputs were missing."
                    )

                predicted_bilateral_grids = self.bilateral_grid_transformer(
                    raw_gt_patches=bilateral_raw_gt_patches,
                    canon_patches=bilateral_canon_patches,
                    view_cam_ids_2d=view_cam_ids_2d,
                    patch_h=patch_h,
                    patch_w=patch_w,
                )

        depth_map_unscale = (
            local_depth * pred_scale[:, :, None, None] + pred_shift[:, :, None, None]
        )
        
        # 保存 scale/shift 调整后的深度（用于深度监督），在 sky fill 和 clamp 之前
        depth_map_unscale_pred = depth_map_unscale.clone()  # [B, S, H, W]
        
        ##### compute unscaled local depths #####
        sky_mask_unscale = input_dict_gs["sky_mask"]
        # give sky depth: 400m for sky; 450m for side cam padding only
        sky_depth = self.cfg.get("sky_mask", 0.0)
        side_padding_depth = self.cfg.get("sky_mask_side_cameras", 0.0)
        max_depth = max(sky_depth, side_padding_depth) if (sky_depth > 0 or side_padding_depth > 0) else 300.0
        if sky_depth > 0:
            sky_invalid = sky_mask_unscale == 0
            depth_map_unscale[sky_invalid] = sky_depth
            # Side cam (3,4) padding: valid_mask=False -> 450m
            if side_padding_depth > 0 and "valid_mask" in output_dict_gs:
                vm = output_dict_gs["valid_mask"]
                if isinstance(vm, torch.Tensor) and vm.numel() > 0:
                    vm = vm.to(device=depth_map_unscale.device, dtype=torch.bool)
                    if vm.dim() == 4:
                        valid_current = vm[:, -S:, :, :]
                    else:
                        valid_current = vm[-S:, :, :].unsqueeze(0)
                    if valid_current.shape[0] == 1 and B > 1:
                        valid_current = valid_current.expand(B, -1, -1, -1)
                    if valid_current.shape[-2:] != depth_map_unscale.shape[-2:]:
                        # `valid_mask` may be at render resolution while depth/sky
                        # are in backbone-input resolution. Align before boolean ops.
                        valid_current = (
                            F.interpolate(
                                valid_current.float(),
                                size=depth_map_unscale.shape[-2:],
                                mode="nearest",
                            )
                            > 0.5
                        )
                    # padding = ~valid on cam 3,4
                    side_padding_mask = ~valid_current
                    side_padding_mask[:, [0, 1, 2], :, :] = False
                    if S > 5:
                        side_padding_mask[:, 5:, :, :] = False
                    depth_map_unscale[sky_invalid & side_padding_mask] = side_padding_depth
        depth_map_unscale = torch.clamp(
            depth_map_unscale, max=max_depth, min=0.0
        )
        depthmap = torch.cat(
            [
                depth_map_unscale[:, :, None, :, :] / max_depth,
                sky_mask_unscale[:, :, None, :, :],
                images,
            ],
            dim=2,
        )
        depthmap = depthmap.reshape(B * S, -1, H, W)
        depth_embeds = self.depth_embeds(depthmap)
        # aggregrate embeds
        agg_embeds = plucker_embeds + depth_embeds
        agg_embeds = self.embed_proj(agg_embeds)
        agg_embeds = agg_embeds.reshape(B * S, patch_h * patch_w, -1)

        out = []
        dpt_idx = 0
        for layer_idx in self.intermediate_layer_idx:
            x = aggregated_tokens_list["intermidiate_output"][layer_idx][
                :, patch_start_idx:
            ]
            x = x + agg_embeds + dinov2_features
            x = x.view(B * S, -1, x.shape[-1])
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)
            out.append(x)
            dpt_idx += 1
        # FPN features
        fpn_out_project = self.voxel_proj(out[0])
        # Fuse features from multiple layers.
        out = self.scratch_forward(out)
        out = custom_interpolate(
            out,
            (int(patch_h * self.patch_size), int(patch_w * self.patch_size)),
            mode="bilinear",
            align_corners=True,
        )
        img_guassian_features = self.feature_norm(out)

        # Compute 3D point positions in LiDAR coordinate system
        # Using rays_o + rays_d * depth (original method)
        # rays_o: [B, S, H, W, 3] - ray origins in LiDAR coords
        # rays_d: [B, S, H, W, 3] - ray directions in LiDAR coords
        rays_o = input_dict_gs["rays_o"].to(device_id, dtype)
        rays_d = input_dict_gs["rays_d"].to(device_id, dtype)
        means = rays_o + rays_d * depth_map_unscale[:, :, :, :, None]  # [B, S, H, W, 3]
        pixel_means = means.reshape(B, -1, 3)  # [B, S*H*W, 3]
        # Dense geometric attribute maps for overlap_geo loss (optional, enabled by train loop).
        attr_xyz_map = means

        need_geometric_loss = bool(input_dict_gs.get("_need_geometric_loss", False))
        pixel_surf_normals = None
        attr_normal_map = None
        if need_geometric_loss:
            dy = means[:, :, 2:, 1:-1, :] - means[:, :, :-2, 1:-1, :]
            dx = means[:, :, 1:-1, 2:, :] - means[:, :, 1:-1, :-2, :]
            surf_normals = torch.nn.functional.normalize(torch.cross(dy, dx, dim=-1), dim=-1)
            surf_normals = torch.nn.functional.pad(surf_normals.permute(0, 1, 4, 2, 3), (1, 1, 1, 1)).permute(0, 1, 3, 4, 2)
            pixel_surf_normals = surf_normals.reshape(B, -1, 3)  # [B, S*H*W, 3]
            attr_normal_map = surf_normals

        # each step select some view to render, reduce computation and memory
        out_view_num = output_dict_gs["rgb"].shape[1]
        select_mask = torch.zeros((out_view_num), dtype=torch.bool, device=device_id)
        select_mask[:] = True
        select_index = torch.nonzero(select_mask, as_tuple=False).squeeze(-1)
        # render parameters
        render_c2w = output_dict_gs["c2w"].to(device_id, dtype)
        render_fovxs = output_dict_gs["fovx"].to(device_id, dtype)
        render_fovys = output_dict_gs["fovy"].to(device_id, dtype)
        rgb_gt = output_dict_gs["rgb"].to(device_id, dtype)
        out_intrinsics = output_dict_gs["intrinsics"].to(device_id, dtype)

        # init voxel
        self.pts_range = self.pts_range.to(device=means.device)
        self.voxel_size = self.voxel_size.to(device=means.device)
        # valid_mask: exclude padding pixels (e.g. side camera black bars) from voxel cloud
        valid_flat = None
        if "valid_mask" in output_dict_gs:
            vm = output_dict_gs["valid_mask"]
            if isinstance(vm, torch.Tensor) and vm.numel() > 0:
                vm = vm.to(device=device_id, dtype=torch.bool)
                # vm: (B, V, H, W) or (V, H, W); use last S views (current frame)
                if vm.dim() == 4:
                    valid_current = vm[:, -S:, :, :]  # [B, S, H, W]
                else:
                    valid_current = vm[-S:, :, :].unsqueeze(0)  # [1, S, H, W]
                if valid_current.shape[0] == 1 and B > 1:
                    valid_current = valid_current.expand(B, -1, -1, -1)
                if valid_current.shape[-2:] != (H, W):
                    valid_current = (
                        F.interpolate(
                            valid_current.float(),
                            size=(H, W),
                            mode="nearest",
                        )
                        > 0.5
                    )
                valid_flat = valid_current.reshape(valid_current.shape[0], -1)  # [B, S*H*W]

        coords = []
        multimodel_gaussians = []
        for i in range(B):
            means = pixel_means[i]
            boundary = self.pts_range.clone()
            boundary[:3] = boundary[:3] + 0.01
            boundary[3:6] = boundary[3:6] - 0.01

            # use a more strict boundary to filter out points
            mask = (
                (means[:, 0] > boundary[0])
                & (means[:, 0] < boundary[3])
                & (means[:, 1] > boundary[1])
                & (means[:, 1] < boundary[4])
                & (means[:, 2] > boundary[2])
                & (means[:, 2] < boundary[5])
            )
            # exclude padding pixels (Stage 1: prevent black padding from contributing to voxels)
            if valid_flat is not None:
                mask = mask & valid_flat[i]
            count = mask.sum()
            # Fallback: when all points filtered out, add dummy point at scene center to avoid empty voxel
            if count == 0:
                center = (self.pts_range[:3] + self.pts_range[3:6]) / 2
                dummy_rgb = means.new_tensor([[0.5, 0.5, 0.5]], dtype=means.dtype)
                batch_idx = means.new_zeros((1, 1), dtype=torch.long) + i
                gaussians_coords = (center - self.pts_range[:3]) / self.voxel_size[:3]
                gaussians_coords = torch.floor(gaussians_coords).int().unsqueeze(0)
                coords.append(torch.cat([batch_idx, gaussians_coords], dim=-1))
                multimodel_gaussians.append(torch.cat([center.unsqueeze(0), dummy_rgb], dim=-1))
            else:
                # print(f"[DEBUG VOXEL] Points in boundary: {count} / {means.shape[0]} ({100*count/means.shape[0]:.2f}%)")
                batch_idx = means.new_zeros((count, 1)) + i
                batch_idx = batch_idx.int()
                gaussians_coords = means[mask]
                gaussians_coords = gaussians_coords - self.pts_range[None, :3]
                gaussians_coords = torch.floor(
                    gaussians_coords / self.voxel_size[None, :3]
                ).int()
                coords.append(torch.cat([batch_idx, gaussians_coords], dim=-1))
                rgbs = images[i].permute(0, 2, 3, 1).reshape(-1, 3)
                multimodel_gaussians.append(torch.cat([means[mask], rgbs[mask]], dim=-1))

        coords = torch.cat(coords, dim=0)
        multimodel_gaussians = torch.cat(multimodel_gaussians, dim=0)
        merge_coords = (
            coords[:, 0].int() * self.scale_xyz
            + coords[:, 1] * self.scale_yz
            + coords[:, 2] * self.scale_z
            + coords[:, 3]
        )
        unq_coords, unq_inv = torch.unique(
            merge_coords, return_inverse=True, return_counts=False, dim=0
        )
        # torch_scatter is optional; use torch-only scatter_mean if not installed.
        if torch_scatter is not None:
            # Calculate num_unique for dim_size to ensure consistent output size
            num_unique = int(unq_inv.max().item()) + 1 if unq_inv.numel() > 0 else 0
            gaussians_voxel = torch_scatter.scatter_mean(
                multimodel_gaussians, unq_inv, dim=0, dim_size=num_unique
            )
        else:
            # unq_inv: (N,) mapping each row of multimodel_gaussians -> [0, num_unique)
            # multimodel_gaussians: (N, C)
            num_unique = int(unq_inv.max().item()) + 1 if unq_inv.numel() > 0 else 0
            gaussians_voxel_sum = multimodel_gaussians.new_zeros(
                (num_unique, multimodel_gaussians.shape[1])
            )
            gaussians_voxel_sum.index_add_(0, unq_inv, multimodel_gaussians)
            counts = multimodel_gaussians.new_zeros((num_unique,))
            counts.index_add_(
                0, unq_inv, torch.ones_like(unq_inv, dtype=multimodel_gaussians.dtype)
            )
            gaussians_voxel = gaussians_voxel_sum / counts.clamp_min(1e-6).unsqueeze(-1)

        unq_coords = unq_coords.int()
        voxel_coords = torch.stack(
            (
                unq_coords // self.scale_xyz,
                (unq_coords % self.scale_xyz) // self.scale_yz,
                (unq_coords % self.scale_yz) // self.scale_z,
                unq_coords % self.scale_z,
            ),
            dim=1,
        )
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]  # bzyx
        
        # Check and fix dimension consistency if needed
        if gaussians_voxel.shape[0] != voxel_coords.shape[0]:
            # This should not happen, but handle gracefully to avoid crash
            print(f"[WARNING] Dimension mismatch: gaussians_voxel={gaussians_voxel.shape}, voxel_coords={voxel_coords.shape}")
            min_voxels = min(gaussians_voxel.shape[0], voxel_coords.shape[0])
            if min_voxels == 0:
                # Empty case - will be caught by empty check below
                pass
            else:
                # Truncate both to minimum size to maintain consistency
                gaussians_voxel = gaussians_voxel[:min_voxels]
                voxel_coords = voxel_coords[:min_voxels]
                print(f"[WARNING] Truncated to min_voxels={min_voxels}")
        
        voxel_centers = get_voxel_centers(
            voxel_coords[:, 1:], 1, self.voxel_size, self.pts_range
        )

        # extract image_features
        intrinsics = input_dict_gs["intrinsics"]
        camera2lidar = input_dict_gs["camera2lidar"]
        # Stage2: valid_mask for feature sampling - exclude padding regions when averaging across views
        valid_mask_hw = None
        if "valid_mask" in output_dict_gs:
            vm = output_dict_gs["valid_mask"]
            if isinstance(vm, torch.Tensor) and vm.numel() > 0:
                vm = vm.to(device=device_id, dtype=torch.bool)
                if vm.dim() == 4:
                    valid_mask_hw = vm[:, -S:, :, :]  # [B, S, H, W]
                else:
                    valid_mask_hw = vm[-S:, :, :].unsqueeze(0)  # [1, S, H, W]
                if valid_mask_hw.shape[0] == 1 and B > 1:
                    valid_mask_hw = valid_mask_hw.expand(B, -1, -1, -1)
                if valid_mask_hw.shape[-2:] != (H, W):
                    valid_mask_hw = (
                        F.interpolate(
                            valid_mask_hw.float(),
                            size=(H, W),
                            mode="nearest",
                        )
                        > 0.5
                    )

        gaussians_imgfeats = []
        for i in range(B):
            batch_mask = voxel_coords[:, 0] == i
            points = voxel_centers[batch_mask]
            projected_points, z_cam, valid_masks = project_world_points_to_images(
                points, intrinsics[i], camera2lidar[i], H, W
            )
            # Stage2: exclude projections that land in padding regions (e.g. side cam black bars)
            if valid_mask_hw is not None and points.shape[0] > 0:
                N_pts = points.shape[0]
                u_idx = projected_points[:, :, 0].long().clamp(0, W - 1)  # (S, N)
                v_idx = projected_points[:, :, 1].long().clamp(0, H - 1)  # (S, N)
                view_idx = torch.arange(S, device=points.device).unsqueeze(1).expand(
                    -1, N_pts
                )  # (S, N)
                valid_non_pad = valid_mask_hw[i, view_idx, v_idx, u_idx]  # (S, N)
                valid_masks = valid_masks & valid_non_pad

            image_modality = fpn_out_project
            projected_points[:, :, 0] = (projected_points[:, :, 0] / W) * 2 - 1
            projected_points[:, :, 1] = (projected_points[:, :, 1] / H) * 2 - 1
            out_features = image_modality.new_zeros(
                (
                    image_modality.shape[0],
                    projected_points.shape[1],
                    image_modality.shape[1],
                )
            )
            for view in range(valid_masks.shape[0]):
                input_features = image_modality[i * S + view : i * S + view + 1]
                input_points = projected_points[view : view + 1]
                input_mask = valid_masks[view : view + 1]
                valid_points = input_points[input_mask]
                tmp_features = F.grid_sample(
                    input_features,
                    valid_points[None, None, :, :],
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )
                # AMP can make out_features bfloat16/float16 while grid_sample may output float32.
                # Ensure dtype matches for indexed assignment.
                sampled = tmp_features[0, :, 0, :].transpose(0, 1)
                if sampled.dtype != out_features.dtype:
                    sampled = sampled.to(out_features.dtype)
                out_features[view][input_mask[0]] = sampled
            out_features = out_features.sum(dim=0) / (
                valid_masks.sum(dim=0)[:, None] + 1e-6
            )
            gaussians_imgfeats.append(out_features)
        gaussians_imgfeats = torch.cat(gaussians_imgfeats, dim=0)
        gaussians_voxel = torch.cat([gaussians_voxel, gaussians_imgfeats], dim=-1)

        # get history
        (
            _,
            _,
            _,
            _,
            _,
            history_lidar2world,
            history_features,
            history_poses,
            voxel_opacities,
            history_gaussians,
            history_gaussians_outview,
        ) = self.history_queue.get(
            B, aggregated_tokens_list["scene"], aggregated_tokens_list["frame"]
        )
        current_lidar2world = aggregated_tokens_list["lidar2world"]
        (
            process_voxel_features,
            process_voxel_coords,
            process_voxel_pos,
            process_masks,
        ) = ([], [], [], [])
        for i in range(B):
            if history_poses[i] is None:
                # no history, construct a dummy history
                # Use current frame's voxels as reference, but ensure dimension consistency
                current_frame_poses = voxel_centers[voxel_coords[:, 0] == i]
                if current_frame_poses.shape[0] > 0:
                    # Use first voxel from current frame
                    tmp_history_poses = current_frame_poses[:1]
                    tmp_history_features = torch.zeros(
                        (1, 64), dtype=gaussians_voxel.dtype, device=gaussians_voxel.device
                    )
                    tmp_mask = torch.tensor([False], dtype=torch.bool, device=gaussians_voxel.device)
                else:
                    # No voxels in current frame - create minimal dummy
                    tmp_history_poses = torch.zeros(
                        (1, 3), dtype=gaussians_voxel.dtype, device=gaussians_voxel.device
                    )
                    tmp_history_features = torch.zeros(
                        (1, 64), dtype=gaussians_voxel.dtype, device=gaussians_voxel.device
                    )
                    tmp_mask = torch.tensor([False], dtype=torch.bool, device=gaussians_voxel.device)
                
                tmp_history_coords = tmp_history_poses - self.pts_range[None, :3]
                tmp_history_coords = torch.floor(
                    tmp_history_coords / (self.voxel_size[None, :3] * 2)
                ).int()
                bidx = (
                    tmp_history_coords.new_zeros((tmp_history_coords.shape[0], 1)) + i
                )
                tmp_history_coords = torch.cat([bidx.int(), tmp_history_coords], dim=-1)
            else:
                history_2_current = (
                    current_lidar2world[i].inverse() @ history_lidar2world[i]
                )[0]
                ones = torch.ones(
                    (history_poses[i].shape[0], 1),
                    dtype=images.dtype,
                    device=images.device,
                )
                history_poses_homo = torch.concat([history_poses[i], ones], dim=-1)
                history_poses_homo = (history_2_current @ history_poses_homo.T).T
                history_poses_homo = history_poses_homo[:, :-1]
                tmp_mask = (
                    (history_poses_homo[:, 0] > boundary[0])
                    & (history_poses_homo[:, 0] < boundary[3])
                    & (history_poses_homo[:, 1] > boundary[1])
                    & (history_poses_homo[:, 1] < boundary[4])
                    & (history_poses_homo[:, 2] > boundary[2])
                    & (history_poses_homo[:, 2] < boundary[5])
                ).to(images.device)

                tmp_history_poses = history_poses_homo[tmp_mask]
                tmp_history_features = history_features[i][tmp_mask]
                new_mask, _ = is_point_in_frustum_batch(
                    tmp_history_poses,
                    intrinsics[i],
                    camera2lidar[i, :, :3, :3],
                    camera2lidar[i, :, :3, 3],
                    0.5,
                    72,
                    518,
                    350,
                )
                new_mask = new_mask.any(dim=-1)
                # in view filter
                inview_opacities = voxel_opacities[i][tmp_mask][new_mask]
                inview_features = tmp_history_features[new_mask]
                inview_poses = tmp_history_poses[new_mask]
                # 边界保护：训练初期预测不准时，可能所有历史体素都被过滤掉
                if inview_opacities.shape[0] > 0:
                    voxel_stream_quantile = min(
                        self.cfg["voxel_stream_quantile"],
                        self.cfg["voxel_stream_inview_num"] / inview_opacities.shape[0],
                    )
                    opacities_thre = torch.quantile(
                        inview_opacities, 1 - voxel_stream_quantile
                    )
                    opacities_thre = max(opacities_thre, self.cfg.voxel_stream_opacity_thre)
                    opacities_mask = inview_opacities > opacities_thre
                    if opacities_mask.sum() == 0:
                        # at least keep one
                        opacities_mask[0] = True
                    inview_features = inview_features[opacities_mask]
                    inview_poses = inview_poses[opacities_mask]
                # out view filter
                if (~new_mask).sum() == 0:
                    outview_features = torch.zeros(
                        (0, tmp_history_features.shape[1]),
                        dtype=tmp_history_features.dtype,
                        device=tmp_history_features.device,
                    )
                    outview_poses = torch.zeros(
                        (0, 3),
                        dtype=tmp_history_poses.dtype,
                        device=tmp_history_poses.device,
                    )
                else:
                    outview_opacities = voxel_opacities[i][tmp_mask][~new_mask]
                    outview_features = tmp_history_features[~new_mask]
                    outview_poses = tmp_history_poses[~new_mask]
                    voxel_stream_quantile = min(
                        self.cfg["voxel_stream_quantile_outview"],
                        self.cfg["voxel_stream_outview_num"]
                        / outview_opacities.shape[0],
                    )
                    opacities_thre = torch.quantile(
                        outview_opacities, 1 - voxel_stream_quantile
                    )
                    opacities_thre = max(
                        opacities_thre, self.cfg.voxel_stream_opacity_thre
                    )
                    opacities_mask = outview_opacities > opacities_thre
                    if opacities_mask.sum() == 0:
                        # at least keep one
                        opacities_mask[0] = True
                    outview_features = outview_features[opacities_mask]
                    outview_poses = outview_poses[opacities_mask]

                tmp_history_features = torch.cat(
                    [inview_features, outview_features], dim=0
                )
                tmp_history_poses = torch.cat([inview_poses, outview_poses], dim=0)
                tmp_history_coords = tmp_history_poses - self.pts_range[None, :3]
                tmp_history_coords = torch.floor(
                    tmp_history_coords / (self.voxel_size[None, :3] * 2)
                ).int()
                bidx = (
                    tmp_history_coords.new_zeros((tmp_history_coords.shape[0], 1)) + i
                )
                tmp_history_coords = torch.cat([bidx.int(), tmp_history_coords], dim=-1)
                tmp_mask = torch.ones(tmp_history_coords.shape[0], dtype=torch.bool, device=images.device)
            process_voxel_features.append(tmp_history_features)
            process_voxel_coords.append(tmp_history_coords)
            process_voxel_pos.append(tmp_history_poses)
            process_masks.append(tmp_mask)

        process_voxel_features = torch.cat(process_voxel_features, dim=0)
        process_voxel_coords = torch.cat(process_voxel_coords, dim=0)
        process_voxel_coords = process_voxel_coords[:, [0, 3, 2, 1]]  # bzyx
        process_voxel_pos = torch.cat(process_voxel_pos, dim=0)
        process_masks = torch.cat(process_masks, dim=0).to(images.device)
        
        # Check dimension consistency for history_infos (critical for sparse_add)
        if process_voxel_features.shape[0] != process_voxel_coords.shape[0]:
            print(f"[WARNING] History voxel dimension mismatch: features={process_voxel_features.shape}, coords={process_voxel_coords.shape}")
            min_history = min(process_voxel_features.shape[0], process_voxel_coords.shape[0])
            if min_history > 0:
                process_voxel_features = process_voxel_features[:min_history]
                process_voxel_coords = process_voxel_coords[:min_history]
                process_voxel_pos = process_voxel_pos[:min_history]
                process_masks = process_masks[:min_history]
                print(f"[WARNING] Truncated history to min_history={min_history}")
            else:
                # Empty history - create dummy
                process_voxel_features = torch.zeros((1, 64), dtype=gaussians_voxel.dtype, device=gaussians_voxel.device)
                process_voxel_coords = torch.zeros((1, 4), dtype=torch.int32, device=gaussians_voxel.device)
                process_voxel_pos = torch.zeros((1, 3), dtype=gaussians_voxel.dtype, device=gaussians_voxel.device)
                process_masks = torch.zeros((1,), dtype=torch.bool, device=gaussians_voxel.device)
                print(f"[WARNING] Created dummy history due to empty mismatch")
        
        history_infos = (
            process_voxel_features,
            process_voxel_coords,
            process_voxel_pos,
            process_masks,
        )
        
        # Check for empty voxel input to avoid spconv crash
        if gaussians_voxel.shape[0] == 0 or voxel_coords.shape[0] == 0:
            raise RuntimeError(
                f"[ERROR] Empty voxel input detected: gaussians_voxel={gaussians_voxel.shape}, "
                f"voxel_coords={voxel_coords.shape}, batch={B}. "
                f"This usually happens when all Gaussians are filtered out. "
                f"Please check your data and filtering thresholds."
            )
        
        # Check dimension consistency between features and coordinates
        if gaussians_voxel.shape[0] != voxel_coords.shape[0]:
            raise RuntimeError(
                f"[ERROR] Voxel features/coords dimension mismatch: "
                f"gaussians_voxel={gaussians_voxel.shape}, voxel_coords={voxel_coords.shape}. "
                f"This should not happen - they must have the same number of rows (voxels)."
            )
        
        gaussians_voxel, point_coords, save_features, save_coords = self.unet(
            gaussians_voxel, voxel_coords, B, history_infos
        )
        gaussians_voxel, voxel_batch_idx, voxel_opacities = self.process_guassian_voxel(
            gaussians_voxel, point_coords[:, 1:], voxel_batch_idx=point_coords[:, 0]
        )

        # final render parts
        sky_mask = sky_mask_unscale.reshape(B, S, H, W) == 0
        img_guassian_features = img_guassian_features.reshape(B, S, -1, H, W)
        save_gaussians, save_gaussians_outview = [], []
        need_fisher_data = bool(input_dict_gs.get("_need_fisher_data", False))
        fisher_gaussians = []
        fisher_semantics = []
        fisher_render_inputs = []
        batch_render_outputs = []  # Accumulate render outputs for all batches
        batch_render_pixel = []    # 点分支独立渲染输出
        batch_render_volume = []   # 体素分支独立渲染输出
        lidar_supervision_batch = input_dict_gs.get("lidar_supervision", None)
        lidar_image_supervision_batch = input_dict_gs.get("lidar_image_supervision", None)
        lidar_intensity_plane = str(
            input_dict_gs.get("_lidar_intensity_supervision_plane", "image")
            or "image"
        ).lower()
        supervise_on_lidar_plane = lidar_intensity_plane == "lidar"
        # Spherical LiDAR render is only required by native LiDAR-plane
        # supervision ("lidar"). In "both" mode the extra LiDAR-plane term is
        # built from camera_projection remapping of image-plane predictions.
        train_lidar_plane_loss = lidar_intensity_plane == "lidar"
        # `render_lidar` serves two paths:
        #  - lidar-plane intensity supervision (with grad), or
        #  - visualization-only spherical debug maps (no grad on vis iterations).
        need_lidar_vis = bool(input_dict_gs.get("_need_lidar_vis", False))
        batch_lidar_render = []
        batch_lidar_image_pred = []
        
        batch_loss_orient_raw = []
        batch_loss_flat_raw = []
        batch_loss_orient_means = []
        batch_loss_orient_is_voxel = []
        need_overlap_geo_attr = bool(input_dict_gs.get("_need_overlap_gaussian_geo", False))
        batch_attr_xyz_maps = [] if need_overlap_geo_attr else None
        batch_attr_scale_maps = [] if need_overlap_geo_attr else None
        batch_attr_rgb_maps = [] if need_overlap_geo_attr else None
        
        for i in range(B):
            valid_mask_i = valid_flat[i] if valid_flat is not None else None
            S_i, H_i, W_i = sky_mask[i].shape
            cam_indices_i = torch.arange(
                S_i * H_i * W_i, device=pixel_means[i].device, dtype=torch.long
            ) // (H_i * W_i)
            extra_pixel_feats_i = None
            if self.lidar_use_image_encoder or self.lidar_use_rgb_input:
                # Per-pixel image-derived inputs at the point-grid resolution
                # (matches img_guassian_features), aligned per source pixel below.
                target_hw = img_guassian_features[i].shape[-2:]
                img_i = images[i].to(img_guassian_features.dtype)
                _extra_maps = []
                if self.lidar_use_image_encoder and self.lidar_image_encoder is not None:
                    _extra_maps.append(self.lidar_image_encoder(img_i, target_hw))
                if self.lidar_use_rgb_input:
                    _extra_maps.append(
                        F.interpolate(
                            img_i,
                            size=tuple(target_hw),
                            mode="bilinear",
                            align_corners=False,
                        )
                    )
                if len(_extra_maps) > 0:
                    extra_pixel_feats_i = torch.cat(_extra_maps, dim=1)
            refine_out = self.refine_guassians(
                save_features[save_coords[:, 0] == i],
                save_coords[save_coords[:, 0] == i],
                pixel_means[i],
                sky_mask[i],
                img_guassian_features[i],
                valid_mask=valid_mask_i,
                cam_indices=cam_indices_i,
                rays_d=rays_d[i],
                extra_pixel_feats=extra_pixel_feats_i,
            )
            (
                gaussians_source1,
                gaussians_source1_dynscore,
                gaussians_sky,
                gaussians_sky_dynscore,
                sample_features,
                pt_sky_mask,
                sky_means_refined,
                lidar_img_feat_per_point,
            ) = refine_out

            # Build dense per-pixel scale map [S, H, W, 3] for overlap_geo loss.
            # Point branch attributes map to non-sky pixels; sky branch attributes map to sky pixels.
            if need_overlap_geo_attr:
                num_pixels_i = S_i * H_i * W_i
                xyz_map_flat = attr_xyz_map[i].reshape(num_pixels_i, 3).clone()
                scale_map_flat = attr_xyz_map.new_zeros(num_pixels_i, 3)
                rgb_map_flat = attr_xyz_map.new_zeros(num_pixels_i, 3)
                pt_scales = gaussians_source1[:, 11:14]
                sky_scales = gaussians_sky[:, 11:14]
                pt_rgbs = gaussians_source1[:, 3:6]
                sky_rgbs = gaussians_sky[:, 3:6]
                sky_xyz = (
                    sky_means_refined
                    if isinstance(sky_means_refined, torch.Tensor)
                    else None
                )

                # Case A: refine_guassians kept full pixels (pt_sky_mask length == S*H*W).
                if pt_sky_mask.numel() == num_pixels_i:
                    non_sky_idx = torch.nonzero(~pt_sky_mask, as_tuple=False).reshape(-1)
                    sky_idx = torch.nonzero(pt_sky_mask, as_tuple=False).reshape(-1)
                    n_pt = min(non_sky_idx.numel(), pt_scales.shape[0])
                    n_sky = min(sky_idx.numel(), sky_scales.shape[0])
                    if n_pt > 0:
                        scale_map_flat[non_sky_idx[:n_pt]] = pt_scales[:n_pt]
                        rgb_map_flat[non_sky_idx[:n_pt]] = pt_rgbs[:n_pt]
                    if n_sky > 0:
                        scale_map_flat[sky_idx[:n_sky]] = sky_scales[:n_sky]
                        rgb_map_flat[sky_idx[:n_sky]] = sky_rgbs[:n_sky]
                    if sky_xyz is not None and sky_xyz.numel() > 0:
                        n_sky_xyz = min(sky_idx.numel(), sky_xyz.shape[0])
                        if n_sky_xyz > 0:
                            xyz_map_flat[sky_idx[:n_sky_xyz]] = sky_xyz[:n_sky_xyz]
                # Case B: refine_guassians filtered with valid_mask (pt_sky_mask length == num_valid).
                elif (
                    valid_mask_i is not None
                    and valid_mask_i.numel() == num_pixels_i
                    and pt_sky_mask.numel() == int(valid_mask_i.sum().item())
                ):
                    valid_idx = torch.nonzero(valid_mask_i, as_tuple=False).reshape(-1)
                    scale_valid = attr_xyz_map.new_zeros(pt_sky_mask.numel(), 3)
                    rgb_valid = attr_xyz_map.new_zeros(pt_sky_mask.numel(), 3)
                    non_sky_idx = torch.nonzero(~pt_sky_mask, as_tuple=False).reshape(-1)
                    sky_idx = torch.nonzero(pt_sky_mask, as_tuple=False).reshape(-1)
                    n_pt = min(non_sky_idx.numel(), pt_scales.shape[0])
                    n_sky = min(sky_idx.numel(), sky_scales.shape[0])
                    if n_pt > 0:
                        scale_valid[non_sky_idx[:n_pt]] = pt_scales[:n_pt]
                        rgb_valid[non_sky_idx[:n_pt]] = pt_rgbs[:n_pt]
                    if n_sky > 0:
                        scale_valid[sky_idx[:n_sky]] = sky_scales[:n_sky]
                        rgb_valid[sky_idx[:n_sky]] = sky_rgbs[:n_sky]
                    n_valid = min(valid_idx.numel(), scale_valid.shape[0])
                    if n_valid > 0:
                        scale_map_flat[valid_idx[:n_valid]] = scale_valid[:n_valid]
                        rgb_map_flat[valid_idx[:n_valid]] = rgb_valid[:n_valid]
                    if sky_xyz is not None and sky_xyz.numel() > 0:
                        n_sky_xyz = min(sky_idx.numel(), sky_xyz.shape[0])
                        if n_sky_xyz > 0:
                            sky_global_idx = valid_idx[sky_idx[:n_sky_xyz]]
                            xyz_map_flat[sky_global_idx] = sky_xyz[:n_sky_xyz]

                batch_attr_xyz_maps.append(xyz_map_flat.view(S_i, H_i, W_i, 3))
                batch_attr_scale_maps.append(scale_map_flat.view(S_i, H_i, W_i, 3))
                batch_attr_rgb_maps.append(rgb_map_flat.view(S_i, H_i, W_i, 3))

            # --- compute per-Gaussian geometric loss raw data ---
            if need_geometric_loss and gaussians_source1.shape[0] > 0:
                rotations = gaussians_source1[:, 7:11]
                scales = gaussians_source1[:, 11:14]
                gs_normals = gaussian_orientation_from_scales(rotations, scales)
                valid_surf_normals = pixel_surf_normals[i].view(-1, 3)[~pt_sky_mask]
                dot = (valid_surf_normals * gs_normals).sum(-1).abs()
                batch_loss_orient_raw.append(1.0 - dot)
                batch_loss_flat_raw.append(scales.min(dim=-1).values)
                # Orient/flat raw losses are built from gaussians_source1 (non-sky point branch),
                # i.e. no voxel branch entries at this stage.
                batch_loss_orient_means.append(gaussians_source1[:, :3].detach())
                batch_loss_orient_is_voxel.append(
                    torch.zeros(
                        gaussians_source1.shape[0],
                        device=device_id,
                        dtype=torch.bool,
                    )
                )
            else:
                batch_loss_orient_raw.append(None)
                batch_loss_flat_raw.append(None)
                batch_loss_orient_means.append(None)
                batch_loss_orient_is_voxel.append(None)

            # process memory gaussians
            extra_gaussians_outview = self.process_history_gaussians(
                history_gaussians[i],
                history_gaussians_outview[i],
                current_lidar2world[i],
                history_lidar2world[i],
                intrinsics[i],
                camera2lidar[i],
                gaussians_source1[:, :3].detach(),
                W,
                H,
            )
            save_gaussians_outview.append(extra_gaussians_outview.detach())
            # add sky
            gaussians_source1 = torch.cat([gaussians_source1, gaussians_sky], dim=0)
            gaussians_source1_dynscore = torch.cat(
                [gaussians_source1_dynscore, gaussians_sky_dynscore], dim=0
            )
            batch_mask = voxel_batch_idx == i
            gaussians_source2 = gaussians_voxel[batch_mask][:, :14]
            gaussians_source2_dynscore = gaussians_voxel[batch_mask][:, 14:]

            H_stack = [rgb_gt.shape[-2]] * rgb_gt.shape[1]
            W_stack = [rgb_gt.shape[-1]] * rgb_gt.shape[1]
            H_stack = torch.tensor(H_stack, dtype=torch.int, device=device_id)
            W_stack = torch.tensor(W_stack, dtype=torch.int, device=device_id)

            # 是否禁用体素分支（通过 config 控制）
            use_voxel_branch = self.cfg.get("use_voxel_branch", True)
            if use_voxel_branch:
                gaussians_all = torch.cat([gaussians_source1, gaussians_source2], dim=0)
                gaussians_all_dynscore = torch.cat(
                    [gaussians_source1_dynscore, gaussians_source2_dynscore], dim=0
                )
            else:
                gaussians_all = gaussians_source1
                gaussians_all_dynscore = gaussians_source1_dynscore

            # 分支独立渲染（用于分支独立 loss 监督）
            branch_separate_render = self.cfg.get("branch_separate_render", False)
            if branch_separate_render and use_voxel_branch:
                # 点分支独立渲染
                tmp_pixel = self.renderer.render(
                    gaussians=gaussians_source1[:, :],
                    c2w=render_c2w[i, select_index],
                    fovx=render_fovxs[i, select_index],
                    fovy=render_fovys[i, select_index],
                    rays_o=None, rays_d=None,
                    K=out_intrinsics[i, select_index],
                    H=H_stack[select_index],
                    W=W_stack[select_index],
                    semantics=gaussians_source1_dynscore,
                )
                batch_render_pixel.append(tmp_pixel)
                # 体素分支独立渲染
                tmp_volume = self.renderer.render(
                    gaussians=gaussians_source2[:, :],
                    c2w=render_c2w[i, select_index],
                    fovx=render_fovxs[i, select_index],
                    fovy=render_fovys[i, select_index],
                    rays_o=None, rays_d=None,
                    K=out_intrinsics[i, select_index],
                    H=H_stack[select_index],
                    W=W_stack[select_index],
                    semantics=gaussians_source2_dynscore,
                )
                batch_render_volume.append(tmp_volume)
            # if i == 0:  # 只打印 batch 0
            #     # 分析两个分支的属性差异
            #     pt_opacity = gaussians_source1[:, 6:7]  # [N, 1]
            #     pt_scales = gaussians_source1[:, 11:14]  # [N, 3]
            #     vx_opacity = gaussians_source2[:, 6:7]
            #     vx_scales = gaussians_source2[:, 11:14]
            #     print(f"[GS] Batch {i}: point={gaussians_source1.shape[0]}, sky={gaussians_sky.shape[0]}, "
            #           f"voxel={gaussians_source2.shape[0]}, total={gaussians_all.shape[0]}")
            #     print(f"[GS]   Point opacity: mean={pt_opacity.mean():.4f}, max={pt_opacity.max():.4f}, "
            #           f"min={pt_opacity.min():.4f}, >0.5={(pt_opacity>0.5).sum()}/{pt_opacity.shape[0]}")
            #     print(f"[GS]   Voxel opacity: mean={vx_opacity.mean():.4f}, max={vx_opacity.max():.4f}, "
            #           f"min={vx_opacity.min():.4f}, >0.5={(vx_opacity>0.5).sum()}/{vx_opacity.shape[0]}")
            #     print(f"[GS]   Point scales: mean={pt_scales.mean():.5f}, max={pt_scales.max():.5f}")
            #     print(f"[GS]   Voxel scales: mean={vx_scales.mean():.5f}, max={vx_scales.max():.5f}")
                
                # # Viser 可视化：point 分支和 voxel 分支的高斯点云
                # try:
                #     import viser
                #     import os
                #     if os.environ.get("VISER_DEBUG", "0") == "1":
                #         server = viser.ViserServer(host="0.0.0.0", port=8890)
                        
                #         # Point 分支（蓝色）- 含 sky
                #         point_positions = gaussians_source1[:, :3].detach().cpu().numpy()
                #         point_colors_raw = gaussians_source1[:, 3:6].detach().cpu().numpy()
                #         point_colors = (point_colors_raw * 255).clip(0, 255).astype("uint8")
                #         server.scene.add_point_cloud(
                #             "/point_branch",
                #             points=point_positions,
                #             colors=point_colors,
                #             point_size=0.02,
                #         )
                #         print(f"[VISER] Point branch: {point_positions.shape[0]} points, "
                #               f"range=[{point_positions.min(0)}, {point_positions.max(0)}]")
                        
                #         # Voxel 分支（红色）
                #         voxel_positions = gaussians_source2[:, :3].detach().cpu().numpy()
                #         voxel_colors_raw = gaussians_source2[:, 3:6].detach().cpu().numpy()
                #         voxel_colors = (voxel_colors_raw * 255).clip(0, 255).astype("uint8")
                #         server.scene.add_point_cloud(
                #             "/voxel_branch",
                #             points=voxel_positions,
                #             colors=voxel_colors,
                #             point_size=0.04,
                #         )
                #         print(f"[VISER] Voxel branch: {voxel_positions.shape[0]} points, "
                #               f"range=[{voxel_positions.min(0)}, {voxel_positions.max(0)}]")
                        
                #         print(f"[VISER] Server running at http://0.0.0.0:8890")
                #         print(f"[VISER] Press Enter to continue training...")
                #         input()  # 暂停等待查看
                #         server.stop()
                # except Exception as e:
                #     print(f"[VISER] Visualization error: {e}")

            # Per-Gaussian LiDAR intensity logit, regressed by `to_lidar` from:
            #   [sample_features(optional), LiDAR ray direction, reflectance
            #    embedding, LiDAR-frame normal, incidence cosine, (optional image
            #    features)].
            # The LiDAR ray direction is the (per-Gaussian, view-independent) unit
            # vector from the LiDAR origin to the Gaussian in the LiDAR frame.
            # The scalar logit is folded into the main camera render as an extra
            # channel (no second rasterization pass).
            lidar_features = None
            lidar_multiscale_logits_i = None
            if self.use_lidar_supervision and (
                isinstance(lidar_supervision_batch, dict)
                or isinstance(lidar_image_supervision_batch, dict)
            ):
                if isinstance(pt_sky_mask, torch.Tensor):
                    inputs_ok = True
                    sample_feat_all = None
                    lidar_feat_dtype = gaussians_all.dtype
                    lidar_supervision_i_for_feat = self._select_lidar_supervision_for_batch(
                        lidar_supervision_batch,
                        batch_idx=i,
                        batch_size=B,
                    )

                    if self.lidar_use_sample_features:
                        if (
                            isinstance(sample_features, torch.Tensor)
                            and sample_features.shape[0] == pt_sky_mask.shape[0]
                        ):
                            # Full per-Gaussian sample_features in gaussians_all order:
                            # [non-sky points, sky points] then voxel (zeros).
                            sample_feat_source1 = torch.cat(
                                [sample_features[~pt_sky_mask], sample_features[pt_sky_mask]],
                                dim=0,
                            )
                            if sample_feat_source1.shape[0] == gaussians_source1.shape[0]:
                                if use_voxel_branch and gaussians_source2.shape[0] > 0:
                                    sample_feat_voxel = sample_features.new_zeros(
                                        gaussians_source2.shape[0],
                                        sample_feat_source1.shape[1],
                                    )
                                    sample_feat_all = torch.cat(
                                        [sample_feat_source1, sample_feat_voxel], dim=0
                                    )
                                else:
                                    sample_feat_all = sample_feat_source1
                                if sample_feat_all.shape[0] == gaussians_all.shape[0]:
                                    lidar_feat_dtype = sample_feat_all.dtype
                                else:
                                    inputs_ok = False
                            else:
                                inputs_ok = False
                        else:
                            inputs_ok = False

                    if inputs_ok:
                        # LiDAR ray direction (LiDAR frame) per Gaussian.
                        lidar2world_ref = current_lidar2world[i][0].to(
                            lidar_feat_dtype
                        )  # [4, 4]
                        world2lidar = torch.inverse(lidar2world_ref)
                        means_w = gaussians_all[:, :3].to(lidar_feat_dtype)
                        pts_lidar = (
                            means_w @ world2lidar[:3, :3].transpose(-1, -2)
                            + world2lidar[:3, 3]
                        )  # [N, 3]
                        lidar_range = pts_lidar.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                        lidar_ray_dir_pred = F.normalize(pts_lidar, dim=-1)  # [N, 3]
                        lidar_ray_dir = lidar_ray_dir_pred

                        def _current_lidar_image_map(map_b):
                            if not isinstance(map_b, torch.Tensor):
                                return None
                            map_i = map_b
                            if map_i.dim() == 4 and map_i.shape[0] == B:
                                map_i = map_i[i]
                            if map_i.dim() == 2:
                                map_i = map_i.unsqueeze(0)
                            if map_i.dim() != 3:
                                return None
                            if map_i.shape[0] != S_i:
                                if map_i.shape[0] > S_i:
                                    map_i = map_i[-S_i:]
                                else:
                                    return None
                            return map_i

                        def _source1_pixel_indices_for_lidar_input():
                            num_pixels_local = int(S_i) * int(H_i) * int(W_i)
                            base_idx_local = None
                            if pt_sky_mask.numel() == num_pixels_local:
                                base_idx_local = torch.arange(
                                    num_pixels_local,
                                    device=means_w.device,
                                    dtype=torch.long,
                                )
                            elif (
                                valid_mask_i is not None
                                and valid_mask_i.numel() == num_pixels_local
                                and int(valid_mask_i.reshape(-1).sum().item())
                                == pt_sky_mask.numel()
                            ):
                                base_idx_local = torch.nonzero(
                                    valid_mask_i.reshape(-1).bool(),
                                    as_tuple=False,
                                ).reshape(-1).to(
                                    device=means_w.device,
                                    dtype=torch.long,
                                )
                            if not isinstance(base_idx_local, torch.Tensor):
                                return None
                            source1_pix_local = torch.cat(
                                [base_idx_local[~pt_sky_mask], base_idx_local[pt_sky_mask]],
                                dim=0,
                            )
                            if (
                                source1_pix_local.shape[0] != gaussians_source1.shape[0]
                                or gaussians_source1.shape[0] > lidar_range.shape[0]
                            ):
                                return None
                            return source1_pix_local

                        if (
                            self.lidar_use_gt_range
                            and isinstance(lidar_image_supervision_batch, dict)
                        ):
                            range_gt_i = _current_lidar_image_map(
                                lidar_image_supervision_batch.get("range", None)
                            )
                            range_valid_i = _current_lidar_image_map(
                                lidar_image_supervision_batch.get("valid_mask", None)
                            )
                            if (
                                isinstance(range_gt_i, torch.Tensor)
                                and isinstance(range_valid_i, torch.Tensor)
                            ):
                                range_gt_i = range_gt_i.to(
                                    device=means_w.device,
                                    dtype=lidar_feat_dtype,
                                )
                                range_valid_i = range_valid_i.to(
                                    device=means_w.device,
                                    dtype=torch.bool,
                                )
                                if range_gt_i.shape[-2:] != (H_i, W_i):
                                    range_gt_i = F.interpolate(
                                        range_gt_i.unsqueeze(1),
                                        size=(H_i, W_i),
                                        mode="nearest",
                                    ).squeeze(1)
                                if range_valid_i.shape[-2:] != (H_i, W_i):
                                    range_valid_i = (
                                        F.interpolate(
                                            range_valid_i.float().unsqueeze(1),
                                            size=(H_i, W_i),
                                            mode="nearest",
                                        )
                                        .squeeze(1)
                                        > 0.5
                                    )
                                source1_pix_range = _source1_pixel_indices_for_lidar_input()
                                if isinstance(source1_pix_range, torch.Tensor):
                                    range_gt_flat = range_gt_i.reshape(-1)
                                    range_valid_flat = (
                                        range_valid_i.reshape(-1)
                                        & torch.isfinite(range_gt_flat)
                                        & (range_gt_flat > 0)
                                    )
                                    range_gt_s1 = range_gt_flat[source1_pix_range]
                                    range_valid_s1 = range_valid_flat[source1_pix_range]
                                    valid_idx_s1 = torch.nonzero(
                                        range_valid_s1, as_tuple=False
                                    ).reshape(-1)
                                    if valid_idx_s1.numel() > 0:
                                        lidar_range = lidar_range.clone()
                                        lidar_range[valid_idx_s1, 0] = range_gt_s1[
                                            valid_idx_s1
                                        ]
                        # Optional ablation: use GT LiDAR ray direction for source1
                        # Gaussians where LiDAR-image depth is valid.
                        if (
                            self.lidar_use_gt_ray_dir
                            and isinstance(lidar_image_supervision_batch, dict)
                        ):
                            depth_gt_b = lidar_image_supervision_batch.get("depth", None)
                            valid_gt_b = lidar_image_supervision_batch.get(
                                "valid_mask", None
                            )
                            if (
                                isinstance(depth_gt_b, torch.Tensor)
                                and isinstance(valid_gt_b, torch.Tensor)
                            ):
                                depth_gt_i = depth_gt_b
                                valid_gt_i = valid_gt_b
                                if depth_gt_i.dim() == 4 and depth_gt_i.shape[0] == B:
                                    depth_gt_i = depth_gt_i[i]
                                if valid_gt_i.dim() == 4 and valid_gt_i.shape[0] == B:
                                    valid_gt_i = valid_gt_i[i]
                                if depth_gt_i.dim() == 2:
                                    depth_gt_i = depth_gt_i.unsqueeze(0)
                                if valid_gt_i.dim() == 2:
                                    valid_gt_i = valid_gt_i.unsqueeze(0)
                                if depth_gt_i.dim() == 3 and valid_gt_i.dim() == 3:
                                    if depth_gt_i.shape[0] != S_i:
                                        if depth_gt_i.shape[0] > S_i:
                                            depth_gt_i = depth_gt_i[-S_i:]
                                        else:
                                            depth_gt_i = None
                                    if (
                                        isinstance(depth_gt_i, torch.Tensor)
                                        and isinstance(valid_gt_i, torch.Tensor)
                                        and valid_gt_i.shape[0] != S_i
                                    ):
                                        if valid_gt_i.shape[0] > S_i:
                                            valid_gt_i = valid_gt_i[-S_i:]
                                        else:
                                            valid_gt_i = None
                                    if (
                                        isinstance(depth_gt_i, torch.Tensor)
                                        and isinstance(valid_gt_i, torch.Tensor)
                                    ):
                                        depth_gt_i = depth_gt_i.to(
                                            device=means_w.device,
                                            dtype=lidar_feat_dtype,
                                        )
                                        valid_gt_i = valid_gt_i.to(
                                            device=means_w.device,
                                            dtype=torch.bool,
                                        )
                                        if depth_gt_i.shape[-2:] != (H_i, W_i):
                                            depth_gt_i = F.interpolate(
                                                depth_gt_i.unsqueeze(1),
                                                size=(H_i, W_i),
                                                mode="nearest",
                                            ).squeeze(1)
                                        if valid_gt_i.shape[-2:] != (H_i, W_i):
                                            valid_gt_i = (
                                                F.interpolate(
                                                    valid_gt_i.float().unsqueeze(1),
                                                    size=(H_i, W_i),
                                                    mode="nearest",
                                                )
                                                .squeeze(1)
                                                > 0.5
                                            )
                                        rays_o_i = rays_o[i].to(
                                            device=means_w.device,
                                            dtype=lidar_feat_dtype,
                                        )
                                        rays_d_i = rays_d[i].to(
                                            device=means_w.device,
                                            dtype=lidar_feat_dtype,
                                        )
                                        if (
                                            rays_o_i.shape[:3] == depth_gt_i.shape
                                            and rays_d_i.shape[:3] == depth_gt_i.shape
                                        ):
                                            pts_lidar_gt = (
                                                rays_o_i
                                                + rays_d_i * depth_gt_i[..., None]
                                            ).reshape(-1, 3)
                                            gt_ray_dir_full = F.normalize(
                                                pts_lidar_gt, dim=-1
                                            )
                                            gt_valid_full = (
                                                valid_gt_i.reshape(-1)
                                                & torch.isfinite(depth_gt_i.reshape(-1))
                                                & (depth_gt_i.reshape(-1) > 0)
                                            )
                                            num_pixels_i = int(S_i) * int(H_i) * int(W_i)
                                            base_idx = None
                                            if pt_sky_mask.numel() == num_pixels_i:
                                                base_idx = torch.arange(
                                                    num_pixels_i,
                                                    device=means_w.device,
                                                    dtype=torch.long,
                                                )
                                            elif (
                                                valid_mask_i is not None
                                                and valid_mask_i.numel() == num_pixels_i
                                                and int(valid_mask_i.reshape(-1).sum().item())
                                                == pt_sky_mask.numel()
                                            ):
                                                base_idx = torch.nonzero(
                                                    valid_mask_i.reshape(-1).bool(),
                                                    as_tuple=False,
                                                ).reshape(-1).to(
                                                    device=means_w.device,
                                                    dtype=torch.long,
                                                )
                                            if isinstance(base_idx, torch.Tensor):
                                                source1_pix = torch.cat(
                                                    [
                                                        base_idx[~pt_sky_mask],
                                                        base_idx[pt_sky_mask],
                                                    ],
                                                    dim=0,
                                                )
                                                if (
                                                    source1_pix.shape[0]
                                                    == gaussians_source1.shape[0]
                                                    and gaussians_source1.shape[0]
                                                    <= lidar_ray_dir.shape[0]
                                                ):
                                                    gt_ray_dir_s1 = gt_ray_dir_full[
                                                        source1_pix
                                                    ]
                                                    gt_valid_s1 = gt_valid_full[
                                                        source1_pix
                                                    ]
                                                    valid_idx_s1 = torch.nonzero(
                                                        gt_valid_s1, as_tuple=False
                                                    ).reshape(-1)
                                                    if valid_idx_s1.numel() > 0:
                                                        lidar_ray_dir = (
                                                            lidar_ray_dir_pred.clone()
                                                        )
                                                        lidar_ray_dir[
                                                            valid_idx_s1
                                                        ] = gt_ray_dir_s1[
                                                            valid_idx_s1
                                                        ]
                        # Learnable reflectance embedding from per-Gaussian RGB
                        # (appearance/material proxy).
                        rgb_all = gaussians_all[:, 3:6].to(lidar_feat_dtype)
                        refl_embed = self.lidar_reflectance_encoder(
                            rgb_all
                        )  # [N, refl_dim]
                        # Surface normal (smallest-scale axis) in the LiDAR
                        # frame + incidence cosine |d.n| (cos(incidence angle)).
                        gs_normal_w = gaussian_orientation_from_scales(
                            gaussians_all[:, 7:11], gaussians_all[:, 11:14]
                        ).to(lidar_feat_dtype)  # [N, 3] world frame
                        normal_lidar = F.normalize(
                            gs_normal_w
                            @ world2lidar[:3, :3].transpose(-1, -2),
                            dim=-1,
                        )  # [N, 3] LiDAR frame
                        incidence_cos = (
                            (lidar_ray_dir * normal_lidar)
                            .sum(-1, keepdim=True)
                            .abs()
                        )  # [N, 1]
                        lidar_ray_embed = self._get_lidar_ray_embedding(
                            lidar_ray_dir=lidar_ray_dir,
                            lidar_supervision_i=lidar_supervision_i_for_feat,
                        )
                        lidar_range_features = None
                        if self.lidar_use_range_features:
                            log_range = torch.log1p(lidar_range)
                            inv_range = 1.0 / lidar_range
                            inv_range2 = 1.0 / (lidar_range * lidar_range)
                            lidar_range_features = torch.cat(
                                [log_range, inv_range, inv_range2], dim=-1
                            )
                        lidar_range_embed = self._get_lidar_range_embedding(
                            lidar_range
                        )
                        to_lidar_inputs = []
                        if self.lidar_use_sample_features and isinstance(
                            sample_feat_all, torch.Tensor
                        ):
                            to_lidar_inputs.append(sample_feat_all.to(lidar_feat_dtype))
                        to_lidar_inputs.extend(
                            [lidar_ray_dir, refl_embed, normal_lidar, incidence_cos]
                        )
                        if isinstance(lidar_ray_embed, torch.Tensor):
                            to_lidar_inputs.append(lidar_ray_embed.to(lidar_feat_dtype))
                        if isinstance(lidar_range_features, torch.Tensor):
                            to_lidar_inputs.append(
                                lidar_range_features.to(lidar_feat_dtype)
                            )
                        if isinstance(lidar_range_embed, torch.Tensor):
                            to_lidar_inputs.append(lidar_range_embed.to(lidar_feat_dtype))
                        img_feat_all = None
                        if self.lidar_use_image_encoder or self.lidar_use_rgb_input:
                            # Reorder the per-point image-derived features
                            # ([non-sky, sky] then voxel zeros) to match gaussians_all.
                            if isinstance(lidar_img_feat_per_point, torch.Tensor):
                                img_feat_src1 = torch.cat(
                                    [
                                        lidar_img_feat_per_point[~pt_sky_mask],
                                        lidar_img_feat_per_point[pt_sky_mask],
                                    ],
                                    dim=0,
                                )
                                if (
                                    use_voxel_branch
                                    and gaussians_source2.shape[0] > 0
                                ):
                                    img_feat_voxel = (
                                        lidar_img_feat_per_point.new_zeros(
                                            gaussians_source2.shape[0],
                                            img_feat_src1.shape[1],
                                        )
                                    )
                                    img_feat_all = torch.cat(
                                        [img_feat_src1, img_feat_voxel], dim=0
                                    )
                                else:
                                    img_feat_all = img_feat_src1
                                if img_feat_all.shape[0] == gaussians_all.shape[0]:
                                    to_lidar_inputs.append(
                                        img_feat_all.to(lidar_feat_dtype)
                                    )
                                else:
                                    inputs_ok = False
                                    img_feat_all = None
                            else:
                                inputs_ok = False
                        if inputs_ok and len(to_lidar_inputs) > 0:
                            if (
                                self.lidar_head_type == "query"
                                and self.lidar_query_head is not None
                            ):
                                # Memory-light path: build only source1 (Q/KV) and
                                # voxel-branch features; avoid the full [N, D]
                                # to_lidar_in concat (built only on fallback).
                                n1_src = gaussians_source1.shape[0]
                                n_all = gaussians_all.shape[0]
                                source1_pix = self._lidar_source1_pixel_index(
                                    n1_src,
                                    pt_sky_mask,
                                    valid_mask_i,
                                    S_i,
                                    H_i,
                                    W_i,
                                    gaussians_all.device,
                                )
                                lidar_features = None
                                if isinstance(source1_pix, torch.Tensor):
                                    # Q = geometry tokens (source1 only).
                                    q_inputs = [
                                        lidar_ray_dir[:n1_src],
                                        normal_lidar[:n1_src],
                                        incidence_cos[:n1_src],
                                    ]
                                    if isinstance(lidar_range_features, torch.Tensor):
                                        q_inputs.append(
                                            lidar_range_features[:n1_src].to(
                                                lidar_feat_dtype
                                            )
                                        )
                                    if isinstance(lidar_range_embed, torch.Tensor):
                                        q_inputs.append(
                                            lidar_range_embed[:n1_src].to(
                                                lidar_feat_dtype
                                            )
                                        )
                                    if isinstance(lidar_ray_embed, torch.Tensor):
                                        q_inputs.append(
                                            lidar_ray_embed[:n1_src].to(
                                                lidar_feat_dtype
                                            )
                                        )
                                    # K/V = appearance/context (source1 only),
                                    # kept as a list so the head can project each
                                    # component additively (no big concat copy).
                                    # Order must match _lidar_query_kv_component_dims.
                                    kv_list = []
                                    if self.lidar_use_sample_features and isinstance(
                                        sample_feat_all, torch.Tensor
                                    ):
                                        kv_list.append(
                                            sample_feat_all[:n1_src].to(lidar_feat_dtype)
                                        )
                                    kv_list.append(refl_embed[:n1_src])
                                    if isinstance(img_feat_all, torch.Tensor):
                                        kv_list.append(
                                            img_feat_all[:n1_src].to(lidar_feat_dtype)
                                        )
                                    q_s1 = torch.cat(q_inputs, dim=-1)
                                    # Voxel-branch features (small): concat voxel rows.
                                    voxel_feat = None
                                    if n_all > n1_src:
                                        voxel_feat = torch.cat(
                                            [t[n1_src:] for t in to_lidar_inputs],
                                            dim=-1,
                                        )
                                    lidar_features = self._lidar_query_forward(
                                        q_s1,
                                        kv_list,
                                        voxel_feat,
                                        source1_pix,
                                        S_i,
                                        H_i,
                                        W_i,
                                    )
                                if lidar_features is None:
                                    to_lidar_in = torch.cat(to_lidar_inputs, dim=-1)
                                    lidar_features = self.to_lidar(to_lidar_in)
                            elif (
                                self.lidar_head_type in {"conv2d", "unet"}
                                and self.lidar_conv2d_head is not None
                            ):
                                to_lidar_in = torch.cat(to_lidar_inputs, dim=-1)
                                lidar_head_out = self._lidar_conv2d_forward(
                                    to_lidar_in,
                                    gaussians_source1.shape[0],
                                    pt_sky_mask,
                                    valid_mask_i,
                                    S_i,
                                    H_i,
                                    W_i,
                                    collect_multiscale=self.lidar_unet_multiscale_supervision,
                                )
                                if isinstance(lidar_head_out, dict):
                                    lidar_features = lidar_head_out.get(
                                        "gaussian_logits", None
                                    )
                                    lidar_multiscale_logits_i = lidar_head_out.get(
                                        "image_logits_multiscale", None
                                    )
                                else:
                                    lidar_features = lidar_head_out
                                if lidar_features is None:
                                    lidar_features = self.to_lidar(to_lidar_in)
                            else:
                                to_lidar_in = torch.cat(to_lidar_inputs, dim=-1)
                                lidar_features = self.to_lidar(to_lidar_in)

            # Fold the per-Gaussian intensity logit into the main render as an
            # extra blended channel only when image-plane intensity is supervised.
            render_extra_feats = None
            if (
                self.use_lidar_supervision
                and isinstance(lidar_image_supervision_batch, dict)
                and isinstance(lidar_features, torch.Tensor)
                and lidar_features.shape[0] == gaussians_all.shape[0]
            ):
                render_extra_feats = lidar_features  # [N, 1]

            render_kwargs = dict(
                gaussians=gaussians_all[:, :],
                c2w=render_c2w[i, select_index],
                fovx=render_fovxs[i, select_index],
                fovy=render_fovys[i, select_index],
                rays_o=None,
                rays_d=None,
                K=out_intrinsics[i, select_index],
                H=H_stack[select_index],
                W=W_stack[select_index],
                semantics=gaussians_all_dynscore,
                enable_cuda_fisher=False,
                extra_feats_per_view=render_extra_feats,
            )
            tmp = self.renderer.render(**render_kwargs)
            # Pull the blended intensity logit out of the render dict so it does
            # not get merged with the regular render outputs downstream.
            lidar_intensity_map = tmp.pop("lidar_intensity", None)
            batch_render_outputs.append(tmp)

            if (
                self.use_lidar_supervision
                and isinstance(lidar_supervision_batch, dict)
                and isinstance(lidar_features, torch.Tensor)
                and (need_lidar_vis or train_lidar_plane_loss)
            ):
                lidar_supervision_i = {}
                for key, value in lidar_supervision_batch.items():
                    if isinstance(value, torch.Tensor):
                        if value.dim() > 0 and value.shape[0] == B:
                            lidar_supervision_i[key] = value[i]
                        else:
                            lidar_supervision_i[key] = value
                    else:
                        lidar_supervision_i[key] = value
                if isinstance(lidar_supervision_i.get("viewmats", None), torch.Tensor):
                    num_lidars_i = int(lidar_supervision_i["viewmats"].shape[0])
                    if train_lidar_plane_loss:
                        # LiDAR-plane intensity supervision: keep gradients.
                        lidar_features_spherical = lidar_features.unsqueeze(0).expand(
                            num_lidars_i, -1, -1
                        )  # [L, N, 1]
                        lidar_render_out = self.renderer.render_lidar(
                            gaussians=gaussians_all[:, :],
                            lidar_supervision=lidar_supervision_i,
                            lidar_features=lidar_features_spherical,
                            velocity=None,
                        )
                    else:
                        # Visualization-only spherical render (legacy path).
                        with torch.no_grad():
                            lidar_features_spherical = lidar_features.detach().unsqueeze(
                                0
                            ).expand(num_lidars_i, -1, -1)  # [L, N, 1]
                            lidar_render_out = self.renderer.render_lidar(
                                gaussians=gaussians_all[:, :],
                                lidar_supervision=lidar_supervision_i,
                                lidar_features=lidar_features_spherical,
                                velocity=None,
                            )
                    if len(lidar_render_out) > 0:
                        batch_lidar_render.append(lidar_render_out)

            if (
                self.use_lidar_supervision
                and isinstance(lidar_image_supervision_batch, dict)
                and isinstance(lidar_features, torch.Tensor)
            ):
                lidar_image_supervision_i = {}
                for key, value in lidar_image_supervision_batch.items():
                    if isinstance(value, torch.Tensor):
                        if value.dim() > 0 and value.shape[0] == B:
                            lidar_image_supervision_i[key] = value[i]
                        else:
                            lidar_image_supervision_i[key] = value
                    else:
                        lidar_image_supervision_i[key] = value

                if isinstance(lidar_image_supervision_i.get("range", None), torch.Tensor):
                    lidar_pred_i = {
                        "range": tmp["depth"],
                    }
                    # Intensity logit was blended together with RGB in the main
                    # render (extra channel) -> read the popped map directly.
                    if isinstance(lidar_intensity_map, torch.Tensor):
                        lidar_pred_i["intensity_logits"] = lidar_intensity_map
                    if isinstance(lidar_multiscale_logits_i, dict):
                        for key, val in lidar_multiscale_logits_i.items():
                            if isinstance(val, torch.Tensor):
                                lidar_pred_i[key] = val
                    if len(lidar_pred_i) > 0:
                        batch_lidar_image_pred.append(lidar_pred_i)

            if need_fisher_data:
                fisher_gaussians.append(gaussians_all)
                fisher_semantics.append(gaussians_all_dynscore)
                fisher_render_inputs.append(
                    {
                        "c2w": render_c2w[i, select_index],
                        "fovx": render_fovxs[i, select_index],
                        "fovy": render_fovys[i, select_index],
                        "K": out_intrinsics[i, select_index],
                        "H": H_stack[select_index],
                        "W": W_stack[select_index],
                    }
                )
            pred_dynamics_region = (tmp["dyn"][-S:] > 0.5)[:, 0, :, :]
            dyn_gs_mask = self.get_dynamic_gs_mask(
                pred_dynamics_region,
                gaussians_all[:, :3].detach(),
                intrinsics[i],
                camera2lidar[i],
            )
            static_mask = ~dyn_gs_mask
            write_mask = static_mask
            save_gaussians.append(gaussians_all.detach()[write_mask])
        cache_dict = {
            "images": images,
            "lidars": [None] * B,
            "scenes": aggregated_tokens_list["scene"],
            "frames": aggregated_tokens_list["frame"],
            "gt_dynamics_mask": [None] * B,
            "pred_dynamics_mask": [None] * B,
            "cam2lidars": [None] * B,
            "lidar2worlds": aggregated_tokens_list["lidar2world"],
            "voxel_features": save_features.detach(),
            "voxel_poses": save_coords.detach(),
            "voxel_opacities": voxel_opacities.detach(),
            "gaussians": save_gaussians,
            "gaussians_outview": save_gaussians_outview,
        }
        self.history_queue.cache(B, **cache_dict)

        # Merge batch render outputs: stack tensors along batch dimension
        def _merge_render_outputs(outputs):
            if len(outputs) == 0:
                return None
            if len(outputs) == 1:
                return outputs[0]
            merged = {}
            for key in outputs[0].keys():
                values = [out[key] for out in outputs]
                # radii: [V, N_i] per batch, N_i can differ; keep as list
                if key in ("radii", "cuda_fishers", "fisher_render_inputs"):
                    merged[key] = values
                    continue
                if isinstance(values[0], torch.Tensor):
                    if values[0].dim() == 0 or values[0].numel() == 1:
                        merged[key] = torch.stack(values).mean()
                    else:
                        merged[key] = torch.cat(values, dim=0)
                else:
                    merged[key] = values
            return merged

        def _merge_lidar_outputs(outputs):
            if len(outputs) == 0:
                return None
            merged = {}
            keys = outputs[0].keys()
            for key in keys:
                values = [out[key] for out in outputs if key in out]
                if len(values) == 0:
                    continue
                if isinstance(values[0], torch.Tensor):
                    merged[key] = torch.stack(values, dim=0)
                else:
                    merged[key] = values
            return merged
        
        test_out = _merge_render_outputs(batch_render_outputs)
        lidar_render_out = _merge_lidar_outputs(batch_lidar_render)
        lidar_image_pred_out = _merge_lidar_outputs(batch_lidar_image_pred)
        
        if "losses" not in test_out:
            test_out["losses"] = {}
        test_out["losses"]["loss_orient_raw"] = batch_loss_orient_raw
        test_out["losses"]["loss_flat_raw"] = batch_loss_flat_raw
        test_out["losses"]["loss_orient_means"] = batch_loss_orient_means
        test_out["losses"]["loss_orient_is_voxel"] = batch_loss_orient_is_voxel
        
        # 分支独立渲染结果
        test_out_pixel = _merge_render_outputs(batch_render_pixel)
        test_out_volume = _merge_render_outputs(batch_render_volume)
        if test_out_pixel is not None:
            test_out["render_pixel"] = test_out_pixel
        if test_out_volume is not None:
            test_out["render_volume"] = test_out_volume
        if lidar_render_out is not None:
            test_out["lidar_render"] = lidar_render_out
        if lidar_image_pred_out is not None:
            test_out["lidar_image_pred"] = lidar_image_pred_out
        
        # 返回 depth_map_unscale（scale/shift 调整后、sky fill/clamp 之前）用于深度监督
        test_out["depth_map_unscale"] = depth_map_unscale_pred  # [B, S, H, W]
        if need_overlap_geo_attr:
            if batch_attr_xyz_maps is not None and len(batch_attr_xyz_maps) == B:
                test_out["gaussian_attr_xyz_map"] = torch.stack(batch_attr_xyz_maps, dim=0)
            else:
                test_out["gaussian_attr_xyz_map"] = attr_xyz_map
            test_out["gaussian_attr_normal_map"] = attr_normal_map if attr_normal_map is not None else torch.zeros_like(attr_xyz_map)
            if batch_attr_scale_maps is not None and len(batch_attr_scale_maps) == B:
                test_out["gaussian_attr_scale_map"] = torch.stack(batch_attr_scale_maps, dim=0)
            else:
                test_out["gaussian_attr_scale_map"] = attr_xyz_map.new_zeros(
                    attr_xyz_map.shape[0], attr_xyz_map.shape[1], attr_xyz_map.shape[2], attr_xyz_map.shape[3], 3
                )
            if batch_attr_rgb_maps is not None and len(batch_attr_rgb_maps) == B:
                test_out["gaussian_attr_rgb_map"] = torch.stack(batch_attr_rgb_maps, dim=0)
            else:
                test_out["gaussian_attr_rgb_map"] = attr_xyz_map.new_zeros(
                    attr_xyz_map.shape[0], attr_xyz_map.shape[1], attr_xyz_map.shape[2], attr_xyz_map.shape[3], 3
                )
        # 返回 local_depth / pred_scale / pred_shift 用于 MoGe 式 scale-shift 对齐 loss（公式3）
        test_out["local_depth"] = local_depth  # [B, S, H, W] 未乘 scale/shift 的局部深度
        test_out["pred_scale"] = pred_scale    # [B, S]
        test_out["pred_shift"] = pred_shift    # [B, S]
        if isinstance(predicted_bilateral_grids, dict):
            test_out["predicted_bilateral_grids"] = predicted_bilateral_grids
        if need_fisher_data:
            test_out["fisher_gaussians"] = fisher_gaussians
            test_out["fisher_semantics"] = fisher_semantics
            test_out["fisher_render_inputs"] = fisher_render_inputs
            test_out["means3D_for_fisher"] = [g[:, :3] for g in fisher_gaussians]

        # Export detached DINO patch tokens for LiDAR-affine semantic gating in train loop.
        if (
            isinstance(dinov2_patchtokens_raw, torch.Tensor)
            and dinov2_patchtokens_raw.dim() == 3
            and dinov2_patchtokens_raw.shape[1] == patch_h * patch_w
        ):
            dino_tokens_5d = dinov2_patchtokens_raw.reshape(
                B, S, patch_h, patch_w, -1
            ).permute(0, 1, 4, 2, 3).contiguous()
            test_out["lidar_affine_dino_tokens"] = dino_tokens_5d.detach()
            test_out["lidar_affine_dino_patch_size"] = int(self.patch_size)

        return test_out, save_gaussians


    def process_guassian_voxel(self, gaussians, means, voxel_batch_idx):
        gaussians = gaussians.reshape(-1, self.cfg.voxel_gs_num, 15)
        means = means[:, None, :].repeat(1, self.cfg.voxel_gs_num, 1)
        gaussians = gaussians.reshape(-1, 15)
        means = means.reshape(-1, 3)
        voxel_batch_idx = voxel_batch_idx[:, None].repeat(1, self.cfg.voxel_gs_num)
        voxel_batch_idx = voxel_batch_idx.reshape(-1)

        offsets = gaussians[..., :3]
        opacities = self.opt_act(gaussians[..., 3:4])
        scales = self.scale_act(gaussians[..., 4:7])
        scales = torch.clamp_max(scales, self.cfg["max_scale_voxel"])
        rotations = self.rot_act(gaussians[..., 7:11])
        rgbs = self.rgb_act(gaussians[..., 11:14])
        if self._freeze_gaussian_rgb:
            rgbs = rgbs.detach()
        dynamic = nn.functional.sigmoid(gaussians[..., 14:15])
        means = means + offsets
        gaussians = torch.cat(
            [means, rgbs, opacities, rotations, scales, dynamic], dim=-1
        )
        voxel_opacities = opacities[:, 0].reshape(-1, self.cfg.voxel_gs_num)
        voxel_opacities = voxel_opacities.mean(dim=-1)
        return gaussians, voxel_batch_idx, voxel_opacities

    def plucker_embedder(self, rays_o, rays_d):
        rays_o = rays_o.permute(0, 1, 4, 2, 3)
        rays_d = rays_d.permute(0, 1, 4, 2, 3)
        plucker = torch.cat([torch.cross(rays_o, rays_d, dim=2), rays_d], dim=2)
        return plucker

    def _apply_pos_embed(
        self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1
    ) -> torch.Tensor:
        """
        Apply positional embedding to tensor x.
        """
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(
            patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device
        )
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed

    def scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the fusion blocks.

        Args:
            features (List[Tensor]): List of feature maps from different layers.

        Returns:
            Tensor: Fused feature map.
        """
        layer_1, layer_2, layer_3, layer_4 = features

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4

        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3

        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2

        out = self.scratch.refinenet1(out, layer_1_rn)
        del layer_1_rn, layer_1

        out = self.scratch.output_conv1(out)
        return out

    def _log_sky_geometry_once(self, key: str, message: str) -> None:
        if key in self._sky_geometry_log_once:
            return
        self._sky_geometry_log_once.add(key)
        print(message)

    def _build_sky_means_base(
        self,
        means: torch.Tensor,
        sky_mask: torch.Tensor,
        rays_d: torch.Tensor = None,
    ) -> torch.Tensor:
        sky_default = means[sky_mask]
        if sky_default.numel() == 0:
            return sky_default

        mode = str(self.cfg.get("sky_geometry_mode", "depth_fill")).lower()
        self._log_sky_geometry_once(
            f"mode_selected_{mode}",
            f"[SkyGeometry] mode={mode}",
        )
        if mode == "depth_fill":
            return sky_default

        if mode == "hemispherical":
            self._log_sky_geometry_once(
                "hemisphere_not_implemented",
                "[SkyGeometry] sky_geometry_mode='hemispherical' is not implemented yet; fallback to depth_fill.",
            )
            return sky_default

        if mode != "cylindrical":
            self._log_sky_geometry_once(
                f"unsupported_mode_{mode}",
                f"[SkyGeometry] Unsupported sky_geometry_mode='{mode}', fallback to depth_fill.",
            )
            return sky_default

        if (
            rays_d is None
            or not isinstance(rays_d, torch.Tensor)
            or rays_d.shape[0] != means.shape[0]
            or rays_d.shape[-1] != 3
        ):
            self._log_sky_geometry_once(
                "cyl_missing_rays",
                "[SkyGeometry] Cylindrical mode requires per-point rays_d with shape [N,3]; fallback to depth_fill.",
            )
            return sky_default

        center_mode = str(self.cfg.get("sky_cylinder_center_mode", "ego_origin")).lower()
        if center_mode != "ego_origin":
            self._log_sky_geometry_once(
                f"cyl_center_mode_{center_mode}",
                f"[SkyGeometry] sky_cylinder_center_mode='{center_mode}' is not supported in v1; use ego_origin fallback.",
            )

        radius = float(self.cfg.get("sky_cylinder_radius", 120.0))
        max_abs_z = float(self.cfg.get("sky_cylinder_max_abs_z", 200.0))
        eps = 1e-6

        sky_rays_d = rays_d[sky_mask]
        dir_xy = sky_rays_d[:, :2]
        norm_xy = torch.linalg.norm(dir_xy, dim=-1, keepdim=True)
        dir_xy_unit = dir_xy / norm_xy.clamp(min=eps)

        x = dir_xy_unit[:, 0:1] * radius
        y = dir_xy_unit[:, 1:2] * radius
        z = (radius * sky_rays_d[:, 2:3] / norm_xy.clamp(min=eps)).clamp(
            min=-max_abs_z, max=max_abs_z
        )
        sky_cyl = torch.cat([x, y, z], dim=-1)

        near_vertical = norm_xy.squeeze(-1) < (eps * 10.0)
        if near_vertical.any():
            sky_cyl = sky_cyl.clone()
            sky_cyl[near_vertical] = sky_default[near_vertical]
        return sky_cyl

    def refine_guassians(
        self,
        voxel_features,
        voxel_coords,
        means,
        sky_mask,
        img_features,
        valid_mask=None,
        cam_indices=None,
        rays_d=None,
        extra_pixel_feats=None,
    ):
        """Exclude padding from voxels; when sky_mask_side_padding_opacity_zero, include padding for opacity=0."""
        if rays_d is not None:
            rays_d = rays_d.to(device=means.device, dtype=means.dtype)
            if rays_d.dim() > 2:
                rays_d = rays_d.reshape(-1, rays_d.shape[-1])
            if rays_d.dim() != 2 or rays_d.shape[-1] != 3:
                rays_d = None

        # Per-pixel extra appearance feature map [S, C, H, W] (e.g. EfficientNetV2),
        # filtered identically to img_features so it stays per-point aligned.
        extra_ch = (
            int(extra_pixel_feats.shape[1]) if extra_pixel_feats is not None else 0
        )

        include_padding = self.cfg.get("sky_mask_side_padding_opacity_zero", False)
        if valid_mask is not None and valid_mask.numel() > 0:
            valid_mask = valid_mask.to(device=means.device, dtype=torch.bool)
            if valid_mask.dim() > 1:
                valid_mask = valid_mask.flatten()
            if valid_mask.shape[0] == means.shape[0]:
                if not include_padding:
                    means = means[valid_mask]
                    sky_mask = sky_mask.flatten()[valid_mask]
                    img_features = img_features.permute(0, 2, 3, 1).reshape(-1, 256)[valid_mask]
                    if extra_pixel_feats is not None:
                        extra_pixel_feats = extra_pixel_feats.permute(
                            0, 2, 3, 1
                        ).reshape(-1, extra_ch)[valid_mask]
                    if rays_d is not None:
                        if rays_d.shape[0] == valid_mask.shape[0]:
                            rays_d = rays_d[valid_mask]
                        else:
                            rays_d = None
                    if cam_indices is not None:
                        cam_indices = cam_indices.to(device=means.device)[valid_mask]
                    valid_mask = None  # no padding in filtered set
                else:
                    sky_mask = sky_mask.flatten()
                    img_features = img_features.permute(0, 2, 3, 1).reshape(-1, 256)
                    if extra_pixel_feats is not None:
                        extra_pixel_feats = extra_pixel_feats.permute(
                            0, 2, 3, 1
                        ).reshape(-1, extra_ch)
                    if rays_d is not None and rays_d.shape[0] != means.shape[0]:
                        rays_d = None
                    cam_indices = cam_indices.to(device=means.device) if cam_indices is not None else None
            else:
                img_features = img_features.permute(0, 2, 3, 1).reshape(-1, 256)
                if extra_pixel_feats is not None:
                    extra_pixel_feats = extra_pixel_feats.permute(
                        0, 2, 3, 1
                    ).reshape(-1, extra_ch)
                cam_indices = None
                valid_mask = None
                rays_d = None
        else:
            img_features = img_features.permute(0, 2, 3, 1).reshape(-1, 256)
            if extra_pixel_feats is not None:
                extra_pixel_feats = extra_pixel_feats.permute(0, 2, 3, 1).reshape(
                    -1, extra_ch
                )
            if rays_d is not None and rays_d.shape[0] != means.shape[0]:
                rays_d = None
            valid_mask = None

        if means.shape[0] == 0:
            dev, dt = means.device, means.dtype
            empty_pt = means.new_zeros(0, 15)
            empty_sky = means.new_zeros(0, 15)
            empty_feat = voxel_features.new_zeros(0, voxel_features.shape[1] + 256)
            empty_sky_mask = means.new_zeros(0, dtype=torch.bool)
            empty_sky_means = means.new_zeros(0, 3)
            out = (
                empty_pt,
                empty_pt[:, 14:15],
                empty_sky,
                empty_sky[:, 14:15],
                empty_feat,
                empty_sky_mask,
                empty_sky_means,
                extra_pixel_feats,
            )
            return out

        means_in = means - self.pts_range[None, :3]
        means_in = torch.floor(means_in / (self.voxel_size[None, :3] * 2)).int()
        means_in = (means_in + 0.5) * self.voxel_size[None, :3] * 2 + self.pts_range[
            None, :3
        ]
        points = torch.cat([means_in, voxel_coords[:, 1:]], dim=0)
        labels = torch.zeros(
            (points.shape[0],), dtype=torch.int, device=means_in.device
        )
        labels[means_in.shape[0] :] = 1
        torch.cuda.empty_cache()
        dist, indices = distCUDACross(points, labels)
        voxel_indices = indices[: means_in.shape[0]] - means_in.shape[0]
        has_voxel_mask = dist[: means_in.shape[0]] == 0
        sample_features = voxel_features.new_zeros(
            means.shape[0], voxel_features.shape[1]
        )
        sample_features[has_voxel_mask] = voxel_features[voxel_indices[has_voxel_mask]]
        sample_features = torch.cat([sample_features, img_features], dim=-1)

        sky_mask = sky_mask.flatten() if sky_mask.dim() > 1 else sky_mask
        raw_pt = self.to_gaussians(sample_features[~sky_mask])
        raw_sky = self.to_gaussians_sky(sample_features[sky_mask])
        gaussians = raw_pt[..., :15]
        gaussians_sky = raw_sky[..., :15]

        offsets = self.offset_act(gaussians[..., :3])
        opacities = self.opt_act(gaussians[..., 3:4])
        scales = self.scale_act(gaussians[..., 4:7])
        scales = torch.clamp_max(scales, self.cfg.get("max_scale", 0.01))
        rotations = self.rot_act(gaussians[..., 7:11])
        rgbs = self.rgb_act(gaussians[..., 11:14])
        if self._freeze_gaussian_rgb:
            rgbs = rgbs.detach()
        dynamic = nn.functional.sigmoid(gaussians[..., 14:15])
        common_means = means[~sky_mask]
        common_means = common_means + offsets
        gaussians = torch.cat(
            [common_means, rgbs, opacities, rotations, scales], dim=-1
        )

        sky_offsets = nn.functional.tanh(gaussians_sky[..., :3]) * self.cfg.get(
            "sky_gaussian_offset", 20
        )
        sky_opacities = self.opt_act(gaussians_sky[..., 3:4])
        # Side camera padding 区域强制 opacity=0（仅 padding，sky 与其他相机相同）
        # 使用 torch.where 避免 inplace 修改（会破坏 SigmoidBackward）
        if self.cfg.get("sky_mask_side_padding_opacity_zero", False) and cam_indices is not None and valid_mask is not None:
            padding_mask_flat = ~valid_mask  # True = padding
            sky_padding_mask = padding_mask_flat[sky_mask] & (
                (cam_indices[sky_mask] == 3) | (cam_indices[sky_mask] == 4)
            )
            sky_opacities = torch.where(
                sky_padding_mask.unsqueeze(1),
                torch.zeros_like(sky_opacities),
                sky_opacities,
            )
        sky_scales = self.scale_act(gaussians_sky[..., 4:7])
        sky_scales = torch.clamp_max(sky_scales, self.cfg.sky_gaussian_scale)
        sky_rotations = self.rot_act(gaussians_sky[..., 7:11])
        sky_rgbs = self.rgb_act(gaussians_sky[..., 11:14])
        if self._freeze_gaussian_rgb:
            sky_rgbs = sky_rgbs.detach()
        sky_dynamic = nn.functional.sigmoid(gaussians_sky[..., 14:15])
        sky_means_base = self._build_sky_means_base(
            means=means,
            sky_mask=sky_mask,
            rays_d=rays_d,
        )
        sky_means = sky_means_base + sky_offsets
        sky_gaussians = torch.cat(
            [sky_means, sky_rgbs, sky_opacities, sky_rotations, sky_scales], dim=-1
        )

        out = (
            gaussians,
            dynamic,
            sky_gaussians,
            sky_dynamic,
            sample_features,
            sky_mask,
            sky_means,
            extra_pixel_feats,
        )
        return out

    def get_dynamic_gs_mask(self, dynamics_region, means, intrinsics, camera2lidars):
        S, H, W = dynamics_region.shape
        dynamics_region_num = dynamics_region.flatten(start_dim=1).sum(dim=-1)
        out_mask = means.new_zeros((means.shape[0]), dtype=torch.bool)
        for i in range(S):
            if dynamics_region_num[i] == 0:
                continue
            inview_mask, pos2d = is_point_in_frustum_batch(
                means,
                intrinsics[i : i + 1],
                camera2lidars[i : i + 1, :3, :3],
                camera2lidars[i : i + 1, :3, 3],
                0.1,
                300,
                W,
                H,
            )
            inview_mask = inview_mask[:, 0]
            pos2d = pos2d[:, 0, :]
            pos2d = pos2d[inview_mask]
            d_mask = dynamics_region[i][pos2d[:, 1], pos2d[:, 0]]
            out_mask[inview_mask] = out_mask[inview_mask] | d_mask.to(torch.bool)
        return out_mask

    def process_history_gaussians(
        self,
        history_gaussians,
        history_gaussians_outview,
        current_lidar2world,
        history_lidar2world,
        intrinsics,
        camera2lidars,
        current_means,
        W,
        H,
    ):
        dtype, device = current_means.dtype, current_means.device
        if history_gaussians is None:
            history_gaussians_outview_save = torch.zeros(
                (0, 14), dtype=dtype, device=device
            )
        else:
            history_2_current = (current_lidar2world.inverse() @ history_lidar2world)[0]
            means = history_gaussians[:, :3]
            means_outview = history_gaussians_outview[:, :3]
            means = torch.cat(
                [means, torch.ones((means.shape[0], 1), dtype=dtype, device=device)],
                dim=-1,
            )
            means_outview = torch.cat(
                [
                    means_outview,
                    torch.ones((means_outview.shape[0], 1), dtype=dtype, device=device),
                ],
                dim=-1,
            )
            means = (history_2_current @ means.T).T
            means = means[:, :-1]
            means_outview = (history_2_current @ means_outview.T).T
            means_outview = means_outview[:, :-1]

            inview_mask, _ = is_point_in_frustum_batch(
                means,
                intrinsics,
                camera2lidars[:, :3, :3],
                camera2lidars[:, :3, 3],
                0.1,
                300,
                W,
                H,
            )
            inview_mask = inview_mask.any(dim=-1)

            history_gaussians[:, :3] = means
            history_gaussians_outview[:, :3] = means_outview
            history_gaussians_outview_save = torch.cat(
                [history_gaussians[~inview_mask], history_gaussians_outview], dim=0
            )

        return history_gaussians_outview_save

    def _create_dummy_render_output(self, B, S, H, W, device, dtype):
        """Create dummy render output when voxel input is empty to prevent crash.
        
        Returns:
            tuple: (render_out_dict, save_gaussians_list) matching normal return format
        """
        dummy_image = torch.zeros((B * S, 3, H, W), dtype=dtype, device=device)
        dummy_dyn = torch.zeros((B * S, 1, H, W), dtype=dtype, device=device)
        dummy_depth = torch.zeros((B * S, 1, H, W), dtype=dtype, device=device)
        dummy_depth_unscale = torch.zeros((B, S, H, W), dtype=dtype, device=device)
        
        render_out = {
            "image": dummy_image,
            "dyn": dummy_dyn,
            "depth": dummy_depth,
            "depth_map_unscale": dummy_depth_unscale,
            "voxel_opacities": torch.zeros((B,), dtype=dtype, device=device),
            "voxel_features": torch.zeros((B, 256), dtype=dtype, device=device),
        }
        
        # Return format matches normal return: (test_out, save_gaussians)
        # save_gaussians is a list of tensors (one per batch)
        dummy_save_gaussians = [torch.zeros((0, 15), dtype=dtype, device=device) for _ in range(B)]
        
        return render_out, dummy_save_gaussians


class Gaussians_Queue_v2(nn.Module):
    def __init__(self):
        super(Gaussians_Queue_v2, self).__init__()
        self.cache_fields = [
            "images",
            "gt_dynamics_mask",
            "pred_dynamics_mask",
            "lidars",
            "cam2lidars",
            "lidar2worlds",
            "voxel_features",
            "voxel_poses",
            "voxel_opacities",
            "gaussians",
            "gaussians_outview",
        ]
        self.voxel_fields = ["voxel_features", "voxel_poses", "voxel_opacities"]

        for field in self.cache_fields + ["scenes", "frames"]:
            setattr(self, field, None)

    def get(self, batch_size, scenes, frames):
        result = {field: [None] * batch_size for field in self.cache_fields}

        # first iter
        if self.scenes is None:
            return tuple(result[field] for field in self.cache_fields)

        for i in range(batch_size):
            if (
                self.scenes[i] == scenes[i]
                and int(self.frames[i]) == int(frames[i]) - 1
            ):
                for field in self.cache_fields:
                    cache_data = getattr(self, field)
                    if cache_data is not None:
                        result[field][i] = cache_data[i]

        return tuple(result[field] for field in self.cache_fields)

    def cache(self, batch_size, **kwargs):
        scenes = kwargs.pop("scenes")
        frames = kwargs.pop("frames")

        if self.scenes is None:
            for field in self.cache_fields + ["scenes", "frames"]:
                setattr(self, field, [None] * batch_size)

        for i in range(batch_size):
            self.scenes[i] = scenes[i]
            self.frames[i] = frames[i]

            for field in self.cache_fields:
                if field in self.voxel_fields:
                    continue

                data = kwargs.get(field)
                if data is not None:
                    getattr(self, field)[i] = (
                        data[i].detach()
                        if isinstance(data[i], torch.Tensor)
                        else data[i]
                    )

        voxel_poses = kwargs.get("voxel_poses")
        if voxel_poses is not None:
            for i in range(batch_size):
                mask = voxel_poses[:, 0] == i
                for field in self.voxel_fields:
                    data = kwargs.get(field)
                    if data is not None:
                        if field == "voxel_poses":
                            getattr(self, field)[i] = data[mask][:, 1:]
                        else:
                            getattr(self, field)[i] = data[mask]
