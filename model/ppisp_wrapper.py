"""
Bilateral-grid transformer modules.

This file keeps CHROMA bilateral-grid modules used by GaussianHead.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class CHROMAScaleDecoderBlock(nn.Module):
    """CHROMA-style decoder block at a single grid scale.

    Input is two parallel patch-token sets at the same (gy, gx) grid:
      - q  : canonical-image patches  (bilateral grid query stream)
      - kv : raw GT image patches     (target appearance reference)

    Per block (pre-norm):
      1) q self-attn      (canonical patches mix within the same view+scale)
      2) q -> kv cross-attn (canonical query attends raw GT key/value)
      3) FFN              on q
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.0, device: str = None):
        super().__init__()
        factory = {"device": device}
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True, **factory
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True, **factory
        )
        self.norm_q1 = nn.LayerNorm(d_model, **factory)
        self.norm_q2 = nn.LayerNorm(d_model, **factory)
        self.norm_q3 = nn.LayerNorm(d_model, **factory)
        self.norm_kv = nn.LayerNorm(d_model, **factory)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, **factory),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model, **factory),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        x = self.norm_q1(q)
        q = q + self.self_attn(x, x, x, need_weights=False)[0]
        qn = self.norm_q2(q)
        kvn = self.norm_kv(kv)
        q = q + self.cross_attn(qn, kvn, kvn, need_weights=False)[0]
        q = q + self.ffn(self.norm_q3(q))
        return q


class CHROMABilateralGridHead(nn.Module):
    """CHROMA-style multi-scale bilateral grid predictor.

    For each scale ``(gx, gy, gw)``:
      1) DINOv2 patch tokens of canonical and raw GT images are pooled to
         the per-scale spatial resolution ``(gy, gx)`` via adaptive avg pool.
      2) A small per-scale decoder runs ``num_decoder_layers`` blocks of
         ``canon-self-attn + canon-to-rawGT-cross-attn + FFN``.
      3) A linear head maps each refined canonical patch token to ``12 * gw``
         coefficients, reshaped into a bilateral grid of shape
         ``[12, gw, gy, gx]``.

    Reference cameras (``ref_cam_ids``) bypass the head entirely and get
    identity grids — only non-reference cameras receive learned grids.
    """

    def __init__(
        self,
        num_cameras: int,
        token_dim: int,
        num_decoder_layers: int,
        num_heads: int,
        ff_dim: int,
        patch_feat_dim: int,
        grid_sizes: list,
        guidance_factors: list,
        ref_cam_ids: Optional[list] = None,
        kv_mode: str = "residual",          # "residual" -> kv = raw_GT - canon
                                            # "raw_gt"   -> legacy: kv = raw_GT
        use_cam_id_embed: bool = True,      # whether to add cam_id_embed to q stream
        device: str = None,
    ) -> None:
        super().__init__()
        factory = {"device": device}
        self.num_cameras = int(num_cameras)
        self.token_dim = int(token_dim)
        self.patch_feat_dim = int(patch_feat_dim)
        self.grid_sizes = [tuple(int(x) for x in t) for t in grid_sizes]
        self.guidance_factors = [int(x) for x in guidance_factors]
        ref_set = set(int(c) for c in (ref_cam_ids or []))
        # Non-reference cameras get learned grids; reference cameras get identity.
        self.predict_cam_set = sorted(set(range(self.num_cameras)) - ref_set)
        self.ref_cam_set = sorted(ref_set)
        self.kv_mode = str(kv_mode).strip().lower()
        if self.kv_mode not in ("residual", "raw_gt"):
            self.kv_mode = "residual"
        self.use_cam_id_embed = bool(use_cam_id_embed)

        # Resolve attention head count divides token_dim cleanly.
        h = max(1, int(num_heads))
        if self.token_dim % h != 0:
            for hh in range(min(h, self.token_dim), 0, -1):
                if self.token_dim % hh == 0:
                    h = hh
                    break
        self.num_heads = h
        self.num_decoder_layers = max(1, int(num_decoder_layers))

        # Shared patch projection (used by both q stream and -- in legacy
        # kv_mode -- the kv stream too).
        self.patch_proj = nn.Linear(self.patch_feat_dim, self.token_dim, **factory)
        self.patch_norm = nn.LayerNorm(self.token_dim, **factory)
        self.cam_id_embed = (
            nn.Embedding(self.num_cameras, self.token_dim, **factory)
            if self.use_cam_id_embed else None
        )

        # When kv_mode == "residual", use a separate projection on the
        # raw_GT - canon residual to keep its statistics independent of the
        # canonical query stream.
        if self.kv_mode == "residual":
            self.residual_proj = nn.Linear(self.patch_feat_dim, self.token_dim, **factory)
            self.residual_norm = nn.LayerNorm(self.token_dim, **factory)
        else:
            self.residual_proj = None
            self.residual_norm = None

        # Per-scale learned 2D positional embedding (small, scale-specific).
        self.pos_embeds = nn.ParameterList()
        # Per-scale decoder + linear grid head.
        self.scale_decoders = nn.ModuleList()
        self.scale_grid_heads = nn.ModuleList()
        for (gx, gy, gw) in self.grid_sizes:
            self.pos_embeds.append(
                nn.Parameter(torch.zeros(1, gy * gx, self.token_dim, device=device))
            )
            self.scale_decoders.append(
                nn.ModuleList([
                    CHROMAScaleDecoderBlock(
                        self.token_dim, self.num_heads, ff_dim, device=device,
                    )
                    for _ in range(self.num_decoder_layers)
                ])
            )
            self.scale_grid_heads.append(
                nn.Linear(self.token_dim, 12 * gw, **factory)
            )

        self.register_buffer(
            "_ident12",
            torch.tensor(
                [1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0],
                dtype=torch.float32,
            ),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        # Identity grid output at init: head weights = 0, head bias = identity
        # affine repeated across the gw guidance bins.
        for i, head in enumerate(self.scale_grid_heads):
            gx, gy, gw = self.grid_sizes[i]
            nn.init.zeros_(head.weight)
            ident_bias = self._ident12.view(12, 1).expand(12, gw).reshape(-1)
            head.bias.data.copy_(ident_bias)
        for pe in self.pos_embeds:
            nn.init.zeros_(pe)

    @staticmethod
    def _pool_patches(
        patches_flat: torch.Tensor, P_h: int, P_w: int, target_h: int, target_w: int
    ) -> torch.Tensor:
        """Reshape [N, P_h*P_w, D] -> [N, target_h*target_w, D] via avg-pool."""
        N, P, D = patches_flat.shape
        assert P == P_h * P_w, f"patch count mismatch: P={P}, P_h*P_w={P_h*P_w}"
        x = patches_flat.transpose(1, 2).reshape(N, D, P_h, P_w)
        if (target_h, target_w) != (P_h, P_w):
            x = F.adaptive_avg_pool2d(x, (target_h, target_w))
        return x.flatten(2).transpose(1, 2)

    def forward(
        self,
        raw_gt_patches: torch.Tensor,
        canon_patches: torch.Tensor,
        view_cam_ids_2d: torch.Tensor,
        patch_h: int,
        patch_w: int,
    ) -> dict:
        """
        Args:
            raw_gt_patches: [B, S, P, patch_feat_dim] DINOv2 patch tokens of raw GT.
            canon_patches:  [B, S, P, patch_feat_dim] DINOv2 patch tokens of canonical
                            (the bilateral-grid query stream).
            view_cam_ids_2d: [B, S] camera id per current-frame view.
            patch_h, patch_w: full-resolution DINOv2 patch grid.

        Cross-attention key/value source depends on ``self.kv_mode``:
          - "residual": kv = projected (raw_GT - canon) patch residual; this
            is the *delta appearance* between target and canonical, which
            usually gives the cleanest learning signal.
          - "raw_gt"  : kv = projected raw_GT patches (legacy behaviour).

        Returns:
            dict with:
              "grids":            list of [B*S, 12, gw, gy, gx] per scale.
              "grid_sizes":       list of (gx, gy, gw).
              "guidance_factors": list of int.
        """
        if raw_gt_patches.dim() != 4 or canon_patches.dim() != 4:
            raise ValueError(
                "CHROMABilateralGridHead expects raw_gt/canon patches with shape "
                f"[B, S, P, D]; got {tuple(raw_gt_patches.shape)} / {tuple(canon_patches.shape)}"
            )
        B, S, P, D_in = raw_gt_patches.shape
        device = raw_gt_patches.device
        N = B * S

        raw_flat = raw_gt_patches.reshape(N, P, D_in)
        canon_flat = canon_patches.reshape(N, P, D_in)

        # ─── q stream: canonical patches (always carries content + optional cam id) ───
        canon_proj = self.patch_norm(self.patch_proj(canon_flat))
        if self.use_cam_id_embed and self.cam_id_embed is not None:
            cam_ids_flat = view_cam_ids_2d.reshape(N).to(device=device, dtype=torch.long)
            cam_emb = self.cam_id_embed(cam_ids_flat).view(N, 1, self.token_dim)
            canon_proj = canon_proj + cam_emb
        else:
            cam_ids_flat = view_cam_ids_2d.reshape(N).to(device=device, dtype=torch.long)

        # ─── kv stream: residual or raw GT, no cam id (avoid shortcut) ───
        if self.kv_mode == "residual":
            residual_flat = raw_flat - canon_flat
            kv_proj_full = self.residual_norm(self.residual_proj(residual_flat))
        else:  # "raw_gt"
            kv_proj_full = self.patch_norm(self.patch_proj(raw_flat))

        # Mask of views whose grids should be learned (non-reference cameras).
        if len(self.predict_cam_set) > 0:
            predict_set = torch.tensor(
                list(self.predict_cam_set), device=device, dtype=torch.long
            )
            learn_mask = (cam_ids_flat.unsqueeze(-1) == predict_set.unsqueeze(0)).any(dim=-1)
        else:
            learn_mask = torch.zeros(N, device=device, dtype=torch.bool)
        any_learn = bool(learn_mask.any().item()) if learn_mask.numel() > 0 else False

        grids: list = []
        with torch.no_grad():
            ident_grid_default = self._ident12.view(12, 1, 1, 1)

        for i, ((gx, gy, gw), blocks, head, pe) in enumerate(zip(
            self.grid_sizes, self.scale_decoders, self.scale_grid_heads, self.pos_embeds,
        )):
            T = gy * gx
            # Pool both streams to this scale's spatial resolution.
            kv_scale = self._pool_patches(kv_proj_full, patch_h, patch_w, gy, gx)
            canon_scale = self._pool_patches(canon_proj, patch_h, patch_w, gy, gx)

            # Add scale-specific positional embedding (shared between q/kv).
            kv_scale = kv_scale + pe
            canon_scale = canon_scale + pe

            # Run learned grids only on non-reference views (saves compute).
            if any_learn:
                q = canon_scale[learn_mask]  # [N_learn, T, D]
                kv = kv_scale[learn_mask]
                for blk in blocks:
                    q = blk(q, kv)
                preds = head(q)  # [N_learn, T, 12*gw]
                preds = preds.reshape(-1, gy, gx, 12, gw).permute(0, 3, 4, 1, 2).contiguous()
            else:
                preds = canon_scale.new_zeros(0, 12, gw, gy, gx)

            # Assemble final grid: identity for ref cams, learned for non-ref cams.
            grid_full = ident_grid_default.expand(N, 12, gw, gy, gx).to(canon_scale.dtype).clone()
            if any_learn:
                grid_full[learn_mask] = preds
            grids.append(grid_full)

        return {
            "grids": grids,
            "grid_sizes": list(self.grid_sizes),
            "guidance_factors": list(self.guidance_factors),
        }


