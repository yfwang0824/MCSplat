import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from dataset.typing import *


_gsplat_import_error = None
_supported_arch_tokens = {
    "7.0",
    "7.5",
    "8.0",
    "8.6",
    "8.7",
    "8.9",
    "9.0",
    "7.5+PTX",
    "8.0+PTX",
    "8.6+PTX",
    "8.9+PTX",
    "9.0+PTX",
}
_fallback_arch_list = "8.0;8.6;8.9;9.0+PTX"
_raw_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
if _raw_arch_list.strip():
    _tokens = _raw_arch_list.replace(";", " ").split()
    _filtered_tokens = [tok for tok in _tokens if tok in _supported_arch_tokens]
    if len(_filtered_tokens) == 0:
        os.environ["TORCH_CUDA_ARCH_LIST"] = _fallback_arch_list
    elif len(_filtered_tokens) != len(_tokens):
        os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(_filtered_tokens)
else:
    # Avoid JIT build failure on very new GPUs (e.g. sm_100) that older
    # torch cpp_extension cannot parse; PTX fallback keeps kernels runnable.
    os.environ["TORCH_CUDA_ARCH_LIST"] = _fallback_arch_list
try:
    from gsplat.rendering import (
        lidar_rasterization as gsplat_lidar_rasterization,
        rasterization as gsplat_rasterization,
    )
except Exception as _err:
    try:
        _repo_root = Path(__file__).resolve().parents[2]
        _splatad_root = _repo_root / "splatad"
        if str(_splatad_root) not in sys.path:
            sys.path.insert(0, str(_splatad_root))
        for _mod_name in list(sys.modules.keys()):
            if _mod_name == "gsplat" or _mod_name.startswith("gsplat."):
                del sys.modules[_mod_name]
        from gsplat.rendering import (
            lidar_rasterization as gsplat_lidar_rasterization,
            rasterization as gsplat_rasterization,
        )
    except Exception as _err2:
        gsplat_rasterization = None
        gsplat_lidar_rasterization = None
        _gsplat_import_error = _err2


def convert_pose(C2W):
    # flip_yz = torch.eye(4, device=C2W.device)
    # flip_yz[1, 1] = -1
    # flip_yz[2, 2] = -1
    # C2W = torch.matmul(C2W, flip_yz)
    # TODO: As we always use the opencv corrdinate system, we need to flip the y and z axis
    return C2W


def getProjectionMatrixK(K, H, W, znear, zfar, device="cuda"):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    s = K[0, 1]

    P = torch.zeros((4, 4), device=device)

    z_sign = 1.0

    P[0, 0] = 2 * fx / W
    P[0, 1] = 2 * s / W
    P[0, 2] = -1 + 2 * (cx / W)

    P[1, 1] = 2 * fy / H
    P[1, 2] = -1 + 2 * (cy / H)

    P[2, 2] = z_sign * (zfar + znear) / (zfar - znear)
    P[2, 3] = -1 * z_sign * 2 * zfar * znear / (zfar - znear)
    P[3, 2] = z_sign

    return P


def get_cam_info_gaussian_v2(c2w, K, H, W, znear, zfar):
    c2w = convert_pose(c2w)
    world_view_transform = torch.inverse(c2w.float())

    world_view_transform = world_view_transform.transpose(0, 1).cuda().float()
    projection_matrix = (
        getProjectionMatrixK(K, H, W, znear, zfar)
        .transpose(0, 1)
        .cuda()
    ).float()
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0).float()
    camera_center = world_view_transform.inverse()[3, :3]

    return world_view_transform, full_proj_transform, camera_center



C0 = 0.28209479177387814
def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    return sh * C0 + 0.5


def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


class GaussianRenderer_dyn:
    '''
    compared to GaussianRenderer_V2:
    We add render of dynamic score
    '''
    def __init__(
        self, 
        resolution: list = [512, 512],
        znear: float = 0.1,
        zfar: float = 100.0, 
        renderer_type: str = "gsplat",
        **kwargs,
    ):  
        self.renderer_type = renderer_type

        self.resolution = resolution
        self.znear = znear
        self.zfar = zfar
        self.bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        if gsplat_rasterization is None:
            raise ImportError(
                "Failed to import gsplat renderer. "
                "Please ensure `splatad/gsplat` is available and buildable."
            ) from _gsplat_import_error

        self.setup_functions()


    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def _normalize_hw_arg(self, value, num_views: int, default_value: int):
        if value is None:
            return [int(default_value)] * num_views
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return [int(value.item())] * num_views
            return [int(v.item()) for v in value.reshape(-1)]
        if isinstance(value, (list, tuple)):
            if len(value) == 1:
                return [int(value[0])] * num_views
            return [int(v) for v in value]
        return [int(value)] * num_views

    def _build_intrinsics_from_fov(
        self,
        fovx: torch.Tensor,
        fovy: torch.Tensor,
        widths: list,
        heights: list,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        intrinsics = []
        for idx in range(fovx.shape[0]):
            width_i = float(widths[idx])
            height_i = float(heights[idx])
            fx = width_i / (2.0 * torch.tan(fovx[idx] * 0.5))
            fy = height_i / (2.0 * torch.tan(fovy[idx] * 0.5))
            cx = width_i * 0.5
            cy = height_i * 0.5
            k = torch.tensor(
                [
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0],
                ],
                device=device,
                dtype=dtype,
            )
            intrinsics.append(k)
        return torch.stack(intrinsics, dim=0)

    def render(
        self, 
        gaussians: Float[Tensor, "B N F"], 
        c2w: Float[Tensor, "B V 4 4"],
        fovx: Float[Tensor, "B V"] = None,
        fovy: Float[Tensor, "B V"] = None,
        rays_o: Float[Tensor, "B V H W 3"] = None,
        rays_d: Float[Tensor, "B V H W 3"] = None,
        bg_color: Float[Tensor, "... 3"] = None, 
        scale_modifier: float = 1.,
        K=None,
        H=None,
        W=None,
        semantics=None,
        semantics_per_view=None,
        colors_per_view=None,
        enable_cuda_fisher=False,
        velocity=None,
        frame_offsets=None,
        extra_feats_per_view=None,
    ):
        # keep API compatibility, but CUDA fisher buffer is no longer used in gsplat path
        _ = enable_cuda_fisher
        assert fovx is not None or fovy is not None
        if fovx is None:
            fovx = fovy
        if fovy is None:
            fovy = fovx

        device = gaussians.device
        dtype = gaussians.dtype
        num_views = c2w.shape[0]

        means3d = gaussians[:, 0:3].contiguous().float()
        rgbs = gaussians[:, 3:6].contiguous().float()
        opacities = gaussians[:, 6].contiguous().float()
        rotations = gaussians[:, 7:11].contiguous().float()
        scales = gaussians[:, 11:14].contiguous().float()

        if K is None:
            heights = self._normalize_hw_arg(H, num_views, self.resolution[0])
            widths = self._normalize_hw_arg(W, num_views, self.resolution[1])
            K = self._build_intrinsics_from_fov(
                fovx=fovx.float(),
                fovy=fovy.float(),
                widths=widths,
                heights=heights,
                device=device,
                dtype=torch.float32,
            )
        else:
            K = K.float()
            heights = self._normalize_hw_arg(H, num_views, self.resolution[0])
            widths = self._normalize_hw_arg(W, num_views, self.resolution[1])

        if semantics_per_view is not None:
            semantic_per_view = semantics_per_view.contiguous().float()
            if semantic_per_view.dim() == 2:
                semantic_per_view = semantic_per_view.unsqueeze(-1)
        elif semantics is not None:
            semantic = semantics.contiguous().float()
            if semantic.dim() == 1:
                semantic = semantic.unsqueeze(-1)
            semantic_per_view = semantic.unsqueeze(0).expand(num_views, -1, -1)
        else:
            semantic_per_view = torch.zeros(
                num_views,
                means3d.shape[0],
                1,
                device=device,
                dtype=torch.float32,
            )

        if colors_per_view is not None:
            colors_per_view = colors_per_view.contiguous().float()
        else:
            colors_per_view = rgbs.unsqueeze(0).expand(num_views, -1, -1)

        # Optional extra per-Gaussian feature channels (e.g. LiDAR intensity logit)
        # to be alpha-blended jointly with RGB in the same rasterization pass.
        extra_dim = 0
        if extra_feats_per_view is not None:
            extra_feats_per_view = extra_feats_per_view.float()
            if extra_feats_per_view.dim() == 2:
                extra_feats_per_view = extra_feats_per_view.unsqueeze(0).expand(
                    num_views, -1, -1
                )
            extra_feats_per_view = extra_feats_per_view.contiguous()
            extra_dim = int(extra_feats_per_view.shape[-1])

        bg = self.bg_color if bg_color is None else bg_color
        if isinstance(bg, torch.Tensor):
            bg = bg.to(device=device, dtype=torch.float32)
        else:
            bg = torch.tensor(bg, device=device, dtype=torch.float32)
        if bg.dim() == 1:
            bg = bg.unsqueeze(0)
        bg = bg[:, :3]

        rendered_images = []
        rendered_dyns = []
        rendered_depths = []
        rendered_alphas = []
        rendered_radii = []
        rendered_extra = [] if extra_dim > 0 else None
        for view_idx in range(num_views):
            means3d_view = means3d
            if velocity is not None and frame_offsets is not None:
                dt = float(frame_offsets[view_idx].item()) if frame_offsets.dim() >= 1 else 0.0
                if dt != 0.0:
                    means3d_view = means3d + velocity * dt

            viewmat = torch.inverse(convert_pose(c2w[view_idx]).float()).unsqueeze(0)
            colors_view = colors_per_view[view_idx]
            semantics_view = semantic_per_view[view_idx]
            if semantics_view.dim() == 1:
                semantics_view = semantics_view.unsqueeze(-1)
            feat_list = [colors_view, semantics_view]
            if extra_dim > 0:
                feat_list.append(extra_feats_per_view[view_idx])
            gs_features = torch.cat(feat_list, dim=-1)

            bg_view = torch.cat(
                [
                    bg[0],
                    torch.zeros(
                        1 + extra_dim, device=device, dtype=torch.float32
                    ),
                ],
                dim=0,
            ).unsqueeze(0)

            render_colors, render_alphas, meta = gsplat_rasterization(
                means=means3d_view,
                quats=rotations,
                scales=scales * float(scale_modifier),
                opacities=opacities,
                colors=gs_features,
                velocities=None,
                viewmats=viewmat,
                Ks=K[view_idx : view_idx + 1],
                width=int(widths[view_idx]),
                height=int(heights[view_idx]),
                near_plane=float(self.znear),
                far_plane=float(self.zfar),
                packed=False,
                backgrounds=bg_view,
                render_mode="RGB+D",
            )

            render_colors = render_colors[0]  # [H, W, 4+extra_dim+1]
            image = render_colors[..., :3].permute(2, 0, 1).contiguous()
            dyn = render_colors[..., 3:4].permute(2, 0, 1).contiguous()
            if extra_dim > 0:
                extra = render_colors[..., 4 : 4 + extra_dim].permute(
                    2, 0, 1
                ).contiguous()
                rendered_extra.append(extra)
            depth = render_colors[..., 4 + extra_dim : 5 + extra_dim].permute(
                2, 0, 1
            ).contiguous()
            # Accumulated opacity [1,H,W]; useful for expected-depth = depth / alpha.
            alpha = render_alphas[0].permute(2, 0, 1).contiguous()
            radii = meta["radii"][0]
            if radii.dim() == 2:
                radii = radii.max(dim=-1).values
            rendered_images.append(torch.clamp(image, min=0.0, max=1.0))
            rendered_dyns.append(dyn)
            rendered_depths.append(depth)
            rendered_alphas.append(alpha)
            rendered_radii.append(radii)

        images = torch.stack(rendered_images, dim=0).to(dtype=dtype)
        dyns = torch.stack(rendered_dyns, dim=0).to(dtype=dtype)
        depths = torch.stack(rendered_depths, dim=0).to(dtype=dtype)
        alphas = torch.stack(rendered_alphas, dim=0).to(dtype=dtype)
        radii_stack = torch.stack(rendered_radii, dim=0)
        extra_stack = (
            torch.stack(rendered_extra, dim=0).to(dtype=dtype)
            if extra_dim > 0
            else None
        )
        out = {
            "image": images,
            "dyn": dyns,
            "depth": depths,
            "alpha": alphas,
            "radii": radii_stack,
        }
        if extra_stack is not None:
            # Blended LiDAR intensity logit (and any other extra channels).
            out["lidar_intensity"] = extra_stack
        return out

    def render_scalar_feature_maps(
        self,
        gaussians: torch.Tensor,
        scalar_per_view: torch.Tensor,
        c2w: torch.Tensor,
        fovx: torch.Tensor = None,
        fovy: torch.Tensor = None,
        K: torch.Tensor = None,
        H=None,
        W=None,
        bg_color: torch.Tensor = None,
        scale_modifier: float = 1.0,
    ) -> torch.Tensor:
        """Render per-Gaussian scalar fields to per-pixel maps via gsplat."""
        assert gaussians.dim() == 2 and gaussians.shape[1] >= 14
        assert scalar_per_view.dim() == 2
        num_views = c2w.shape[0]
        if fovx is None:
            fovx = fovy
        if fovy is None:
            fovy = fovx
        assert scalar_per_view.shape[0] == num_views
        assert scalar_per_view.shape[1] == gaussians.shape[0]

        means3d = gaussians[:, 0:3].contiguous().float()
        opacities = gaussians[:, 6].contiguous().float()
        rotations = gaussians[:, 7:11].contiguous().float()
        scales = gaussians[:, 11:14].contiguous().float()

        if K is None:
            heights = self._normalize_hw_arg(H, num_views, self.resolution[0])
            widths = self._normalize_hw_arg(W, num_views, self.resolution[1])
            K = self._build_intrinsics_from_fov(
                fovx=fovx.float(),
                fovy=fovy.float(),
                widths=widths,
                heights=heights,
                device=means3d.device,
                dtype=torch.float32,
            )
        else:
            K = K.float()
            heights = self._normalize_hw_arg(H, num_views, self.resolution[0])
            widths = self._normalize_hw_arg(W, num_views, self.resolution[1])

        out_maps = []
        for view_idx in range(num_views):
            viewmat = torch.inverse(convert_pose(c2w[view_idx]).float()).unsqueeze(0)
            colors = scalar_per_view[view_idx][:, None].contiguous().float()
            bg_view = torch.zeros(1, 1, device=means3d.device, dtype=torch.float32)
            render_colors, _, _ = gsplat_rasterization(
                means=means3d,
                quats=rotations,
                scales=scales * float(scale_modifier),
                opacities=opacities,
                colors=colors,
                velocities=None,
                viewmats=viewmat,
                Ks=K[view_idx : view_idx + 1],
                width=int(widths[view_idx]),
                height=int(heights[view_idx]),
                near_plane=float(self.znear),
                far_plane=float(self.zfar),
                packed=False,
                backgrounds=bg_view,
                render_mode="RGB",
            )
            out_maps.append(render_colors[0, ..., :1].permute(2, 0, 1).contiguous())

        return torch.stack(out_maps, dim=0)

    def render_lidar(
        self,
        gaussians: torch.Tensor,
        lidar_supervision: Dict,
        lidar_features: torch.Tensor = None,
        velocity: torch.Tensor = None,
    ) -> Dict:
        """Render LiDAR supervision maps using gsplat lidar rasterization."""
        if gsplat_lidar_rasterization is None:
            raise ImportError(
                "LiDAR rendering requires gsplat lidar_rasterization."
            ) from _gsplat_import_error
        if not isinstance(lidar_supervision, dict):
            return {}

        means3d = gaussians[:, 0:3].contiguous().float()
        opacities = gaussians[:, 6].contiguous().float()
        rotations = gaussians[:, 7:11].contiguous().float()
        scales = gaussians[:, 11:14].contiguous().float()

        viewmats = lidar_supervision.get("viewmats", None)
        raster_pts = lidar_supervision.get("raster_pts", None)
        tile_elev_bounds = lidar_supervision.get("tile_elevation_boundaries", None)
        if (
            not isinstance(viewmats, torch.Tensor)
            or not isinstance(raster_pts, torch.Tensor)
            or not isinstance(tile_elev_bounds, torch.Tensor)
        ):
            return {}

        viewmats = viewmats.float()
        raster_pts = raster_pts.float()
        tile_elev_bounds = tile_elev_bounds.float()
        num_lidars = viewmats.shape[0]
        num_gaussians = means3d.shape[0]

        if lidar_features is None:
            lidar_features = gaussians[:, 3:6].contiguous().float().mean(
                dim=-1, keepdim=True
            )
            lidar_features = lidar_features.unsqueeze(0).expand(num_lidars, -1, -1)
        elif lidar_features.dim() == 2:
            lidar_features = lidar_features.unsqueeze(0).expand(num_lidars, -1, -1)
        lidar_features = lidar_features.contiguous().float()
        if lidar_features.shape[:2] != (num_lidars, num_gaussians):
            raise ValueError(
                f"lidar_features shape mismatch: expected {(num_lidars, num_gaussians, 'D')}, "
                f"got {tuple(lidar_features.shape)}"
            )

        linear_velocity = lidar_supervision.get("linear_velocity", None)
        angular_velocity = lidar_supervision.get("angular_velocity", None)
        rolling_shutter_time = lidar_supervision.get("rolling_shutter_time", None)
        if linear_velocity is not None:
            linear_velocity = linear_velocity.float()
        if angular_velocity is not None:
            angular_velocity = angular_velocity.float()
        if rolling_shutter_time is not None:
            rolling_shutter_time = rolling_shutter_time.float()

        def _as_scalar(val, cast_fn, default):
            if val is None:
                return default
            if isinstance(val, torch.Tensor):
                if val.numel() == 0:
                    return default
                return cast_fn(val.reshape(-1)[0].item())
            return cast_fn(val)

        render_feats, render_alphas, _alpha_sum_until_points, meta = (
            gsplat_lidar_rasterization(
                means=means3d,
                quats=rotations,
                scales=scales,
                opacities=opacities,
                lidar_features=lidar_features,
                velocities=velocity,
                viewmats=viewmats,
                raster_pts=raster_pts,
                tile_elevation_boundaries=tile_elev_bounds,
                linear_velocity=linear_velocity,
                angular_velocity=angular_velocity,
                rolling_shutter_time=rolling_shutter_time,
                min_azimuth=_as_scalar(
                    lidar_supervision.get("min_azimuth", -180.0), float, -180.0
                ),
                max_azimuth=_as_scalar(
                    lidar_supervision.get("max_azimuth", 180.0), float, 180.0
                ),
                min_elevation=_as_scalar(
                    lidar_supervision.get("min_elevation", -80.0), float, -80.0
                ),
                max_elevation=_as_scalar(
                    lidar_supervision.get("max_elevation", 80.0), float, 80.0
                ),
                n_elevation_channels=_as_scalar(
                    lidar_supervision.get("n_elevation_channels", raster_pts.shape[1]),
                    int,
                    int(raster_pts.shape[1]),
                ),
                azimuth_resolution=_as_scalar(
                    lidar_supervision.get("azimuth_resolution", 0.1), float, 0.1
                ),
                tile_width=_as_scalar(
                    lidar_supervision.get("tile_width", 32), int, 32
                ),
                tile_height=_as_scalar(
                    lidar_supervision.get("tile_height", 8), int, 8
                ),
                near_plane=float(self.znear),
                far_plane=float(self.zfar),
                packed=False,
                # LOS (line-of-sight) supervision is disabled.
                # expected_range / feature_maps can still be consumed by loss
                # and visualization paths.
                compute_alpha_sum_until_points=False,
                use_depth_compensation=bool(
                    lidar_supervision.get("use_depth_compensation", True)
                ),
            )
        )

        feature_maps = render_feats[..., :-1].permute(0, 3, 1, 2).contiguous()
        expected_range = render_feats[..., -1:].permute(0, 3, 1, 2).contiguous()
        out = {
            "feature_maps": feature_maps,
            "expected_range": expected_range,
            "render_alphas": render_alphas.permute(0, 3, 1, 2).contiguous(),
            "meta": meta,
        }
        return out


    
  
  