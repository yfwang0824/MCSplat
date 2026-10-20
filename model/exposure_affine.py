"""Photometric canonicalization and bilateral-grid application for inference."""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple

def _project_points_to_camera(
    pts_world: torch.Tensor,
    camera_pose: torch.Tensor,
    intrinsics: torch.Tensor,
    H: int,
    W: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project 3D points in a shared frame to camera image plane.

    Args:
        pts_world: [N, 3] in shared frame (LiDAR/world/Pi3X common frame)
        camera_pose: [4, 4] or [3, 4], camera to shared frame
        intrinsics: [3, 3]
        H, W: image size

    Returns:
        uv: [N, 2] (u, v) pixel coords
        depth: [N] depth in camera
        valid: [N] bool
    """
    device = pts_world.device
    dtype = pts_world.dtype
    pts = pts_world.float()
    c2w = camera_pose.float()
    if c2w.shape[0] == 3:
        c2w = F.pad(c2w, (0, 0, 0, 1), value=0)
        c2w[3, 3] = 1.0
    world2cam = torch.linalg.inv(c2w)
    R = world2cam[:3, :3]
    t = world2cam[:3, 3]
    pts_cam = (pts @ R.T) + t

    z = pts_cam[:, 2]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    u = (pts_cam[:, 0] * fx / z) + cx
    v = (pts_cam[:, 1] * fy / z) + cy

    valid = (z > 0.01) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return torch.stack([u, v], dim=-1).to(dtype), z.to(dtype), valid

def _solve_diagonal_affine_with_prior(
    rgb_ref: torch.Tensor,
    rgb_src: torch.Tensor,
    sample_w: Optional[torch.Tensor],
    lam_a: float,
    lam_b: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Per-channel weighted diagonal affine fit with identity prior.

    Solves for each channel ``c`` independently:

        min_{a_c, b_c}  Σ_n w_n (a_c x_nc + b_c - y_nc)^2
                       + lam_a (a_c - 1)^2 + lam_b * b_c^2

    Returns a [3, 4] diagonal affine, or ``None`` if the input is degenerate.
    """
    N = rgb_src.shape[0]
    if N < 2:
        return None
    if sample_w is None:
        sample_w = rgb_src.new_ones(N)
    w = sample_w.float().clamp(min=0)
    sw = w.sum().clamp(min=1e-6)

    result = torch.eye(3, 4, device=device, dtype=dtype)
    for c in range(3):
        x = rgb_src[:, c].float()
        y = rgb_ref[:, c].float()
        S_x = (w * x).sum()
        S_y = (w * y).sum()
        S_xx = (w * x * x).sum()
        S_xy = (w * x * y).sum()
        # Symmetric 2x2 system with identity prior on (a-1) and b.
        M00 = S_xx + lam_a
        M01 = S_x
        M11 = sw + lam_b
        rhs0 = S_xy + lam_a
        rhs1 = S_y
        det = M00 * M11 - M01 * M01
        if not torch.isfinite(det) or float(det.abs().item()) < 1e-12:
            continue
        a_c = (rhs0 * M11 - M01 * rhs1) / det
        b_c = (M00 * rhs1 - rhs0 * M01) / det
        if not (torch.isfinite(a_c) and torch.isfinite(b_c)):
            continue
        result[c, c] = a_c.to(dtype)
        result[c, 3] = b_c.to(dtype)
    return result

def _solve_shared_gain_affine_with_prior(
    rgb_ref: torch.Tensor,
    rgb_src: torch.Tensor,
    sample_w: Optional[torch.Tensor],
    lam_g: float,
    lam_b: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Single shared gain across channels + per-channel bias, with identity prior.

    Solves:

        min_{g, b_c}  Σ_n Σ_c w_n (g x_nc + b_c - y_nc)^2
                    + lam_g (g - 1)^2 + lam_b Σ_c b_c^2

    Used as a degeneracy fallback when source chromaticity span is too narrow
    (so a per-channel diagonal solve would amplify per-channel noise into a
    visible hue shift).
    """
    N = rgb_src.shape[0]
    if N < 2:
        return None
    if sample_w is None:
        sample_w = rgb_src.new_ones(N)
    w = sample_w.float().clamp(min=0)
    sw = w.sum().clamp(min=1e-6)

    x = rgb_src.float()  # [N, 3]
    y = rgb_ref.float()  # [N, 3]

    w_col = w.unsqueeze(-1)
    S_xc = (w_col * x).sum(dim=0)        # [3]
    S_yc = (w_col * y).sum(dim=0)        # [3]
    S_xxc = (w_col * x * x).sum(dim=0)   # [3]
    S_xyc = (w_col * x * y).sum(dim=0)   # [3]

    denom_b = sw + lam_b
    A = S_xxc.sum() + lam_g - (S_xc * S_xc).sum() / denom_b
    B = S_xyc.sum() + lam_g - (S_yc * S_xc).sum() / denom_b
    if not torch.isfinite(A) or float(A.abs().item()) < 1e-12:
        return None
    g = B / A
    if not torch.isfinite(g):
        return None
    b = (S_yc - g * S_xc) / denom_b   # [3]
    if not torch.isfinite(b).all():
        return None

    result = torch.eye(3, 4, device=device, dtype=dtype)
    g_t = g.to(dtype)
    result[0, 0] = g_t
    result[1, 1] = g_t
    result[2, 2] = g_t
    result[0, 3] = b[0].to(dtype)
    result[1, 3] = b[1].to(dtype)
    result[2, 3] = b[2].to(dtype)
    return result

def _solve_affine_from_correspondences(
    rgb_ref: torch.Tensor,
    rgb_src: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    *,
    identity_prior_strength: float = 0.01,
    chroma_degenerate_std: float = 0.04,
    min_chroma_count: int = 64,
    irls_iters: int = 2,
    irls_huber_k: float = 1.5,
    irls_min_scale: float = 0.02,
) -> Optional[torch.Tensor]:
    """Robust per-channel affine ``rgb_ref_c ≈ a_c * rgb_src_c + b_c``.

    Improvements over plain LSQ:
      1) Tikhonov / identity prior (a_c → 1, b_c → 0) with adaptive strength
         that scales up when correspondences are few or chromatically narrow.
      2) Chromaticity-degeneracy fallback to a single shared gain across
         channels (avoids per-channel hue drift on near-monochromatic samples
         like asphalt that dominate Waymo LiDAR returns).
      3) 2-iter IRLS with Huber weights for outlier robustness.

    Returns a [3, 4] diagonal affine (off-diagonal = 0, no cross-channel
    mixing). Falls back to identity on degenerate input.
    """
    if rgb_src is None or rgb_ref is None:
        return None
    N = rgb_src.shape[0]
    if N < 2:
        return torch.eye(3, 4, device=device, dtype=dtype)

    rgb_src_f = rgb_src.float()
    rgb_ref_f = rgb_ref.float()

    # Adaptive identity prior: stronger when the sample is small or its color
    # span is narrow. base_lam scales with N so the prior weight stays
    # comparable to the data term in well-determined cases (where chroma_factor
    # and n_factor both reduce to 1).
    base_lam = float(identity_prior_strength) * float(N)
    chroma_std = rgb_src_f.std(dim=0)            # [3]
    chroma_min = float(chroma_std.min().clamp(min=1e-6).item())
    chroma_floor = float(chroma_degenerate_std)
    chroma_factor = max(1.0, (chroma_floor / max(chroma_min, 1e-6)) ** 2)
    n_factor = max(1.0, float(min_chroma_count) / max(N, 1))
    lam_a = base_lam * chroma_factor * n_factor
    lam_b = base_lam * chroma_factor * n_factor

    chroma_degenerate = (chroma_min < chroma_floor) or (N < int(min_chroma_count))

    # Initial fit (uniform weights).
    if chroma_degenerate:
        Ab = _solve_shared_gain_affine_with_prior(
            rgb_ref_f, rgb_src_f, None, lam_a, lam_b, device, dtype,
        )
    else:
        Ab = _solve_diagonal_affine_with_prior(
            rgb_ref_f, rgb_src_f, None, lam_a, lam_b, device, dtype,
        )
    if Ab is None:
        return torch.eye(3, 4, device=device, dtype=dtype)

    # IRLS: Huber-weighted refit using current residuals.
    for _ in range(int(max(0, irls_iters))):
        rgb_pred = (Ab[:3, :3].float() @ rgb_src_f.T).T + Ab[:3, 3].float()
        residual = (rgb_pred - rgb_ref_f).abs().mean(dim=-1)  # [N]
        if residual.numel() == 0:
            break
        scale = (1.4826 * residual.median()).clamp(min=float(irls_min_scale))
        thr = float(irls_huber_k) * scale
        sample_w = (thr / residual.clamp(min=thr)).clamp(max=1.0)
        if not torch.isfinite(sample_w).all():
            break
        if chroma_degenerate:
            Ab_new = _solve_shared_gain_affine_with_prior(
                rgb_ref_f, rgb_src_f, sample_w, lam_a, lam_b, device, dtype,
            )
        else:
            Ab_new = _solve_diagonal_affine_with_prior(
                rgb_ref_f, rgb_src_f, sample_w, lam_a, lam_b, device, dtype,
            )
        if Ab_new is None:
            break
        Ab = Ab_new

    return Ab

def compute_lidar_affine_blend(
    lidar_depth: torch.Tensor,
    images: torch.Tensor,
    intrinsics: torch.Tensor,
    camera2lidar: torch.Tensor,
    ref_cameras: Tuple[int, ...] = (1, 2),
    cam_ids: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    dynamics_mask: Optional[torch.Tensor] = None,
    exclude_dynamic: bool = True,
    sky_mask: Optional[torch.Tensor] = None,
    sky_points: Optional[torch.Tensor] = None,
    stride: int = 2,
    min_points: int = 50,
    use_rgb_refit: bool = True,
    rgb_residual_thresh: float = 0.05,
    rgb_refit_min_points: int = 20,
    depth_check_thresh: float = 1.5,
    return_stats: bool = False,
):
    """LiDAR affine with configurable reference cameras and blended correction.

    Two modes controlled by ref_cameras:

    Mode A: ref_cameras=(1, 2) — FL/FR as reference (default)
        FL, FR keep identity. FRONT gets per-pixel blend from FL->FRONT and
        FR->FRONT. SIDE_LEFT copies FL affine, SIDE_RIGHT copies FR affine.

    Mode B: ref_cameras=(0, 3, 4) — FRONT + side/back anchors as reference
        Cameras in ``ref_cameras`` keep identity. Target cameras are blended from
        adjacent references:
          - cam 1 (FL): blend 3->1 (left)  and 0->1 (right)
          - cam 2 (FR): blend 0->2 (left)  and 4->2 (right)
          - cam 5 (BACK, if present): blend 3->5 (left) and 4->5 (right)

    Returns:
        ground_affine: [B, S, H, W, 3, 4] per-pixel affine.
        sky_affine: [B, S, H, W, 3, 4] per-pixel sky affine.
    """
    device = lidar_depth.device
    dtype = images.dtype
    B, S, H, W = lidar_depth.shape

    identity = torch.eye(3, 4, device=device, dtype=dtype)
    ground_affine = identity.reshape(1, 1, 1, 1, 3, 4).expand(B, S, H, W, -1, -1).clone()
    sky_affine = identity.reshape(1, 1, 1, 1, 3, 4).expand(B, S, H, W, -1, -1).clone()

    stride_eff = max(1, int(stride))
    dynamic_filter = bool(exclude_dynamic) and (dynamics_mask is not None)
    ref_set = set(int(c) for c in ref_cameras)

    stats_info = {
        "num_correspondences": torch.zeros(B, S, device=device, dtype=dtype),
        "num_inliers": torch.zeros(B, S, device=device, dtype=dtype),
    }

    def _resolve_local(cam_ids_row, target_cam_id):
        if isinstance(cam_ids_row, torch.Tensor) and cam_ids_row.numel() >= S:
            match = torch.nonzero(
                cam_ids_row[:S].to(device=device, dtype=torch.long) == int(target_cam_id),
                as_tuple=False,
            ).reshape(-1)
            if match.numel() > 0:
                return int(match[0].item())
        return int(target_cam_id) if 0 <= int(target_cam_id) < S else -1

    def _get_correspondences_and_solve(b_idx, ref_local, tgt_local):
        """Find LiDAR correspondences from ref->tgt and solve global affine.

        Returns ``(affine [3, 4], n_corr, uv_u_tgt_valid)`` where:
          - affine maps tgt colors to ref color space, i.e.
            ``rgb_ref ≈ A * rgb_tgt + b``;
          - ``uv_u_tgt_valid`` is the column-coordinate (u) of the inlier
            correspondences in the *target* image, used downstream to build a
            per-column coverage / trust weight for blending.
        """
        empty_uv = torch.zeros(0, device=device, dtype=dtype)
        def _remap_hw_indices(
            h_idx_src: torch.Tensor,
            w_idx_src: torch.Tensor,
            src_h: int,
            src_w: int,
            dst_h: int,
            dst_w: int,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            h_m = h_idx_src.float()
            w_m = w_idx_src.float()
            if src_h > 1 and dst_h > 1 and src_h != dst_h:
                h_m = h_m * (float(dst_h - 1) / float(src_h - 1))
            if src_w > 1 and dst_w > 1 and src_w != dst_w:
                w_m = w_m * (float(dst_w - 1) / float(src_w - 1))
            h_m = h_m.round().long().clamp(0, max(dst_h - 1, 0))
            w_m = w_m.round().long().clamp(0, max(dst_w - 1, 0))
            return h_m, w_m

        depth_ref = lidar_depth[b_idx, ref_local]
        has_depth = depth_ref[::stride_eff, ::stride_eff] > 0
        if not has_depth.any():
            return None, 0, empty_uv
        h_idx, w_idx = torch.where(has_depth)
        h_full = h_idx * stride_eff
        w_full = w_idx * stride_eff
        z_vals = depth_ref[h_full, w_full]

        keep = torch.ones(z_vals.shape[0], dtype=torch.bool, device=device)
        if dynamic_filter:
            dyn_ref = dynamics_mask[b_idx, ref_local]
            h_dyn, w_dyn = _remap_hw_indices(
                h_full, w_full, H, W, int(dyn_ref.shape[-2]), int(dyn_ref.shape[-1])
            )
            keep = keep & (~dyn_ref[h_dyn, w_dyn].bool())
        if valid_mask is not None:
            vm_ref = valid_mask[b_idx, ref_local]
            h_vm, w_vm = _remap_hw_indices(
                h_full, w_full, H, W, int(vm_ref.shape[-2]), int(vm_ref.shape[-1])
            )
            keep = keep & vm_ref[h_vm, w_vm].bool()
        if sky_mask is not None:
            sm_ref = sky_mask[b_idx, ref_local]
            h_sm, w_sm = _remap_hw_indices(
                h_full, w_full, H, W, int(sm_ref.shape[-2]), int(sm_ref.shape[-1])
            )
            keep = keep & (sm_ref[h_sm, w_sm] > 0.5)
        if not keep.any():
            return None, 0, empty_uv
        h_full, w_full, z_vals = h_full[keep], w_full[keep], z_vals[keep]

        fx = intrinsics[b_idx, ref_local, 0, 0]
        fy = intrinsics[b_idx, ref_local, 1, 1]
        cx = intrinsics[b_idx, ref_local, 0, 2]
        cy = intrinsics[b_idx, ref_local, 1, 2]
        x_cam = (w_full.float() - cx) * z_vals / fx
        y_cam = (h_full.float() - cy) * z_vals / fy
        pts_cam = torch.stack([x_cam, y_cam, z_vals], dim=-1)

        c2w_ref = camera2lidar[b_idx, ref_local].float()
        if c2w_ref.shape[0] == 3:
            c2w_ref = F.pad(c2w_ref, (0, 0, 0, 1), value=0)
            c2w_ref[3, 3] = 1.0
        pts_world = (pts_cam @ c2w_ref[:3, :3].T) + c2w_ref[:3, 3]

        uv_tgt, z_tgt, valid_proj = _project_points_to_camera(
            pts_world, camera2lidar[b_idx, tgt_local],
            intrinsics[b_idx, tgt_local], H, W,
        )
        if not valid_proj.any():
            return None, 0, empty_uv

        h_ref_v = h_full[valid_proj]
        w_ref_v = w_full[valid_proj]
        uv_tgt_v = uv_tgt[valid_proj]
        z_tgt_v = z_tgt[valid_proj]

        u_t = uv_tgt_v[:, 0].long().clamp(0, W - 1)
        v_t = uv_tgt_v[:, 1].long().clamp(0, H - 1)
        depth_tgt_at = lidar_depth[b_idx, tgt_local, v_t, u_t]
        keep2 = torch.ones(uv_tgt_v.shape[0], dtype=torch.bool, device=device)
        has_tgt_d = depth_tgt_at > 0
        if has_tgt_d.any():
            depth_ok = (depth_tgt_at[has_tgt_d] - z_tgt_v[has_tgt_d]).abs() < depth_check_thresh
            keep2[has_tgt_d] = depth_ok
        if valid_mask is not None:
            vm_tgt = valid_mask[b_idx, tgt_local]
            v_vm, u_vm = _remap_hw_indices(
                v_t, u_t, H, W, int(vm_tgt.shape[-2]), int(vm_tgt.shape[-1])
            )
            keep2 = keep2 & vm_tgt[v_vm, u_vm].bool()
        if dynamic_filter:
            dyn_tgt = dynamics_mask[b_idx, tgt_local]
            v_dyn, u_dyn = _remap_hw_indices(
                v_t, u_t, H, W, int(dyn_tgt.shape[-2]), int(dyn_tgt.shape[-1])
            )
            keep2 = keep2 & (~dyn_tgt[v_dyn, u_dyn].bool())
        n_corr = int(keep2.sum().item())
        if n_corr < min_points:
            return None, n_corr, empty_uv

        uv_tgt_keep = uv_tgt_v[keep2]
        img_ref = images[b_idx, ref_local]
        img_tgt = images[b_idx, tgt_local]
        H_img_ref, W_img_ref = int(img_ref.shape[-2]), int(img_ref.shape[-1])
        H_img_tgt, W_img_tgt = int(img_tgt.shape[-2]), int(img_tgt.shape[-1])

        h_ref_img, w_ref_img = _remap_hw_indices(
            h_ref_v[keep2], w_ref_v[keep2], H, W, H_img_ref, W_img_ref
        )
        rgb_ref = img_ref[:, h_ref_img, w_ref_img].T.float()
        uv_tgt_img = uv_tgt_keep.clone()
        if H_img_tgt > 1 and H > 1 and H_img_tgt != H:
            uv_tgt_img[:, 1] = uv_tgt_img[:, 1] * (float(H_img_tgt - 1) / float(H - 1))
        if W_img_tgt > 1 and W > 1 and W_img_tgt != W:
            uv_tgt_img[:, 0] = uv_tgt_img[:, 0] * (float(W_img_tgt - 1) / float(W - 1))
        denom = torch.tensor(
            [max(W_img_tgt - 1, 1), max(H_img_tgt - 1, 1)],
            device=device,
            dtype=dtype,
        )
        grid_tgt = (
            uv_tgt_img / denom
        ) * 2 - 1
        rgb_tgt = F.grid_sample(
            img_tgt.unsqueeze(0).float(),
            grid_tgt.unsqueeze(0).unsqueeze(2),
            mode="bilinear", padding_mode="zeros", align_corners=True,
        ).squeeze(0).permute(1, 2, 0).squeeze(1)

        Ab = _solve_affine_from_correspondences(rgb_ref, rgb_tgt, device, dtype)
        if Ab is None:
            return None, n_corr, uv_tgt_keep[:, 0].to(dtype)

        # Track which correspondences end up as inliers; if a refit is done,
        # restrict the coverage estimate to those inliers as well so we don't
        # claim trust over columns whose color residual was discarded.
        uv_u_inlier = uv_tgt_keep[:, 0].to(dtype)
        if use_rgb_refit and rgb_residual_thresh > 0:
            rgb_pred = (Ab[:3, :3] @ rgb_tgt.T).T + Ab[:3, 3]
            residual = (rgb_pred - rgb_ref).abs().mean(dim=-1)
            inlier = residual <= rgb_residual_thresh
            n_inlier = int(inlier.sum().item())
            if n_inlier >= max(2, rgb_refit_min_points):
                Ab2 = _solve_affine_from_correspondences(
                    rgb_ref[inlier], rgb_tgt[inlier], device, dtype,
                )
                if Ab2 is not None:
                    Ab = Ab2
                    uv_u_inlier = uv_tgt_keep[inlier, 0].to(dtype)

        return Ab, n_corr, uv_u_inlier

    def _solve_sky_global(b_idx, ref_local, tgt_local):
        if sky_mask is None or sky_points is None:
            return None
        sky_ref = sky_mask[b_idx, ref_local] <= 0.5
        if not sky_ref.any():
            return None
        sky_h, sky_w = torch.where(sky_ref[::stride_eff, ::stride_eff])
        if sky_h.numel() == 0:
            return None
        sky_h_full = sky_h * stride_eff
        sky_w_full = sky_w * stride_eff
        if valid_mask is not None:
            vm_ref = valid_mask[b_idx, ref_local]
            h_vm_ref, w_vm_ref = int(vm_ref.shape[-2]), int(vm_ref.shape[-1])
            h_ref_m = sky_h_full
            w_ref_m = sky_w_full
            if H > 1 and h_vm_ref > 1 and h_vm_ref != H:
                h_ref_m = (
                    (h_ref_m.float() * (float(h_vm_ref - 1) / float(H - 1)))
                    .round()
                    .long()
                    .clamp(0, h_vm_ref - 1)
                )
            if W > 1 and w_vm_ref > 1 and w_vm_ref != W:
                w_ref_m = (
                    (w_ref_m.float() * (float(w_vm_ref - 1) / float(W - 1)))
                    .round()
                    .long()
                    .clamp(0, w_vm_ref - 1)
                )
            keep_ref_valid = vm_ref[h_ref_m, w_ref_m].bool()
            if not keep_ref_valid.any():
                return None
            sky_h_full = sky_h_full[keep_ref_valid]
            sky_w_full = sky_w_full[keep_ref_valid]
        pts_sky = sky_points[b_idx, ref_local, sky_h_full, sky_w_full, :]
        uv_sky, _, valid_sky = _project_points_to_camera(
            pts_sky, camera2lidar[b_idx, tgt_local],
            intrinsics[b_idx, tgt_local], H, W,
        )
        if not valid_sky.any():
            return None
        sky_h_ref = sky_h_full[valid_sky]
        sky_w_ref = sky_w_full[valid_sky]
        uv_sky_v = uv_sky[valid_sky]
        u_sk = uv_sky_v[:, 0].long().clamp(0, W - 1)
        v_sk = uv_sky_v[:, 1].long().clamp(0, H - 1)
        if sky_mask[b_idx, tgt_local].numel() > 0:
            sm_tgt = sky_mask[b_idx, tgt_local]
            h_sm_tgt, w_sm_tgt = int(sm_tgt.shape[-2]), int(sm_tgt.shape[-1])
            v_sk_m = v_sk
            u_sk_m = u_sk
            if H > 1 and h_sm_tgt > 1 and h_sm_tgt != H:
                v_sk_m = (
                    (v_sk.float() * (float(h_sm_tgt - 1) / float(H - 1)))
                    .round()
                    .long()
                    .clamp(0, h_sm_tgt - 1)
                )
            if W > 1 and w_sm_tgt > 1 and w_sm_tgt != W:
                u_sk_m = (
                    (u_sk.float() * (float(w_sm_tgt - 1) / float(W - 1)))
                    .round()
                    .long()
                    .clamp(0, w_sm_tgt - 1)
                )
            tgt_is_sky = sm_tgt[v_sk_m, u_sk_m] <= 0.5
            if valid_mask is not None:
                vm_tgt = valid_mask[b_idx, tgt_local]
                h_vm_tgt, w_vm_tgt = int(vm_tgt.shape[-2]), int(vm_tgt.shape[-1])
                v_vm = v_sk
                u_vm = u_sk
                if H > 1 and h_vm_tgt > 1 and h_vm_tgt != H:
                    v_vm = (
                        (v_sk.float() * (float(h_vm_tgt - 1) / float(H - 1)))
                        .round()
                        .long()
                        .clamp(0, h_vm_tgt - 1)
                    )
                if W > 1 and w_vm_tgt > 1 and w_vm_tgt != W:
                    u_vm = (
                        (u_sk.float() * (float(w_vm_tgt - 1) / float(W - 1)))
                        .round()
                        .long()
                        .clamp(0, w_vm_tgt - 1)
                    )
                tgt_is_sky = tgt_is_sky & vm_tgt[v_vm, u_vm].bool()
            if tgt_is_sky.sum() < min_points:
                return None
            rgb_ref_sky = images[b_idx, ref_local, :, sky_h_ref[tgt_is_sky], sky_w_ref[tgt_is_sky]].T.float()
            img_tgt = images[b_idx, tgt_local]
            H_img_tgt, W_img_tgt = int(img_tgt.shape[-2]), int(img_tgt.shape[-1])
            uv_sky_img = uv_sky_v[tgt_is_sky].clone()
            if H_img_tgt > 1 and H > 1 and H_img_tgt != H:
                uv_sky_img[:, 1] = uv_sky_img[:, 1] * (
                    float(H_img_tgt - 1) / float(H - 1)
                )
            if W_img_tgt > 1 and W > 1 and W_img_tgt != W:
                uv_sky_img[:, 0] = uv_sky_img[:, 0] * (
                    float(W_img_tgt - 1) / float(W - 1)
                )
            grid_sky = (
                uv_sky_img
                / torch.tensor(
                    [max(W_img_tgt - 1, 1), max(H_img_tgt - 1, 1)],
                    device=device,
                    dtype=dtype,
                )
            ) * 2 - 1
            rgb_tgt_sky = F.grid_sample(
                img_tgt.unsqueeze(0).float(),
                grid_sky.unsqueeze(0).unsqueeze(2),
                mode="bilinear", padding_mode="zeros", align_corners=True,
            ).squeeze(0).permute(1, 2, 0).squeeze(1)
            return _solve_affine_from_correspondences(rgb_ref_sky, rgb_tgt_sky, device, dtype)
        return None

    # Waymo / nuScenes-compatible ring anchors:
    #   3 -- 1 -- 0 -- 2 -- 4 -- 5(optional back)
    use_three_ref = (ref_set == {0, 3, 4})
    # Geometric prior weight ramp across image width (0 = leftmost column,
    # 1 = rightmost). Each blended camera has two reference sources and we
    # interpolate them with this ramp so the camera spatially closer to the
    # source dominates that side of the target image.
    alpha_w = torch.linspace(0.0, 1.0, W, device=device, dtype=dtype)

    def _blend_two_affines(
        A_left: torch.Tensor,    # [3, 4]
        A_right: torch.Tensor,   # [3, 4]
    ) -> torch.Tensor:
        """Pure geometric per-column blend of two global affines.

        ``A(u) = (1 - alpha_w(u)) * A_left + alpha_w(u) * A_right``

        Each affine is a *global* 3x4 transform that captures the color
        response difference between a pair of cameras and is physically valid
        across the whole target image (not only where the LiDAR samples were
        taken). Robustness against under-determined / chromatically narrow
        samples is handled inside ``_solve_affine_from_correspondences``
        (Tikhonov prior + chroma-degeneracy fallback to shared gain + IRLS),
        so we no longer modulate the blend by per-column LiDAR coverage:
        doing so would wrongly suppress the affine in regions that simply
        had no LiDAR returns (e.g. mid-FR between FRONT and SR overlap).

        Identity fallback for an entirely-failed solve is achieved by passing
        ``identity`` as ``A_left`` or ``A_right`` at the call site.
        """
        A_l = A_left.reshape(1, 3, 4)
        A_r = A_right.reshape(1, 3, 4)
        w = alpha_w.reshape(W, 1, 1)
        out_w = (1.0 - w) * A_l + w * A_r              # [W, 3, 4]
        return out_w.unsqueeze(0).expand(H, W, 3, 4)   # [H, W, 3, 4]

    def _is_affine_reliable(Ab, max_scale_dev=0.5, max_bias=0.25):
        for c in range(3):
            if abs(float(Ab[c, c].item()) - 1.0) > max_scale_dev:
                return False
            if abs(float(Ab[c, 3].item())) > max_bias:
                return False
        return True

    def _invert_affine_3x4(Ab: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Invert a color affine y = A x + b represented as [3, 4]."""
        if Ab is None:
            return None
        A = Ab[:3, :3].float()
        b = Ab[:3, 3].float()
        if (not torch.isfinite(A).all()) or (not torch.isfinite(b).all()):
            return None
        try:
            det = torch.det(A)
            if (not torch.isfinite(det)) or (abs(float(det.item())) < 1e-8):
                return None
            A_inv = torch.linalg.inv(A)
        except RuntimeError:
            return None
        if not torch.isfinite(A_inv).all():
            return None
        b_inv = -A_inv @ b
        out = torch.zeros((3, 4), device=device, dtype=dtype)
        out[:3, :3] = A_inv.to(dtype)
        out[:3, 3] = b_inv.to(dtype)
        return out

    def _solve_pair_bidirectional(
        b_idx: int,
        ref_local: int,
        tgt_local: int,
    ) -> Tuple[Optional[torch.Tensor], int]:
        """Solve tgt->ref affine with reverse-direction fallback.

        Primary solve uses ref->tgt correspondences (as before). If it fails,
        we try tgt->ref direction and invert the fitted affine, which is often
        more stable when one camera has denser overlap depth (e.g. FR vs FRONT).
        """
        if ref_local < 0 or tgt_local < 0:
            return None, 0
        Ab, n_corr, _ = _get_correspondences_and_solve(b_idx, ref_local, tgt_local)
        if Ab is not None:
            return Ab, int(n_corr)

        # Reverse-direction rescue: solve ref<-tgt counterpart then invert.
        Ab_rev, n_corr_rev, _ = _get_correspondences_and_solve(
            b_idx, tgt_local, ref_local
        )
        if Ab_rev is None:
            return None, int(max(n_corr, n_corr_rev))
        Ab_inv = _invert_affine_3x4(Ab_rev)
        if Ab_inv is None:
            return None, int(max(n_corr, n_corr_rev))
        # Keep fallback robust: reject obviously unstable inverses.
        if not _is_affine_reliable(Ab_inv, max_scale_dev=0.75, max_bias=0.35):
            return None, int(max(n_corr, n_corr_rev))
        return Ab_inv, int(max(n_corr, n_corr_rev))

    for b_idx in range(B):
        cam_ids_row = None
        if isinstance(cam_ids, torch.Tensor):
            cam_ids_t = cam_ids.to(device=device, dtype=torch.long)
            if cam_ids_t.dim() == 2 and cam_ids_t.shape[0] > b_idx and cam_ids_t.shape[1] >= S:
                cam_ids_row = cam_ids_t[b_idx, :S]
            elif cam_ids_t.dim() == 1 and cam_ids_t.shape[0] >= S:
                cam_ids_row = cam_ids_t[:S]

        front_local = _resolve_local(cam_ids_row, 0)
        fl_local = _resolve_local(cam_ids_row, 1)
        fr_local = _resolve_local(cam_ids_row, 2)
        sl_local = _resolve_local(cam_ids_row, 3)
        sr_local = _resolve_local(cam_ids_row, 4)
        back_local = _resolve_local(cam_ids_row, 5)

        if use_three_ref:
            # Mode B: refs={0,3,4}; blend FL/FR/(optional BACK)
            # FL: blend SL->FL (left) with FRONT->FL (right)
            Ab_sl_fl, n_sl_fl = _solve_pair_bidirectional(
                b_idx, sl_local, fl_local
            )
            affine_sl_to_fl = Ab_sl_fl if Ab_sl_fl is not None else identity.clone()

            Ab_f_fl, n_f_fl = _solve_pair_bidirectional(
                b_idx, front_local, fl_local
            )
            affine_front_to_fl = Ab_f_fl if Ab_f_fl is not None else identity.clone()

            if fl_local >= 0:
                blended_fl = _blend_two_affines(affine_sl_to_fl, affine_front_to_fl)
                ground_affine[b_idx, fl_local] = blended_fl.to(dtype)
                stats_info["num_correspondences"][b_idx, fl_local] = float(n_sl_fl + n_f_fl)

            # FR: blend FRONT->FR (left) with SR->FR (right)
            Ab_f_fr, n_f_fr = _solve_pair_bidirectional(
                b_idx, front_local, fr_local
            )
            affine_front_to_fr = Ab_f_fr if Ab_f_fr is not None else identity.clone()

            Ab_sr_fr, n_sr_fr = _solve_pair_bidirectional(
                b_idx, sr_local, fr_local
            )
            affine_sr_to_fr = Ab_sr_fr if Ab_sr_fr is not None else identity.clone()

            if fr_local >= 0:
                blended_fr = _blend_two_affines(affine_front_to_fr, affine_sr_to_fr)
                ground_affine[b_idx, fr_local] = blended_fr.to(dtype)
                stats_info["num_correspondences"][b_idx, fr_local] = float(n_f_fr + n_sr_fr)

            # BACK (nuScenes cam id 5): blend BL/SL->BACK (left) with BR/SR->BACK (right)
            Ab_sl_back, n_sl_back = _solve_pair_bidirectional(
                b_idx, sl_local, back_local
            )
            affine_sl_to_back = Ab_sl_back if Ab_sl_back is not None else identity.clone()

            Ab_sr_back, n_sr_back = _solve_pair_bidirectional(
                b_idx, sr_local, back_local
            )
            affine_sr_to_back = Ab_sr_back if Ab_sr_back is not None else identity.clone()

            if back_local >= 0:
                blended_back = _blend_two_affines(affine_sl_to_back, affine_sr_to_back)
                ground_affine[b_idx, back_local] = blended_back.to(dtype)
                stats_info["num_correspondences"][b_idx, back_local] = float(
                    n_sl_back + n_sr_back
                )

            # Sky affine (nuScenes mode):
            # - FL: blend SL->FL (left) with FRONT->FL (right)
            # - FR: blend FRONT->FR (left) with SR->FR (right)
            # If sky solves are unavailable, fall back to the corresponding
            # ground solves to keep behavior stable.
            if fl_local >= 0:
                sky_sl_fl = (
                    _solve_sky_global(b_idx, sl_local, fl_local)
                    if sl_local >= 0
                    else None
                )
                sky_f_fl = (
                    _solve_sky_global(b_idx, front_local, fl_local)
                    if front_local >= 0
                    else None
                )
                sky_from_sl_fl = (
                    sky_sl_fl if sky_sl_fl is not None else affine_sl_to_fl
                )
                sky_from_f_fl = (
                    sky_f_fl if sky_f_fl is not None else affine_front_to_fl
                )
                sky_fl = _blend_two_affines(sky_from_sl_fl, sky_from_f_fl)
                sky_affine[b_idx, fl_local] = sky_fl.to(dtype)

            if fr_local >= 0:
                sky_f_fr = (
                    _solve_sky_global(b_idx, front_local, fr_local)
                    if front_local >= 0
                    else None
                )
                sky_sr_fr = (
                    _solve_sky_global(b_idx, sr_local, fr_local)
                    if sr_local >= 0
                    else None
                )
                sky_from_f_fr = (
                    sky_f_fr if sky_f_fr is not None else affine_front_to_fr
                )
                sky_from_sr_fr = (
                    sky_sr_fr if sky_sr_fr is not None else affine_sr_to_fr
                )
                sky_fr = _blend_two_affines(sky_from_f_fr, sky_from_sr_fr)
                sky_affine[b_idx, fr_local] = sky_fr.to(dtype)

            # BACK sky:
            # - if both SL/SR sky solves exist: blend them;
            # - if only one side exists: trust that single sky solve globally
            #   (avoid mixing with potentially biased non-sky fallback);
            # - if neither exists: fallback to ground-based blend.
            if back_local >= 0:
                sky_sl_back = (
                    _solve_sky_global(b_idx, sl_local, back_local)
                    if sl_local >= 0
                    else None
                )
                sky_sr_back = (
                    _solve_sky_global(b_idx, sr_local, back_local)
                    if sr_local >= 0
                    else None
                )
                if sky_sl_back is not None and sky_sr_back is not None:
                    sky_back = _blend_two_affines(sky_sl_back, sky_sr_back)
                elif sky_sl_back is not None:
                    sky_back = (
                        sky_sl_back.reshape(1, 1, 3, 4)
                        .expand(H, W, -1, -1)
                    )
                elif sky_sr_back is not None:
                    sky_back = (
                        sky_sr_back.reshape(1, 1, 3, 4)
                        .expand(H, W, -1, -1)
                    )
                else:
                    sky_back = _blend_two_affines(affine_sl_to_back, affine_sr_to_back)
                sky_affine[b_idx, back_local] = sky_back.to(dtype)

        else:
            # Mode A (default): FL/FR as reference, blend FRONT
            fl_cam_id = int(ref_cameras[0]) if len(ref_cameras) > 0 else 1
            fr_cam_id = int(ref_cameras[1]) if len(ref_cameras) > 1 else 2

            Ab_fl, n_fl = (None, 0)
            if fl_local >= 0 and front_local >= 0:
                Ab_fl, n_fl, _ = _get_correspondences_and_solve(
                    b_idx, fl_local, front_local,
                )
            affine_from_fl = Ab_fl if Ab_fl is not None else identity.clone()

            Ab_fr, n_fr = (None, 0)
            if fr_local >= 0 and front_local >= 0:
                Ab_fr, n_fr, _ = _get_correspondences_and_solve(
                    b_idx, fr_local, front_local,
                )
            affine_from_fr = Ab_fr if Ab_fr is not None else identity.clone()

            if front_local >= 0:
                stats_info["num_correspondences"][b_idx, front_local] = float(n_fl + n_fr)
                blended = _blend_two_affines(affine_from_fl, affine_from_fr)
                ground_affine[b_idx, front_local] = blended.to(dtype)

            if sl_local >= 0 and fl_local >= 0:
                Ab_sl, n_sl, _ = _get_correspondences_and_solve(b_idx, fl_local, sl_local)
                if Ab_sl is not None and _is_affine_reliable(Ab_sl):
                    ground_affine[b_idx, sl_local] = Ab_sl.reshape(1, 1, 3, 4).expand(H, W, -1, -1).to(dtype)
                    stats_info["num_correspondences"][b_idx, sl_local] = float(n_sl)
                else:
                    ground_affine[b_idx, sl_local] = affine_from_fl.reshape(1, 1, 3, 4).expand(H, W, -1, -1).to(dtype)

            if sr_local >= 0 and fr_local >= 0:
                Ab_sr, n_sr, _ = _get_correspondences_and_solve(b_idx, fr_local, sr_local)
                if Ab_sr is not None and _is_affine_reliable(Ab_sr):
                    ground_affine[b_idx, sr_local] = Ab_sr.reshape(1, 1, 3, 4).expand(H, W, -1, -1).to(dtype)
                    stats_info["num_correspondences"][b_idx, sr_local] = float(n_sr)
                else:
                    ground_affine[b_idx, sr_local] = affine_from_fr.reshape(1, 1, 3, 4).expand(H, W, -1, -1).to(dtype)

            # Sky affine: per-pixel blend for FRONT, global for SIDE. If a
            # sky solve failed entirely we fall back to the ground affine for
            # that side, matching previous behavior.
            sky_fl_ab = _solve_sky_global(b_idx, fl_local, front_local) if fl_local >= 0 and front_local >= 0 else None
            sky_fr_ab = _solve_sky_global(b_idx, fr_local, front_local) if fr_local >= 0 and front_local >= 0 else None
            sky_from_fl = sky_fl_ab if sky_fl_ab is not None else affine_from_fl
            sky_from_fr = sky_fr_ab if sky_fr_ab is not None else affine_from_fr
            if front_local >= 0:
                sky_blended = _blend_two_affines(sky_from_fl, sky_from_fr)
                sky_affine[b_idx, front_local] = sky_blended.to(dtype)
            if sl_local >= 0:
                sky_affine[b_idx, sl_local] = ground_affine[b_idx, sl_local]
            if sr_local >= 0:
                sky_affine[b_idx, sr_local] = ground_affine[b_idx, sr_local]

    if return_stats:
        return ground_affine, sky_affine, stats_info
    return ground_affine, sky_affine

def apply_affine_to_images(
    images: torch.Tensor,
    ground_affine: torch.Tensor,
    sky_affine: Optional[torch.Tensor] = None,
    sky_mask: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply per-camera affine color correction, with separate transforms for ground and sky.

    Args:
        images: [B, S, 3, H, W] or [V, 3, H, W]
        ground_affine: [B, S, 3, 4] affine for non-sky pixels.
        sky_affine: [B, S, 3, 4] affine for sky pixels (optional).
        sky_mask: [B, S, H, W] or [V, H, W], >0 = non-sky (optional).

    Returns:
        corrected images, same shape as input.
    """
    def _apply_affine_percam(img_hw3, ap_34):
        """Apply per-camera affine: img [H*W, 3], ap [3, 4] -> [H*W, 3]."""
        return (img_hw3 @ ap_34[:3, :3].T) + ap_34[:3, 3]

    def _apply_affine_perpixel(img_hw3, ap_hw34):
        """Apply per-pixel affine: img [H, W, 3], ap [H, W, 3, 4] -> [H, W, 3]."""
        return (ap_hw34[..., :3, :3] @ img_hw3.unsqueeze(-1)).squeeze(-1) + ap_hw34[..., :3, 3]

    is_perpixel = ground_affine.dim() == 6

    if images.dim() == 5:
        B, S, C, H, W = images.shape
        img_bhw3 = images.permute(0, 1, 3, 4, 2).float()
        if is_perpixel:
            out = _apply_affine_perpixel(img_bhw3, ground_affine)
        else:
            out = img_bhw3.reshape(B, S, H * W, 3).clone()
            for b in range(B):
                for s in range(S):
                    out[b, s] = _apply_affine_percam(out[b, s], ground_affine[b, s])
            out = out.reshape(B, S, H, W, 3)
        if sky_affine is not None and sky_mask is not None:
            sm = sky_mask
            if sm.dim() == 4:
                sm = sm[:B, :S]
            sky_bool = (sm <= 0.5).unsqueeze(-1)
            if valid_mask is not None:
                vm = valid_mask
                if vm.dim() == 4:
                    vm = vm[:B, :S]
                sky_bool = sky_bool & vm.unsqueeze(-1).bool()
            is_sky_perpixel = sky_affine.dim() == 6
            if is_sky_perpixel:
                out_sky = _apply_affine_perpixel(img_bhw3, sky_affine)
            else:
                out_sky = img_bhw3.reshape(B, S, H * W, 3).clone()
                for b in range(B):
                    for s in range(S):
                        out_sky[b, s] = _apply_affine_percam(out_sky[b, s], sky_affine[b, s])
                out_sky = out_sky.reshape(B, S, H, W, 3)
            out = torch.where(sky_bool, out_sky, out)
        if valid_mask is not None:
            vm = valid_mask
            if vm.dim() == 4:
                vm = vm[:B, :S]
            pad_bool = ~vm.bool().unsqueeze(-1)
            out = torch.where(pad_bool, img_bhw3, out)
        out = out.clamp(0, 1).permute(0, 1, 4, 2, 3).to(images.dtype)
        return out
    elif images.dim() == 4:
        V, C, H, W = images.shape
        img_vhw3 = images.permute(0, 2, 3, 1).float()
        if is_perpixel:
            ap_g = ground_affine.reshape(-1, H, W, 3, 4)[:V]
            out = _apply_affine_perpixel(img_vhw3, ap_g)
        else:
            ap_g = ground_affine.reshape(-1, 3, 4)[:V]
            out = img_vhw3.reshape(V, H * W, 3).clone()
            for v in range(V):
                out[v] = _apply_affine_percam(out[v], ap_g[v])
            out = out.reshape(V, H, W, 3)
        if sky_affine is not None and sky_mask is not None:
            sm = sky_mask.reshape(-1, H, W)[:V] if sky_mask.dim() >= 3 else sky_mask
            sky_bool = (sm <= 0.5).unsqueeze(-1)
            if valid_mask is not None:
                vm_4d = valid_mask.reshape(-1, H, W)[:V] if valid_mask.dim() >= 3 else valid_mask
                sky_bool = sky_bool & vm_4d.unsqueeze(-1).bool()
            is_sky_perpixel = sky_affine.dim() == 6
            if is_sky_perpixel:
                ap_s = sky_affine.reshape(-1, H, W, 3, 4)[:V]
                out_sky = _apply_affine_perpixel(img_vhw3, ap_s)
            else:
                ap_s = sky_affine.reshape(-1, 3, 4)[:V]
                out_sky = img_vhw3.reshape(V, H * W, 3).clone()
                for v in range(V):
                    out_sky[v] = _apply_affine_percam(out_sky[v], ap_s[v])
                out_sky = out_sky.reshape(V, H, W, 3)
            out = torch.where(sky_bool, out_sky, out)
        if valid_mask is not None:
            vm_4d = valid_mask.reshape(-1, H, W)[:V] if valid_mask.dim() >= 3 else valid_mask
            pad_bool = ~vm_4d.bool().unsqueeze(-1)
            out = torch.where(pad_bool, img_vhw3, out)
        out = out.clamp(0, 1).permute(0, 3, 1, 2).to(images.dtype)
        return out
    else:
        return images

def apply_predicted_bilateral_grids(
    predicted_grids: List[torch.Tensor],
    rgb: torch.Tensor,
    grid_sizes: List[Tuple[int, int, int]],
    guidance_factors: List[int],
) -> torch.Tensor:
    """Apply externally predicted multi-scale bilateral grids to RGB.

    This is a stateless helper that performs slicing / multi-scale affine
    composition with grids predicted by a network.

    Args:
        predicted_grids: list of ``[N, 12, L_i, H_i, W_i]`` tensors, one per
            scale.  ``N`` must equal the batch (view) dimension of ``rgb``.
        rgb: ``[N, 3, H, W]`` rendered image in ``[0, 1]``.
        grid_sizes: list of ``(grid_X, grid_Y, grid_W)`` tuples matching the
            grids (used only for documentation/assertion).
        guidance_factors: per-scale spatial downsample factor for slicing.

    Returns:
        corrected: ``[N, 3, H, W]`` in ``[0, 1]``.
    """
    _rgb2gray = torch.tensor([0.299, 0.587, 0.114], dtype=rgb.dtype, device=rgb.device)
    N, _, H, W = rgb.shape
    mat = torch.eye(4, device=rgb.device, dtype=rgb.dtype).view(1, 1, 1, 4, 4).expand(N, H, W, 4, 4)

    for i, grid_5d in enumerate(predicted_grids):
        factor = guidance_factors[i]
        Hp, Wp = max(1, H // factor), max(1, W // factor)
        rgb_low = F.interpolate(rgb, size=(Hp, Wp), mode="bilinear", align_corners=False)
        rgb_nhwc = rgb_low.permute(0, 2, 3, 1)  # [N, Hp, Wp, 3]
        y = torch.linspace(0.5 / Hp, 1.0 - 0.5 / Hp, Hp, device=rgb.device, dtype=rgb.dtype)
        x = torch.linspace(0.5 / Wp, 1.0 - 0.5 / Wp, Wp, device=rgb.device, dtype=rgb.dtype)
        gy, gx = torch.meshgrid(y, x, indexing="ij")
        grid_xy = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(N, -1, -1, -1)  # [N, Hp, Wp, 2]

        gray = (rgb_nhwc @ _rgb2gray).unsqueeze(-1) * 2.0 - 1.0  # [N, Hp, Wp, 1]
        grid_xy_norm = (grid_xy - 0.5) * 2.0
        grid_xyz = torch.cat([grid_xy_norm, gray], dim=-1)  # [N, Hp, Wp, 3]
        grid_xyz_5d = grid_xyz.unsqueeze(1)  # [N, 1, Hp, Wp, 3]

        sampled = F.grid_sample(
            grid_5d, grid_xyz_5d,
            mode="bilinear", align_corners=True, padding_mode="border",
        )  # [N, 12, 1, Hp, Wp]
        aff = sampled.squeeze(2).permute(0, 2, 3, 1).reshape(N, Hp, Wp, 3, 4)

        # Upsample affine to full resolution
        if Hp != H or Wp != W:
            flat = aff.reshape(N, Hp, Wp, -1).permute(0, 3, 1, 2)
            up = F.interpolate(flat, size=(H, W), mode="bilinear", align_corners=False)
            aff = up.permute(0, 2, 3, 1).reshape(N, H, W, 3, 4)

        hom = torch.zeros(N, H, W, 4, 4, device=rgb.device, dtype=rgb.dtype)
        hom[..., :3, :] = aff
        hom[..., 3, 3] = 1
        mat = hom @ mat

    aff_final = mat[..., :3, :]  # [N, H, W, 3, 4]
    rgb_nhwc_full = rgb.permute(0, 2, 3, 1)  # [N, H, W, 3]
    out = (aff_final[..., :3, :3] @ rgb_nhwc_full[..., None] + aff_final[..., :3, 3:4])[..., 0]
    return out.permute(0, 3, 1, 2).clamp(0, 1)

def _prepare_canonical_input_for_backbone(
    images,
    images_pi3x,
    input_dict_gs,
    output_dict_gs,
    *,
    device,
    is_main_process,
    iteration,
    logger,
    use_canonical_input,
    use_lidar_affine_canonical,
    use_pi3_depth_for_affine,
    gaussian_only_canonical_mem_opt,
    lidar_affine_blend_ref_cameras,
    lidar_affine_exclude_dynamic,
    lidar_affine_min_points,
    lidar_affine_use_rgb_refit,
    lidar_affine_rgb_residual_thresh,
    lidar_affine_rgb_refit_min_points,
    sky_depth_value,
):
    gt_rgb_affine_canonical_early = None

    if (
        use_canonical_input
        and use_lidar_affine_canonical
        and (not use_pi3_depth_for_affine)
    ):
        _n_cur_views = images.shape[1]
        _n_pi3x_views = images_pi3x.shape[1]
        _ci_lidar_depth = input_dict_gs.get("single_depthmaps", None)
        _ci_images_5d = output_dict_gs.get("rgb", None)
        _ci_c2l = input_dict_gs.get("camera2lidar", None)
        _ci_intr = input_dict_gs.get("intrinsics", None)
        if (
            isinstance(_ci_images_5d, torch.Tensor)
            and _ci_images_5d.dim() == 5
            and isinstance(_ci_lidar_depth, torch.Tensor)
            and isinstance(_ci_c2l, torch.Tensor)
            and isinstance(_ci_intr, torch.Tensor)
        ):
            _ci_images_5d = _ci_images_5d.to(device, images.dtype)
            _ci_B, _ci_S = _ci_lidar_depth.shape[:2]
            _ci_imgs = _ci_images_5d[:, -_ci_S:].to(device, images.dtype)
            _ci_intr_c = _ci_intr[:, -_ci_S:].to(device, images.dtype)
            _ci_c2l_c = _ci_c2l[:, -_ci_S:].to(device, images.dtype)
            _ci_depth = _ci_lidar_depth.to(device, images.dtype)
            # Affine correspondences must be solved in the same spatial domain as
            # LiDAR depth/intrinsics (input resolution). `output_dict_gs["rgb"]`
            # may be render-resolution (e.g. 400x224), so fall back to input
            # current-frame images when resolutions differ.
            _ci_imgs_for_affine = _ci_imgs
            if (
                _ci_imgs_for_affine.shape[-2] != _ci_depth.shape[-2]
                or _ci_imgs_for_affine.shape[-1] != _ci_depth.shape[-1]
            ):
                _ci_imgs_for_affine = images[:, -_ci_S:].to(device, images.dtype)

            _ci_dyn = input_dict_gs.get("dynamics_region", None)
            if isinstance(_ci_dyn, torch.Tensor) and _ci_dyn.dim() == 4:
                _ci_dyn = _ci_dyn[:, -_ci_S:].to(device).bool()
            else:
                _ci_dyn = None

            # For canonical-input LiDAR affine, masks must be aligned with input
            # depth resolution. Prefer input_valid_mask (when render/input
            # resolutions differ), then fall back to rendered-view valid_mask.
            _ci_valid = output_dict_gs.get("input_valid_mask", None)
            if _ci_valid is None:
                _ci_valid = output_dict_gs.get("valid_mask", None)
            if isinstance(_ci_valid, torch.Tensor):
                if (
                    _ci_valid.dim() == 4
                    and _ci_valid.shape[0] >= _ci_B
                    and _ci_valid.shape[1] >= _ci_S
                ):
                    _ci_valid = _ci_valid[:_ci_B, -_ci_S:].to(device).bool()
                elif (
                    _ci_valid.dim() == 3
                    and _ci_B == 1
                    and _ci_valid.shape[0] >= _ci_S
                ):
                    _ci_valid = _ci_valid[-_ci_S:].unsqueeze(0).to(device).bool()
                else:
                    _ci_valid = None
            else:
                _ci_valid = None
            if isinstance(_ci_valid, torch.Tensor):
                if (
                    _ci_valid.shape[-2] != _ci_depth.shape[-2]
                    or _ci_valid.shape[-1] != _ci_depth.shape[-1]
                ):
                    # Align render-resolution valid mask to depth/input resolution,
                    # so sky affine never treats side-camera padding as sky.
                    _ci_valid = (
                        F.interpolate(
                            _ci_valid.float(),
                            size=_ci_depth.shape[-2:],
                            mode="nearest",
                        )
                        > 0.5
                    )

            _ci_cam_ids = None
            _ci_vc = output_dict_gs.get("view_cam_ids", None)
            if isinstance(_ci_vc, torch.Tensor) and _ci_vc.numel() >= _ci_B * _ci_S:
                _ci_vc = _ci_vc.to(device, dtype=torch.long)
                if (
                    _ci_vc.dim() == 2
                    and _ci_vc.shape[0] >= _ci_B
                    and _ci_vc.shape[1] >= _ci_S
                ):
                    _ci_cam_ids = _ci_vc[:_ci_B, -_ci_S:]
                elif _ci_vc.dim() == 1 and _ci_vc.numel() >= _ci_B * _ci_S:
                    _ci_cam_ids = _ci_vc[-_ci_B * _ci_S:].view(_ci_B, _ci_S)

            _ci_sky_mask = input_dict_gs.get("sky_mask", None)
            _ci_sky_pts = None
            if isinstance(_ci_sky_mask, torch.Tensor):
                _ci_sky_mask = _ci_sky_mask[:_ci_B, -_ci_S:].to(device)
                _ci_rays_o = input_dict_gs.get("rays_o", None)
                _ci_rays_d = input_dict_gs.get("rays_d", None)
                if isinstance(_ci_rays_o, torch.Tensor) and isinstance(
                    _ci_rays_d, torch.Tensor
                ):
                    _ci_ro = _ci_rays_o[:_ci_B, -_ci_S:].to(device, images.dtype)
                    _ci_rd = _ci_rays_d[:_ci_B, -_ci_S:].to(device, images.dtype)
                    _ci_sky_pts = _ci_ro + _ci_rd * sky_depth_value
            else:
                _ci_sky_mask = None

            with torch.no_grad():
                _ci_affine_out = compute_lidar_affine_blend(
                    _ci_depth,
                    _ci_imgs_for_affine,
                    _ci_intr_c,
                    _ci_c2l_c,
                    ref_cameras=lidar_affine_blend_ref_cameras,
                    cam_ids=_ci_cam_ids,
                    valid_mask=_ci_valid,
                    dynamics_mask=_ci_dyn,
                    exclude_dynamic=lidar_affine_exclude_dynamic,
                    sky_mask=_ci_sky_mask,
                    sky_points=_ci_sky_pts,
                    stride=2,
                    min_points=lidar_affine_min_points,
                    use_rgb_refit=lidar_affine_use_rgb_refit,
                    rgb_residual_thresh=lidar_affine_rgb_residual_thresh,
                    rgb_refit_min_points=lidar_affine_rgb_refit_min_points,
                    return_stats=True,
                )
                if isinstance(_ci_affine_out, tuple) and len(_ci_affine_out) == 3:
                    _ci_ground_aff, _ci_sky_aff, _ = _ci_affine_out
                else:
                    _ci_ground_aff, _ci_sky_aff = _ci_affine_out

            # Keep an input-resolution copy for backbone canonicalization.
            _ci_ground_aff_input = _ci_ground_aff
            _ci_sky_aff_input = _ci_sky_aff
            _ci_canonical_input_5d = None
            _ci_input_images_5d = images[:, -_n_cur_views:].to(device, images.dtype)
            if (
                isinstance(_ci_input_images_5d, torch.Tensor)
                and _ci_input_images_5d.dim() == 5
            ):
                _ci_B_in, _ci_V_in = _ci_input_images_5d.shape[:2]
                _ci_num_rep_in = (_ci_V_in + _ci_S - 1) // _ci_S
                if _ci_ground_aff_input.dim() == 6:
                    _ci_ground_in_exp = _ci_ground_aff_input.repeat(
                        1, _ci_num_rep_in, 1, 1, 1, 1
                    )[:, :_ci_V_in]
                else:
                    _ci_ground_in_exp = _ci_ground_aff_input.repeat(
                        1, _ci_num_rep_in, 1, 1
                    )[:, :_ci_V_in]
                if _ci_sky_aff_input.dim() == 6:
                    _ci_sky_in_exp = _ci_sky_aff_input.repeat(
                        1, _ci_num_rep_in, 1, 1, 1, 1
                    )[:, :_ci_V_in]
                else:
                    _ci_sky_in_exp = _ci_sky_aff_input.repeat(
                        1, _ci_num_rep_in, 1, 1
                    )[:, :_ci_V_in]

                _ci_input_sky_mask_5d = None
                _ci_sm_input = input_dict_gs.get("sky_mask", None)
                if isinstance(_ci_sm_input, torch.Tensor):
                    if _ci_sm_input.dim() == 4:
                        _ci_input_sky_mask_5d = _ci_sm_input[:_ci_B_in, -_ci_V_in:].to(
                            device
                        )
                    elif _ci_sm_input.dim() == 3 and _ci_B_in == 1:
                        _ci_input_sky_mask_5d = _ci_sm_input[-_ci_V_in:].unsqueeze(0).to(
                            device
                        )

                _ci_input_valid_5d = None
                if isinstance(_ci_valid, torch.Tensor):
                    _ci_input_valid_5d = _ci_valid[:_ci_B_in, -_ci_V_in:].to(device)

                _ci_canonical_input_5d = apply_affine_to_images(
                    _ci_input_images_5d,
                    _ci_ground_in_exp,
                    sky_affine=_ci_sky_in_exp,
                    sky_mask=_ci_input_sky_mask_5d,
                    valid_mask=_ci_input_valid_5d,
                )

            _ci_B_img, _ci_V_img = _ci_images_5d.shape[:2]
            _ci_H_img, _ci_W_img = _ci_images_5d.shape[-2:]
            # If affine maps were solved at input resolution but target
            # supervision RGB is render-resolution, resize per-pixel affine maps.
            if _ci_ground_aff.dim() == 6:
                _ci_H_aff, _ci_W_aff = _ci_ground_aff.shape[2], _ci_ground_aff.shape[3]
                if (_ci_H_aff != _ci_H_img) or (_ci_W_aff != _ci_W_img):
                    _ci_g = (
                        _ci_ground_aff.permute(0, 1, 4, 5, 2, 3).reshape(
                            _ci_B, _ci_S * 12, _ci_H_aff, _ci_W_aff
                        )
                    )
                    _ci_g = F.interpolate(
                        _ci_g,
                        size=(_ci_H_img, _ci_W_img),
                        mode="bilinear",
                        align_corners=True,
                    )
                    _ci_ground_aff = (
                        _ci_g.view(_ci_B, _ci_S, 3, 4, _ci_H_img, _ci_W_img)
                        .permute(0, 1, 4, 5, 2, 3)
                        .contiguous()
                    )
            if _ci_sky_aff.dim() == 6:
                _ci_H_aff_s, _ci_W_aff_s = _ci_sky_aff.shape[2], _ci_sky_aff.shape[3]
                if (_ci_H_aff_s != _ci_H_img) or (_ci_W_aff_s != _ci_W_img):
                    _ci_s = (
                        _ci_sky_aff.permute(0, 1, 4, 5, 2, 3).reshape(
                            _ci_B, _ci_S * 12, _ci_H_aff_s, _ci_W_aff_s
                        )
                    )
                    _ci_s = F.interpolate(
                        _ci_s,
                        size=(_ci_H_img, _ci_W_img),
                        mode="bilinear",
                        align_corners=True,
                    )
                    _ci_sky_aff = (
                        _ci_s.view(_ci_B, _ci_S, 3, 4, _ci_H_img, _ci_W_img)
                        .permute(0, 1, 4, 5, 2, 3)
                        .contiguous()
                    )

            _ci_num_rep = (_ci_V_img + _ci_S - 1) // _ci_S
            if _ci_ground_aff.dim() == 6:
                _ci_ground_exp = _ci_ground_aff.repeat(1, _ci_num_rep, 1, 1, 1, 1)[
                    :, :_ci_V_img
                ]
            else:
                _ci_ground_exp = _ci_ground_aff.repeat(1, _ci_num_rep, 1, 1)[:, :_ci_V_img]
            if _ci_sky_aff.dim() == 6:
                _ci_sky_exp = _ci_sky_aff.repeat(1, _ci_num_rep, 1, 1, 1, 1)[:, :_ci_V_img]
            else:
                _ci_sky_exp = _ci_sky_aff.repeat(1, _ci_num_rep, 1, 1)[:, :_ci_V_img]

            _ci_sky_mask_5d = None
            if _ci_sky_mask is not None:
                _ci_sm_all = output_dict_gs.get("sky_mask", None)
                if isinstance(_ci_sm_all, torch.Tensor) and _ci_sm_all.dim() == 4:
                    _ci_sky_mask_5d = _ci_sm_all[:_ci_B_img, :_ci_V_img].to(device)
                elif isinstance(_ci_sm_all, torch.Tensor) and _ci_sm_all.dim() == 3:
                    _ci_sky_mask_5d = _ci_sm_all[:_ci_V_img].unsqueeze(0).to(device)

            _ci_valid_5d = None
            _ci_vm_all = output_dict_gs.get("valid_mask", None)
            if isinstance(_ci_vm_all, torch.Tensor):
                if _ci_vm_all.dim() == 4:
                    _ci_valid_5d = _ci_vm_all[:_ci_B_img, :_ci_V_img].to(device)
                elif _ci_vm_all.dim() == 3:
                    _ci_valid_5d = _ci_vm_all[:_ci_V_img].unsqueeze(0).to(device)
            _ci_canonical_5d = apply_affine_to_images(
                _ci_images_5d,
                _ci_ground_exp,
                sky_affine=_ci_sky_exp,
                sky_mask=_ci_sky_mask_5d,
                valid_mask=_ci_valid_5d,
            )

            # Gaussian-only phase: input & RGB supervision are both canonical; merge the
            # affine result into `output_dict_gs["rgb"]` and drop the extra 5D buffer so we
            # do not hold original RGB + canonical RGB as two full [B,V,3,H,W] tensors.
            _rgb_master = output_dict_gs.get("rgb", None)
            if (
                gaussian_only_canonical_mem_opt
                and isinstance(_rgb_master, torch.Tensor)
                and _rgb_master.shape == _ci_canonical_5d.shape
            ):
                _rgb_master.copy_(_ci_canonical_5d.to(dtype=_rgb_master.dtype))
                del _ci_canonical_5d
                _ci_canonical_5d = _rgb_master

            gt_rgb_affine_canonical_early = _ci_canonical_5d.view(
                _ci_B_img * _ci_V_img, *_ci_canonical_5d.shape[2:]
            ).to(images.dtype)

            images_canonical_5d = (
                _ci_canonical_input_5d.to(images.dtype)
                if isinstance(_ci_canonical_input_5d, torch.Tensor)
                else _ci_canonical_5d.to(images.dtype)
            )
            images = images_canonical_5d[:, -_n_cur_views:]
            images_pi3x = images_canonical_5d[:, -_n_pi3x_views:]

            if is_main_process and iteration == 0:
                _mem_note = (
                    " (merged canonical into output_dict_gs['rgb'], dropped duplicate 5D buffer)"
                    if gaussian_only_canonical_mem_opt
                    and isinstance(_rgb_master, torch.Tensor)
                    and _rgb_master is _ci_canonical_5d
                    else ""
                )
                logger.info(
                    f"[CanonicalInput] Enabled: images replaced with LiDAR affine canonical GT. "
                    f"Shape: {tuple(images.shape)}, ref_cameras={list(lidar_affine_blend_ref_cameras)}{_mem_note}"
                )
    elif (
        use_canonical_input
        and use_lidar_affine_canonical
        and use_pi3_depth_for_affine
        and is_main_process
        and iteration == 0
    ):
        logger.info(
            "[CanonicalInput] Skip pre-backbone canonical replacement because "
            "lidar_affine_depth_source=pi3 needs post-backbone depth. "
            "Pi3 depth will be used for loss-side affine correspondences."
        )

    return images, images_pi3x, gt_rgb_affine_canonical_early
