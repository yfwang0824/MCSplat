import math
import struct
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def _quat_scale_to_covar_preci(
    quats: Tensor,  # [N, 4],
    scales: Tensor,  # [N, 3],
    compute_covar: bool = True,
    compute_preci: bool = True,
    triu: bool = False,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """PyTorch implementation of `gsplat.cuda._wrapper.quat_scale_to_covar_preci()`."""
    quats = F.normalize(quats, p=2, dim=-1)
    w, x, y, z = torch.unbind(quats, dim=-1)
    R = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )

    R = R.reshape(quats.shape[:-1] + (3, 3))  # (..., 3, 3)
    # R.register_hook(lambda grad: print("grad R", grad))

    if compute_covar:
        M = R * scales[..., None, :]  # (..., 3, 3)
        covars = torch.bmm(M, M.transpose(-1, -2))  # (..., 3, 3)
        if triu:
            covars = covars.reshape(covars.shape[:-2] + (9,))  # (..., 9)
            covars = (covars[..., [0, 1, 2, 4, 5, 8]] + covars[..., [0, 3, 6, 4, 7, 8]]) / 2.0  # (..., 6)
    if compute_preci:
        P = R * (1 / scales[..., None, :])  # (..., 3, 3)
        precis = torch.bmm(P, P.transpose(-1, -2))  # (..., 3, 3)
        if triu:
            precis = precis.reshape(precis.shape[:-2] + (9,))
            precis = (precis[..., [0, 1, 2, 4, 5, 8]] + precis[..., [0, 3, 6, 4, 7, 8]]) / 2.0

    return covars if compute_covar else None, precis if compute_preci else None


def _persp_proj(
    means: Tensor,  # [C, N, 3]
    covars: Tensor,  # [C, N, 3, 3]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
) -> Tuple[Tensor, Tensor]:
    """PyTorch implementation of prespective projection for 3D Gaussians.

    Args:
        means: Gaussian means in camera coordinate system. [C, N, 3].
        covars: Gaussian covariances in camera coordinate system. [C, N, 3, 3].
        Ks: Camera intrinsics. [C, 3, 3].
        width: Image width.
        height: Image height.

    Returns:
        A tuple:

        - **means2d**: Projected means. [C, N, 2].
        - **cov2d**: Projected covariances. [C, N, 2, 2].
    """
    C, N, _ = means.shape

    tx, ty, tz = torch.unbind(means, dim=-1)  # [C, N]
    tz2 = tz**2  # [C, N]

    fx = Ks[..., 0, 0, None]  # [C, 1]
    fy = Ks[..., 1, 1, None]  # [C, 1]
    cx = Ks[..., 0, 2, None]  # [C, 1]
    cy = Ks[..., 1, 2, None]  # [C, 1]
    tan_fovx = 0.5 * width / fx  # [C, 1]
    tan_fovy = 0.5 * height / fy  # [C, 1]

    lim_x_pos = (width - cx) / fx + 0.3 * tan_fovx
    lim_x_neg = cx / fx + 0.3 * tan_fovx
    lim_y_pos = (height - cy) / fy + 0.3 * tan_fovy
    lim_y_neg = cy / fy + 0.3 * tan_fovy
    tx = tz * torch.clamp(tx / tz, min=-lim_x_neg, max=lim_x_pos)
    ty = tz * torch.clamp(ty / tz, min=-lim_y_neg, max=lim_y_pos)

    O = torch.zeros((C, N), device=means.device, dtype=means.dtype)
    J = torch.stack([fx / tz, O, -fx * tx / tz2, O, fy / tz, -fy * ty / tz2], dim=-1).reshape(C, N, 2, 3)

    cov2d = torch.einsum("...ij,...jk,...kl->...il", J, covars, J.transpose(-1, -2))
    means2d = torch.einsum("cij,cnj->cni", Ks[:, :2, :3], means)  # [C, N, 2]
    means2d = means2d / tz[..., None]  # [C, N, 2]
    return means2d, cov2d  # [C, N, 2], [C, N, 2, 2]


def _depth_compensation_from_cov3d(cov3d: Tensor, eps2d: float) -> Tensor:
    # cov3d [[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]]
    # depth_comp is [invcov3d[2,0]/invcov3d[2,2], invcov3d[2,1]/invcov3d[2,2]]
    # we get invcov3d by dividing adjugate of cov3d by det(cov3d)
    a1 = cov3d[..., 0, 0] + eps2d
    a2 = cov3d[..., 0, 1]
    b1 = cov3d[..., 1, 0]
    b2 = cov3d[..., 1, 1] + eps2d
    c1 = cov3d[..., 2, 0]
    c2 = cov3d[..., 2, 1]

    invD = 1 / (a1 * b2 - a2 * b1)

    return torch.stack([(b1 * c2 - b2 * c1) * invD, (a2 * c1 - a1 * c2) * invD], dim=-1)


def _lidar_proj(
    means: Tensor,  # [C, N, 3]
    covars: Tensor,  # [C, N, 3, 3]
    eps2d: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """PyTorch implementation of lidar projection for 3D Gaussians.

    Args:
        means: Gaussian means in camera coordinate system. [C, N, 3].
        covars: Gaussian covariances in camera coordinate system. [C, N, 3, 3].

    Returns:
        A tuple:

        - **means2d**: Projected means. [C, N, 2].
        - **cov2d**: Projected covariances. [C, N, 2, 2].
        - **r**: Range. [C, N]
        - **depth_comp**: Depth compensation. [C, N]
    """
    C, N, _ = means.shape

    tx, ty, tz = torch.unbind(means, dim=-1)  # [C, N]

    r = torch.linalg.vector_norm(means, dim=-1)  # [C, N]
    rinv = torch.rsqrt(tx**2 + ty**2 + tz**2)  # [C, N]
    norm_means = means * rinv[..., None]
    J = torch.stack(
        [
            torch.rad2deg(-ty / (tx**2 + ty**2)),
            torch.rad2deg(tx / (tx**2 + ty**2)),
            torch.rad2deg(torch.zeros_like(tx)),
            torch.rad2deg(-tx * tz * torch.rsqrt(tx**2 + ty**2) / (r**2)),
            torch.rad2deg(-ty * tz * torch.rsqrt(tx**2 + ty**2) / (r**2)),
            torch.rad2deg(torch.sqrt(tx**2 + ty**2) / (r**2)),
            norm_means[..., 0],
            norm_means[..., 1],
            norm_means[..., 2],
        ],
        dim=-1,
    ).reshape(C, N, 3, 3)  # [C, N, 3, 3]

    cov3d_spherical = torch.einsum("...ij,...jk,...kl->...il", J, covars, J.transpose(-1, -2))
    depth_comp = _depth_compensation_from_cov3d(cov3d_spherical, eps2d)
    cov2d = cov3d_spherical[..., :2, :2]  # [C, N, 2, 2]
    means2d = torch.rad2deg(torch.stack([torch.atan2(ty, tx), torch.asin(norm_means[..., 2])], dim=-1))  # [C, N, 2]
    return means2d, cov2d, r, depth_comp  # [C, N, 2], [C, N, 2, 2], [C, N], [C, N, 2]


def _world_to_cam(
    means: Tensor,  # [N, 3]
    covars: Tensor,  # [N, 3, 3]
    viewmats: Tensor,  # [C, 4, 4]
) -> Tuple[Tensor, Tensor]:
    """PyTorch implementation of world to camera transformation on Gaussians.

    Args:
        means: Gaussian means in world coordinate system. [C, N, 3].
        covars: Gaussian covariances in world coordinate system. [C, N, 3, 3].
        viewmats: world to camera transformation matrices. [C, 4, 4].

    Returns:
        A tuple:

        - **means_c**: Gaussian means in camera coordinate system. [C, N, 3].
        - **covars_c**: Gaussian covariances in camera coordinate system. [C, N, 3, 3].
    """
    R = viewmats[:, :3, :3]  # [C, 3, 3]
    t = viewmats[:, :3, 3]  # [C, 3]
    means_c = torch.einsum("cij,nj->cni", R, means) + t[:, None, :]  # (C, N, 3)
    covars_c = torch.einsum("cij,njk,clk->cnil", R, covars, R)  # [C, N, 3, 3]
    return means_c, covars_c


def _compute_pix_velocity(
    p_view: Tensor, lin_vel: Tensor, ang_vel: Tensor, velocities: Tensor, Ks: Tensor, width: int, height: int
) -> Tensor:
    C, N = p_view.shape[:2]

    tx, ty, tz = torch.unbind(p_view, dim=-1)
    tz2 = tz**2  # [C, N]

    lin_vel = lin_vel.unsqueeze(1)
    ang_vel = ang_vel.unsqueeze(1)

    fx = Ks[..., 0, 0, None]
    fy = Ks[..., 1, 1, None]
    cx = Ks[..., 0, 2, None]
    cy = Ks[..., 1, 2, None]
    tan_fovx = 0.5 * width / fx  # [C, 1]
    tan_fovy = 0.5 * height / fy  # [C, 1]

    lim_x_pos = (width - cx) / fx + 0.3 * tan_fovx
    lim_x_neg = cx / fx + 0.3 * tan_fovx
    lim_y_pos = (height - cy) / fy + 0.3 * tan_fovy
    lim_y_neg = cy / fy + 0.3 * tan_fovy
    tx = tz * torch.clamp(tx / tz, min=-lim_x_neg, max=lim_x_pos)
    ty = tz * torch.clamp(ty / tz, min=-lim_y_neg, max=lim_y_pos)

    Os = torch.zeros((C, N), device=p_view.device, dtype=p_view.dtype)
    J = torch.stack([fx / tz, Os, -fx * tx / tz2, Os, fy / tz, -fy * ty / tz2], dim=-1).reshape(C, N, 2, 3)

    rot_part = torch.cross(ang_vel, p_view, dim=-1)
    total_vel = lin_vel + rot_part - velocities

    return (-J @ total_vel.unsqueeze(-1)).squeeze(-1).to(torch.float32)


def _compute_lidar_velocity(p_view: Tensor, lin_vel: Tensor, ang_vel: Tensor, velocities: Tensor) -> Tensor:
    C, N = p_view.shape[:2]
    tx, ty, tz = torch.unbind(p_view, dim=-1)  # [C, N]

    lin_vel = lin_vel.unsqueeze(1)
    ang_vel = ang_vel.unsqueeze(1)

    rot_part = torch.cross(ang_vel, p_view, dim=-1)
    total_vel = lin_vel + rot_part - velocities

    r = torch.linalg.vector_norm(p_view, dim=-1)  # [C, N]
    rinv = torch.rsqrt(tx**2 + ty**2 + tz**2)  # [C, N]
    norm_means = p_view * rinv[..., None]
    J_angles = torch.stack(
        [
            torch.rad2deg(-ty / (tx**2 + ty**2)),
            torch.rad2deg(tx / (tx**2 + ty**2)),
            torch.rad2deg(torch.zeros_like(tx)),
            torch.rad2deg(-tx * tz * torch.rsqrt(tx**2 + ty**2) / (r**2)),
            torch.rad2deg(-ty * tz * torch.rsqrt(tx**2 + ty**2) / (r**2)),
            torch.rad2deg(torch.sqrt(tx**2 + ty**2) / (r**2)),
        ],
        dim=-1,
    ).reshape(C, N, 2, 3)  # [C, N, 2, 3]
    J_range = torch.stack(
        [
            norm_means[..., 0],
            norm_means[..., 1],
            norm_means[..., 2],
        ],
        dim=-1,
    ).reshape(C, N, 1, 3)

    J = torch.cat([J_angles, J_range], dim=-2)

    return (-J @ total_vel.unsqueeze(-1)).squeeze(-1)


def _fully_fused_projection(
    means: Tensor,  # [N, 3]
    covars: Tensor,  # [N, 3, 3]
    velocities: Tensor,  # [N, 3]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    linear_velocity: Tensor,  # [C, 3]
    angular_velocity: Tensor,  # [C, 3]
    rolling_shutter_time: Tensor,  # [C]
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    calc_compensations: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor], Tensor]:
    """PyTorch implementation of `gsplat.cuda._wrapper.fully_fused_projection()`

    .. note::

        This is a minimal implementation of fully fused version, which has more
        arguments. Not all arguments are supported.
    """
    means_c, covars_c = _world_to_cam(means, covars, viewmats)
    if velocities is None:
        vel_c = torch.zeros((means_c.shape[0], means_c.shape[1], 3), device=means.device, dtype=means.dtype)
    else:
        vel_c = torch.einsum("cij,nj->cni", viewmats[:, :3, :3], velocities)  # (C, N, 3)
    means2d, covars2d = _persp_proj(means_c, covars_c, Ks, width, height)
    det_orig = covars2d[..., 0, 0] * covars2d[..., 1, 1] - covars2d[..., 0, 1] * covars2d[..., 1, 0]
    covars2d = covars2d + torch.eye(2, device=means.device, dtype=means.dtype) * eps2d

    det = covars2d[..., 0, 0] * covars2d[..., 1, 1] - covars2d[..., 0, 1] * covars2d[..., 1, 0]
    det = det.clamp(min=1e-10)

    if calc_compensations:
        compensations = torch.sqrt(torch.clamp(det_orig / det, min=0.0))
    else:
        compensations = None

    conics = torch.stack(
        [
            covars2d[..., 1, 1] / det,
            -(covars2d[..., 0, 1] + covars2d[..., 1, 0]) / 2.0 / det,
            covars2d[..., 0, 0] / det,
        ],
        dim=-1,
    )  # [C, N, 3]

    depths = means_c[..., 2]  # [C, N]

    radius = 3 * torch.sqrt(torch.stack([covars2d[..., 0, 0], covars2d[..., 1, 1]], dim=-1))
    # v2 = b - torch.sqrt(torch.clamp(b**2 - det, min=0.01))  # (...,)
    # radius = torch.ceil(3.0 * torch.sqrt(torch.max(v1, v2)))  # (...,)

    pix_vels = _compute_pix_velocity(means_c, linear_velocity, angular_velocity, vel_c, Ks, width, height)
    radius += pix_vels.abs() * 0.5 * rolling_shutter_time.unsqueeze(-1).unsqueeze(-1)

    valid_rolling_shutter_time_mask = (rolling_shutter_time > 0).reshape(-1, 1, 1) * torch.ones_like(pix_vels)
    pix_vels *= valid_rolling_shutter_time_mask

    valid = (det > 0) & (depths > near_plane) & (depths < far_plane)
    radius[~valid] = 0.0

    inside = (
        (means2d[..., 0] + radius[..., 0] > 0)
        & (means2d[..., 0] - radius[..., 0] < width)
        & (means2d[..., 1] + radius[..., 1] > 0)
        & (means2d[..., 1] - radius[..., 1] < height)
    )
    radius[~inside] = 0.0

    radii = radius.int()
    return radii, means2d, depths, conics, compensations, pix_vels


def _fully_fused_lidar_projection(
    means: Tensor,  # [N, 3]
    covars: Tensor,  # [N, 3, 3]
    velocities: Tensor,  # [N, 3]
    viewmats: Tensor,  # [C, 4, 4]
    linear_velocity: Tensor,  # [C, 3]
    angular_velocity: Tensor,  # [C, 3]
    rolling_shutter_time: Tensor,  # [C]
    min_elevation: float = -45,
    max_elevation: float = 45,
    min_azimuth: float = -180,
    max_azimuth: float = 180,
    eps2d: float = 0.01,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    calc_compensations: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor], Tensor, Tensor]:
    """PyTorch implementation of `gsplat.cuda._wrapper.fully_fused_projection()`

    .. note::

        This is a minimal implementation of fully fused version, which has more
        arguments. Not all arguments are supported.
    """
    means_c, covars_c = _world_to_cam(means, covars, viewmats)
    if velocities is None:
        vel_c = torch.zeros((means_c.shape[0], means_c.shape[1], 3), device=means.device, dtype=means.dtype)
    else:
        vel_c = torch.einsum("cij,nj->cni", viewmats[:, :3, :3], velocities)  # (C, N, 3)
    means2d, covars2d, distances, depth_comp = _lidar_proj(means_c, covars_c, eps2d)
    det_orig = covars2d[..., 0, 0] * covars2d[..., 1, 1] - covars2d[..., 0, 1] * covars2d[..., 1, 0]
    covars2d = covars2d + torch.eye(2, device=means.device, dtype=means.dtype) * eps2d

    det = covars2d[..., 0, 0] * covars2d[..., 1, 1] - covars2d[..., 0, 1] * covars2d[..., 1, 0]
    det = det.clamp(min=1e-10)

    if calc_compensations:
        compensations = torch.sqrt(torch.clamp(det_orig / det, min=0.0))
    else:
        compensations = None

    conics = torch.stack(
        [
            covars2d[..., 1, 1] / det,
            -(covars2d[..., 0, 1] + covars2d[..., 1, 0]) / 2.0 / det,
            covars2d[..., 0, 0] / det,
        ],
        dim=-1,
    )  # [C, N, 3]

    depths = distances  # [C, N]

    # b = (covars2d[..., 0, 0] + covars2d[..., 1, 1]) / 2  # (...,)
    # v1 = b + torch.sqrt(torch.clamp(b**2 - det, min=1e-6))  # (...,)
    # radius = 3.0 * torch.sqrt(v1)  # (...,)
    # v2 = b - torch.sqrt(torch.clamp(b**2 - det, min=0.01))  # (...,)
    # radius = torch.ceil(3.0 * torch.sqrt(torch.max(v1, v2)))  # (...,)
    radius = 3 * torch.sqrt(torch.stack([covars2d[..., 0, 0], covars2d[..., 1, 1]], dim=-1))

    pix_vels = _compute_lidar_velocity(means_c, linear_velocity, angular_velocity, vel_c)
    radius += pix_vels[..., :2].abs() * 0.5 * rolling_shutter_time.unsqueeze(-1).unsqueeze(-1)

    valid_rolling_shutter_time_mask = (rolling_shutter_time > 0).reshape(-1, 1, 1) * torch.ones_like(pix_vels)
    pix_vels *= valid_rolling_shutter_time_mask

    valid = (det > 0) & (depths > near_plane) & (depths < far_plane)
    radius[~valid] = 0.0

    outside = (
        (means2d[..., 1] + radius[..., 1] <= min_elevation)
        | (means2d[..., 1] - radius[..., 1] >= max_elevation)
        | (means2d[..., 0] + radius[..., 0] <= min_azimuth)
        | (means2d[..., 0] - radius[..., 0] >= max_azimuth)
    )
    radius[outside] = 0.0

    radii = radius
    return radii, means2d, depths, conics, compensations, pix_vels, depth_comp


@torch.no_grad()
def _isect_tiles(
    means2d: Tensor,
    radii: Tensor,
    depths: Tensor,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    sort: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Pytorch implementation of `gsplat.cuda._wrapper.isect_tiles()`.

    .. note::

        This is a minimal implementation of the fully fused version, which has more
        arguments. Not all arguments are supported.
    """
    C, N = means2d.shape[:2]
    device = means2d.device

    # compute tiles_per_gauss
    tile_means2d = means2d / tile_size
    tile_radii = radii / tile_size
    tile_mins = torch.floor(tile_means2d - tile_radii).int()
    tile_maxs = torch.ceil(tile_means2d + tile_radii).int()
    tile_mins[..., 0] = torch.clamp(tile_mins[..., 0], 0, tile_width)
    tile_mins[..., 1] = torch.clamp(tile_mins[..., 1], 0, tile_height)
    tile_maxs[..., 0] = torch.clamp(tile_maxs[..., 0], 0, tile_width)
    tile_maxs[..., 1] = torch.clamp(tile_maxs[..., 1], 0, tile_height)
    tiles_per_gauss = (tile_maxs - tile_mins).prod(dim=-1)  # [C, N]
    tiles_per_gauss *= radii[..., 0] > 0.0

    n_isects = tiles_per_gauss.sum().item()
    isect_ids = torch.empty(n_isects, dtype=torch.int64, device=device)
    flatten_ids = torch.empty(n_isects, dtype=torch.int32, device=device)

    cum_tiles_per_gauss = torch.cumsum(tiles_per_gauss.flatten(), dim=0)
    tile_n_bits = (tile_width * tile_height).bit_length()

    def binary(num):
        return "".join("{:0>8b}".format(c) for c in struct.pack("!f", num))

    def kernel(cam_id, gauss_id):
        if radii[cam_id, gauss_id, 0] <= 0.0:
            return
        index = cam_id * N + gauss_id
        curr_idx = cum_tiles_per_gauss[index - 1] if index > 0 else 0

        depth_id = struct.unpack("i", struct.pack("f", depths[cam_id, gauss_id]))[0]

        tile_min = tile_mins[cam_id, gauss_id]
        tile_max = tile_maxs[cam_id, gauss_id]
        for y in range(tile_min[1], tile_max[1]):
            for x in range(tile_min[0], tile_max[0]):
                tile_id = y * tile_width + x
                isect_ids[curr_idx] = (cam_id << 32 << tile_n_bits) | (tile_id << 32) | depth_id
                flatten_ids[curr_idx] = index  # flattened index
                curr_idx += 1

    for cam_id in range(C):
        for gauss_id in range(N):
            kernel(cam_id, gauss_id)

    if sort:
        isect_ids, sort_indices = torch.sort(isect_ids)
        flatten_ids = flatten_ids[sort_indices]

    return tiles_per_gauss.int(), isect_ids, flatten_ids


@torch.no_grad()
def _isect_lidar_tiles(
    means2d: Tensor,
    radii: Tensor,
    depths: Tensor,
    elev_boundaries: Tensor,
    tile_azim_resolution: float,
    min_azimuth: float,
    sort: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Pytorch implementation of `gsplat.cuda._wrapper.isect_tiles()`.

    .. note::

        This is a minimal implementation of the fully fused version, which has more
        arguments. Not all arguments are supported.
    """
    C, N = means2d.shape[:2]
    device = means2d.device

    # compute tiles_per_gauss
    n_tiles_azim = math.ceil(360 / tile_azim_resolution)
    azim = means2d[..., 0] - min_azimuth  # move to 0 to 360
    azim_lower = azim - radii[..., 0]
    azim_upper = azim + radii[..., 0]
    tile_mins_azim = torch.floor(
        torch.where(
            azim_lower >= 0,
            azim_lower / tile_azim_resolution,
            (azim_lower % 360 - n_tiles_azim * tile_azim_resolution) / tile_azim_resolution,
        )
    ).int()
    tile_maxs_azim = torch.ceil(
        torch.where(
            azim_upper <= 360,
            azim_upper / tile_azim_resolution,
            n_tiles_azim + (azim_upper % 360) / tile_azim_resolution,
        )
    ).int()

    tile_mins_elev = (torch.searchsorted(elev_boundaries, means2d[..., 1] - radii[..., 1], side="right") - 1).clamp_min(
        0
    )
    tile_maxs_elev = torch.searchsorted(elev_boundaries, means2d[..., 1] + radii[..., 1], side="left").clamp_max(
        len(elev_boundaries) - 1
    )

    tile_mins = torch.stack([tile_mins_azim, tile_mins_elev], dim=-1)
    tile_maxs = torch.stack([tile_maxs_azim, tile_maxs_elev], dim=-1)

    tiles_per_gauss = (tile_maxs - tile_mins).prod(dim=-1)  # [C, N]
    tiles_per_gauss *= radii[..., 0] > 0.0

    n_isects = tiles_per_gauss.sum().item()
    isect_ids = torch.empty(n_isects, dtype=torch.int64, device=device)
    flatten_ids = torch.empty(n_isects, dtype=torch.int32, device=device)

    cum_tiles_per_gauss = torch.cumsum(tiles_per_gauss.flatten(), dim=0)
    n_tiles_azim = math.ceil(360 / tile_azim_resolution)
    n_tiles_elev = len(elev_boundaries) - 1
    tile_n_bits = (n_tiles_azim * n_tiles_elev).bit_length()

    def kernel(cam_id, gauss_id):
        if radii[cam_id, gauss_id, 0] <= 0.0:
            return
        index = cam_id * N + gauss_id
        curr_idx = cum_tiles_per_gauss[index - 1] if index > 0 else 0

        depth_id = struct.unpack("i", struct.pack("f", depths[cam_id, gauss_id]))[0]

        tile_min = tile_mins[cam_id, gauss_id]
        tile_max = tile_maxs[cam_id, gauss_id]
        for y in range(tile_min[1], tile_max[1]):  # elevation
            for x in range(tile_min[0], tile_max[0]):  # azimuth
                # wrap x to 0 to n_tiles_azim - 1
                x = x % n_tiles_azim
                tile_id = y * n_tiles_azim + x

                isect_ids[curr_idx] = (cam_id << 32 << tile_n_bits) | (tile_id << 32) | depth_id
                flatten_ids[curr_idx] = index  # flattened index
                curr_idx += 1

    for cam_id in range(C):
        for gauss_id in range(N):
            kernel(cam_id, gauss_id)

    if sort:
        isect_ids, sort_indices = torch.sort(isect_ids)
        flatten_ids = flatten_ids[sort_indices]

    return tiles_per_gauss.int(), isect_ids, flatten_ids


@torch.no_grad()
def _isect_offset_encode(isect_ids: Tensor, C: int, tile_width: int, tile_height: int) -> Tensor:
    """Pytorch implementation of `gsplat.cuda._wrapper.isect_offset_encode()`.

    .. note::

        This is a minimal implementation of the fully fused version, which has more
        arguments. Not all arguments are supported.
    """
    tile_n_bits = (tile_width * tile_height).bit_length()
    tile_counts = torch.zeros((C, tile_height, tile_width), dtype=torch.int64, device=isect_ids.device)

    isect_ids_uq, counts = torch.unique_consecutive(isect_ids >> 32, return_counts=True)

    cam_ids_uq = isect_ids_uq >> tile_n_bits
    tile_ids_uq = isect_ids_uq & ((1 << tile_n_bits) - 1)
    tile_ids_x_uq = tile_ids_uq % tile_width
    tile_ids_y_uq = tile_ids_uq // tile_width

    tile_counts[cam_ids_uq, tile_ids_y_uq, tile_ids_x_uq] = counts

    cum_tile_counts = torch.cumsum(tile_counts.flatten(), dim=0).reshape_as(tile_counts)
    offsets = cum_tile_counts - tile_counts
    return offsets.int()


def accumulate(
    means2d: Tensor,  # [C, N, 2]
    conics: Tensor,  # [C, N, 3]
    opacities: Tensor,  # [C, N]
    colors: Tensor,  # [C, N, channels]
    pix_vels: Tensor,  # [C, N, 2]
    rolling_shutter_time: Tensor,  # [C]
    gaussian_ids: Tensor,  # [M]
    pixel_ids: Tensor,  # [M]
    camera_ids: Tensor,  # [M]
    image_width: int,
    image_height: int,
) -> Tuple[Tensor, Tensor]:
    """Alpha compositing of 2D Gaussians in Pure Pytorch.

    This function performs alpha compositing for Gaussians based on the pair of indices
    {gaussian_ids, pixel_ids, camera_ids}, which annotates the intersection between all
    pixels and Gaussians. These intersections can be accquired from
    `gsplat.rasterize_to_indices_in_range`.

    .. note::

        This function exposes the alpha compositing process into pure Pytorch.
        So it relies on Pytorch's autograd for the backpropagation. It is much slower
        than our fully fused rasterization implementation and comsumes much more GPU memory.
        But it could serve as a playground for new ideas or debugging, as no backward
        implementation is needed.

    .. warning::

        This function requires the `nerfacc` package to be installed. Please install it
        using the following command `pip install nerfacc`.

    Args:
        means2d: Gaussian means in 2D. [C, N, 2]
        conics: Inverse of the 2D Gaussian covariance, Only upper triangle values. [C, N, 3]
        opacities: Per-view Gaussian opacities (for example, when antialiasing is
            enabled, Gaussian in each view would efficiently have different opacity). [C, N]
        colors: Per-view Gaussian colors. Supports N-D features. [C, N, channels]
        gaussian_ids: Collection of Gaussian indices to be rasterized. A flattened list of shape [M].
        pixel_ids: Collection of pixel indices (row-major) to be rasterized. A flattened list of shape [M].
        camera_ids: Collection of camera indices to be rasterized. A flattened list of shape [M].
        image_width: Image width.
        image_height: Image height.

    Returns:
        A tuple:

        - **renders**: Accumulated colors. [C, image_height, image_width, channels]
        - **alphas**: Accumulated opacities. [C, image_height, image_width, 1]
    """

    try:
        from nerfacc import accumulate_along_rays, render_weight_from_alpha
    except ImportError:
        raise ImportError("Please install nerfacc package: pip install nerfacc")

    C, N = means2d.shape[:2]
    channels = colors.shape[-1]

    pixel_ids_x = pixel_ids % image_width
    pixel_ids_y = pixel_ids // image_width
    rolling_shutter_times = rolling_shutter_time[camera_ids] * (pixel_ids_y / (image_height - 1) - 0.5)
    pixel_coords = torch.stack([pixel_ids_x, pixel_ids_y], dim=-1) + 0.5  # [M, 2]
    deltas = (
        means2d[camera_ids, gaussian_ids]
        + rolling_shutter_times.unsqueeze(-1) * pix_vels[camera_ids, gaussian_ids]
        - pixel_coords
    )  # [M, 2]
    c = conics[camera_ids, gaussian_ids]  # [M, 3]
    sigmas = (
        0.5 * (c[:, 0] * deltas[:, 0] ** 2 + c[:, 2] * deltas[:, 1] ** 2) + c[:, 1] * deltas[:, 0] * deltas[:, 1]
    )  # [M]
    alphas = torch.clamp_max(opacities[camera_ids, gaussian_ids] * torch.exp(-sigmas), 0.999)

    indices = camera_ids * image_height * image_width + pixel_ids
    total_pixels = C * image_height * image_width

    weights, trans = render_weight_from_alpha(alphas, ray_indices=indices, n_rays=total_pixels)
    renders = accumulate_along_rays(
        weights,
        colors[camera_ids, gaussian_ids],
        ray_indices=indices,
        n_rays=total_pixels,
    ).reshape(C, image_height, image_width, channels)
    alphas = accumulate_along_rays(weights, None, ray_indices=indices, n_rays=total_pixels).reshape(
        C, image_height, image_width, 1
    )

    return renders, alphas


def accumulate_lidar(
    means2d: Tensor,  # [C, N, 2]
    conics: Tensor,  # [C, N, 3]
    opacities: Tensor,  # [C, N]
    lidar_features: Tensor,  # [C, N, channels]
    pix_vels: Tensor,  # [C, N, 3]
    depth_compensations: Tensor,  # [C, N, 2]
    raster_pts: Tensor,  # [C, image_height, image_width, 3] (azimuth, elevation, range)
    gaussian_ids: Tensor,  # [M]
    pixel_ids: Tensor,  # [M]
    camera_ids: Tensor,  # [M]
    image_width: int,
    image_height: int,
    compute_alpha_sum_until_pts: bool,
    compute_alpha_sum_until_pts_threshold: float,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Alpha compositing of 2D Gaussians in Pure Pytorch.

    This function performs alpha compositing for Gaussians based on the pair of indices
    {gaussian_ids, pixel_ids, camera_ids}, which annotates the intersection between all
    pixels and Gaussians. These intersections can be accquired from
    `gsplat.rasterize_to_indices_in_range`.

    .. note::

        This function exposes the alpha compositing process into pure Pytorch.
        So it relies on Pytorch's autograd for the backpropagation. It is much slower
        than our fully fused rasterization implementation and comsumes much more GPU memory.
        But it could serve as a playground for new ideas or debugging, as no backward
        implementation is needed.

    .. warning::

        This function requires the `nerfacc` package to be installed. Please install it
        using the following command `pip install nerfacc`.

    Args:
        means2d: Gaussian means in 2D. [C, N, 2]
        conics: Inverse of the 2D Gaussian covariance, Only upper triangle values. [C, N, 3]
        opacities: Per-view Gaussian opacities (for example, when antialiasing is
            enabled, Gaussian in each view would efficiently have different opacity). [C, N]
        colors: Per-view Gaussian colors. Supports N-D features. [C, N, channels]
        gaussian_ids: Collection of Gaussian indices to be rasterized. A flattened list of shape [M].
        pixel_ids: Collection of pixel indices (row-major) to be rasterized. A flattened list of shape [M].
        camera_ids: Collection of camera indices to be rasterized. A flattened list of shape [M].
        image_width: Image width.
        image_height: Image height.

    Returns:
        A tuple:

        - **renders**: Accumulated colors. [C, image_height, image_width, channels]
        - **alphas**: Accumulated opacities. [C, image_height, image_width, 1]
    """

    try:
        from nerfacc import accumulate_along_rays, render_weight_from_alpha
    except ImportError:
        raise ImportError("Please install nerfacc package: pip install nerfacc")

    C, N = means2d.shape[:2]
    channels = lidar_features.shape[-1]

    pixel_ids_x = pixel_ids % image_width
    pixel_ids_y = pixel_ids // image_width
    pixel_coords = raster_pts[camera_ids, pixel_ids_y, pixel_ids_x][..., :2]
    rolling_shutter_times = raster_pts[camera_ids, pixel_ids_y, pixel_ids_x][..., -1]
    deltas = (
        (
            (
                means2d[camera_ids, gaussian_ids]
                + rolling_shutter_times.unsqueeze(-1) * pix_vels[camera_ids, gaussian_ids, ..., :2]
            )
            - pixel_coords
            + 180
        )
        % 360
    ) - 180  # [M, 2]
    deltas = torch.where(deltas < -180, deltas + 360, deltas)
    c = conics[camera_ids, gaussian_ids]  # [M, 3]
    sigmas = (
        0.5 * (c[:, 0] * deltas[:, 0] ** 2 + c[:, 2] * deltas[:, 1] ** 2) + c[:, 1] * deltas[:, 0] * deltas[:, 1]
    )  # [M]
    alphas = torch.clamp_max(opacities[camera_ids, gaussian_ids] * torch.exp(-sigmas), 0.999)

    gs_ranges = (
        lidar_features[camera_ids, gaussian_ids, -1]
        + torch.einsum("bi,bi->b", deltas, depth_compensations[camera_ids, gaussian_ids])
        + pix_vels[camera_ids, gaussian_ids, ..., 2] * rolling_shutter_times
    )
    render_features = torch.cat([lidar_features[camera_ids, gaussian_ids, :-1], gs_ranges.unsqueeze(-1)], dim=-1)

    indices = camera_ids * image_height * image_width + pixel_ids
    total_pixels = C * image_height * image_width

    if compute_alpha_sum_until_pts:
        pts_ranges = raster_pts[camera_ids, pixel_ids_y, pixel_ids_x][..., 2]
        gs_before_point = lidar_features[camera_ids, gaussian_ids, -1] < (
            pts_ranges - compute_alpha_sum_until_pts_threshold
        )

        alphas_until_points = alphas.clone()
        alphas_until_points[~gs_before_point] = 0.0
        alpha_sum_until_pts = accumulate_along_rays(
            alphas_until_points, None, ray_indices=indices, n_rays=total_pixels
        ).reshape(C, image_height, image_width, 1)

    weights, trans = render_weight_from_alpha(alphas, ray_indices=indices, n_rays=total_pixels)
    renders = accumulate_along_rays(
        weights,
        render_features,
        ray_indices=indices,
        n_rays=total_pixels,
    ).reshape(C, image_height, image_width, channels)
    alphas = accumulate_along_rays(weights, None, ray_indices=indices, n_rays=total_pixels).reshape(
        C, image_height, image_width, 1
    )

    return renders, alphas, alpha_sum_until_pts if compute_alpha_sum_until_pts else None


def _rasterize_to_pixels(
    means2d: Tensor,  # [C, N, 2]
    conics: Tensor,  # [C, N, 3]
    colors: Tensor,  # [C, N, channels]
    opacities: Tensor,  # [C, N]
    pix_vels: Tensor,  # [C, N, 2]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [C, tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    rolling_shutter_time: Tensor,  # [C]
    backgrounds: Optional[Tensor] = None,  # [C, channels]
    batch_per_iter: int = 100,
    rolling_shutter_direction: int = 1,
):
    """Pytorch implementation of `gsplat.cuda._wrapper.rasterize_to_pixels()`.

    This function rasterizes 2D Gaussians to pixels in a Pytorch-friendly way. It
    iteratively accumulates the renderings within each batch of Gaussians. The
    interations are controlled by `batch_per_iter`.

    .. note::
        This is a minimal implementation of the fully fused version, which has more
        arguments. Not all arguments are supported.

    .. note::

        This function relies on Pytorch's autograd for the backpropagation. It is much slower
        than our fully fused rasterization implementation and comsumes much more GPU memory.
        But it could serve as a playground for new ideas or debugging, as no backward
        implementation is needed.

    .. warning::

        This function requires the `nerfacc` package to be installed. Please install it
        using the following command `pip install nerfacc`.
    """
    from ._wrapper import rasterize_to_indices_in_range

    C, N = means2d.shape[:2]
    n_isects = len(flatten_ids)
    device = means2d.device

    render_colors = torch.zeros((C, image_height, image_width, colors.shape[-1]), device=device)
    render_alphas = torch.zeros((C, image_height, image_width, 1), device=device)

    # Split Gaussians into batches and iteratively accumulate the renderings
    block_size = tile_size * tile_size
    isect_offsets_fl = torch.cat([isect_offsets.flatten(), torch.tensor([n_isects], device=device)])
    max_range = (isect_offsets_fl[1:] - isect_offsets_fl[:-1]).max().item()
    num_batches = (max_range + block_size - 1) // block_size
    for step in range(0, num_batches, batch_per_iter):
        transmittances = 1.0 - render_alphas[..., 0]

        # Find the M intersections between pixels and gaussians.
        # Each intersection corresponds to a tuple (gs_id, pixel_id, camera_id)
        gs_ids, pixel_ids, camera_ids = rasterize_to_indices_in_range(
            step,
            step + batch_per_iter,
            transmittances,
            means2d,
            conics,
            opacities,
            pix_vels,
            image_width,
            image_height,
            tile_size,
            isect_offsets,
            flatten_ids,
            rolling_shutter_time,
        )  # [M], [M]
        if len(gs_ids) == 0:
            break

        # Accumulate the renderings within this batch of Gaussians.
        renders_step, accs_step = accumulate(
            means2d,
            conics,
            opacities,
            colors,
            pix_vels,
            rolling_shutter_time,
            gs_ids,
            pixel_ids,
            camera_ids,
            image_width,
            image_height,
        )
        render_colors = render_colors + renders_step * transmittances[..., None]
        render_alphas = render_alphas + accs_step * transmittances[..., None]

    if backgrounds is not None:
        render_colors = render_colors + backgrounds[:, None, None, :] * (1.0 - render_alphas)

    return render_colors, render_alphas


def _rasterize_to_points(
    means2d: Tensor,  # [C, N, 2]
    conics: Tensor,  # [C, N, 3]
    lidar_features: Tensor,  # [C, N, channels]
    opacities: Tensor,  # [C, N]
    pix_vels: Tensor,  # [C, N, 3] or [nnz, 3]
    depth_compensations: Tensor,  # [C, N, 2]
    raster_pts: Tensor,  # [C, image_height, image_width, 4], azim,elev,range,time
    image_width: int,
    image_height: int,
    tile_width: int,
    tile_height: int,
    isect_offsets: Tensor,  # [C, tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    compute_alpha_sum_until_points: bool,
    compute_alpha_sum_until_points_threshold: float,
    batch_per_iter: int = 100,
):
    """Pytorch implementation of `gsplat.cuda._wrapper.rasterize_to_points()`.

    This function rasterizes 2D Gaussians to points in a Pytorch-friendly way. It
    iteratively accumulates the renderings within each batch of Gaussians. The
    interations are controlled by `batch_per_iter`.

    .. note::
        This is a minimal implementation of the fully fused version, which has more
        arguments. Not all arguments are supported.

    .. note::

        This function relies on Pytorch's autograd for the backpropagation. It is much slower
        than our fully fused rasterization implementation and comsumes much more GPU memory.
        But it could serve as a playground for new ideas or debugging, as no backward
        implementation is needed.

    .. warning::

        This function requires the `nerfacc` package to be installed. Please install it
        using the following command `pip install nerfacc`.
    """
    from ._wrapper import rasterize_to_indices_in_range_lidar

    C, N = means2d.shape[:2]
    n_isects = len(flatten_ids)
    device = means2d.device

    render_lidar_features = torch.zeros((C, image_height, image_width, lidar_features.shape[-1]), device=device)
    render_alphas = torch.zeros((C, image_height, image_width, 1), device=device)

    # Split Gaussians into batches and iteratively accumulate the renderings
    block_size = tile_width * tile_height
    isect_offsets_fl = torch.cat([isect_offsets.flatten(), torch.tensor([n_isects], device=device)])
    max_range = (isect_offsets_fl[1:] - isect_offsets_fl[:-1]).max().item()
    num_batches = (max_range + block_size - 1) // block_size
    alpha_sum_until_points = torch.zeros_like(render_alphas) if compute_alpha_sum_until_points else None
    for step in range(0, num_batches, batch_per_iter):
        transmittances = 1.0 - render_alphas[..., 0]

        # Find the M intersections between pixels and gaussians.
        # Each intersection corresponds to a tuple (gs_id, pixel_id, camera_id)
        gs_ids, pixel_ids, camera_ids = rasterize_to_indices_in_range_lidar(
            step,
            step + batch_per_iter,
            transmittances,
            means2d,
            conics,
            opacities,
            pix_vels,
            raster_pts,
            image_width,
            image_height,
            tile_width,
            tile_height,
            isect_offsets,
            flatten_ids,
        )  # [M], [M]
        if len(gs_ids) == 0:
            break

        # Accumulate the renderings within this batch of Gaussians.
        renders_step, accs_step, alpha_sum_until_points_step = accumulate_lidar(
            means2d,
            conics,
            opacities,
            lidar_features,
            pix_vels,
            depth_compensations,
            raster_pts,
            gs_ids,
            pixel_ids,
            camera_ids,
            image_width,
            image_height,
            compute_alpha_sum_until_pts=compute_alpha_sum_until_points,
            compute_alpha_sum_until_pts_threshold=compute_alpha_sum_until_points_threshold,
        )
        render_lidar_features = render_lidar_features + renders_step * transmittances[..., None]
        render_alphas = render_alphas + accs_step * transmittances[..., None]
        if compute_alpha_sum_until_points:
            assert alpha_sum_until_points_step is not None
            alpha_sum_until_points += alpha_sum_until_points_step

    return render_lidar_features, render_alphas, alpha_sum_until_points


def _eval_sh_bases_fast(basis_dim: int, dirs: Tensor):
    """
    Evaluate spherical harmonics bases at unit direction for high orders
    using approach described by
    Efficient Spherical Harmonic Evaluation, Peter-Pike Sloan, JCGT 2013
    https://jcgt.org/published/0002/02/06/


    :param basis_dim: int SH basis dim. Currently, only 1-25 square numbers supported
    :param dirs: torch.Tensor (..., 3) unit directions

    :return: torch.Tensor (..., basis_dim)

    See reference C++ code in https://jcgt.org/published/0002/02/06/code.zip
    """
    result = torch.empty((*dirs.shape[:-1], basis_dim), dtype=dirs.dtype, device=dirs.device)

    result[..., 0] = 0.2820947917738781

    if basis_dim <= 1:
        return result

    x, y, z = dirs.unbind(-1)

    fTmpA = -0.48860251190292
    result[..., 2] = -fTmpA * z
    result[..., 3] = fTmpA * x
    result[..., 1] = fTmpA * y

    if basis_dim <= 4:
        return result

    z2 = z * z
    fTmpB = -1.092548430592079 * z
    fTmpA = 0.5462742152960395
    fC1 = x * x - y * y
    fS1 = 2 * x * y
    result[..., 6] = 0.9461746957575601 * z2 - 0.3153915652525201
    result[..., 7] = fTmpB * x
    result[..., 5] = fTmpB * y
    result[..., 8] = fTmpA * fC1
    result[..., 4] = fTmpA * fS1

    if basis_dim <= 9:
        return result

    fTmpC = -2.285228997322329 * z2 + 0.4570457994644658
    fTmpB = 1.445305721320277 * z
    fTmpA = -0.5900435899266435
    fC2 = x * fC1 - y * fS1
    fS2 = x * fS1 + y * fC1
    result[..., 12] = z * (1.865881662950577 * z2 - 1.119528997770346)
    result[..., 13] = fTmpC * x
    result[..., 11] = fTmpC * y
    result[..., 14] = fTmpB * fC1
    result[..., 10] = fTmpB * fS1
    result[..., 15] = fTmpA * fC2
    result[..., 9] = fTmpA * fS2

    if basis_dim <= 16:
        return result

    fTmpD = z * (-4.683325804901025 * z2 + 2.007139630671868)
    fTmpC = 3.31161143515146 * z2 - 0.47308734787878
    fTmpB = -1.770130769779931 * z
    fTmpA = 0.6258357354491763
    fC3 = x * fC2 - y * fS2
    fS3 = x * fS2 + y * fC2
    result[..., 20] = 1.984313483298443 * z2 * (1.865881662950577 * z2 - 1.119528997770346) + -1.006230589874905 * (
        0.9461746957575601 * z2 - 0.3153915652525201
    )
    result[..., 21] = fTmpD * x
    result[..., 19] = fTmpD * y
    result[..., 22] = fTmpC * fC1
    result[..., 18] = fTmpC * fS1
    result[..., 23] = fTmpB * fC2
    result[..., 17] = fTmpB * fS2
    result[..., 24] = fTmpA * fC3
    result[..., 16] = fTmpA * fS3
    return result


def _spherical_harmonics(
    degree: int,
    dirs: torch.Tensor,  # [..., 3]
    coeffs: torch.Tensor,  # [..., K, 3]
):
    """Pytorch implementation of `gsplat.cuda._wrapper.spherical_harmonics()`."""
    dirs = F.normalize(dirs, p=2, dim=-1)
    num_bases = (degree + 1) ** 2
    bases = torch.zeros_like(coeffs[..., 0])
    bases[..., :num_bases] = _eval_sh_bases_fast(num_bases, dirs)
    return (bases[..., None] * coeffs).sum(dim=-2)
