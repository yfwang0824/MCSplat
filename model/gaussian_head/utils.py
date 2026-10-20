# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# VGGT positional embedding and UV-grid helpers retain the VGGT License
# in ../../licenses/VGGT.txt. Modified for MCSplat inference; see
# ../../THIRD_PARTY_NOTICES.md for the scope of third-party code.

import torch
import torch.nn as nn

def position_grid_to_embed(pos_grid: torch.Tensor, embed_dim: int, omega_0: float = 100) -> torch.Tensor:
    """
    Convert 2D position grid (HxWx2) to sinusoidal embeddings (HxWxC)

    Args:
        pos_grid: Tensor of shape (H, W, 2) containing 2D coordinates
        embed_dim: Output channel dimension for embeddings

    Returns:
        Tensor of shape (H, W, embed_dim) with positional embeddings
    """
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2
    pos_flat = pos_grid.reshape(-1, grid_dim)  # Flatten to (H*W, 2)

    # Process x and y coordinates separately
    emb_x = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 0], omega_0=omega_0)  # [1, H*W, D/2]
    emb_y = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 1], omega_0=omega_0)  # [1, H*W, D/2]

    # Combine and reshape
    emb = torch.cat([emb_x, emb_y], dim=-1)  # [1, H*W, D]

    return emb.view(H, W, embed_dim)

def make_sincos_pos_embed(embed_dim: int, pos: torch.Tensor, omega_0: float = 100) -> torch.Tensor:
    """
    This function generates a 1D positional embedding from a given grid using sine and cosine functions.

    Args:
    - embed_dim: The embedding dimension.
    - pos: The position to generate the embedding from.

    Returns:
    - emb: The generated 1D positional embedding.
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.double, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / omega_0**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb.float()

def create_uv_grid(
    width: int, height: int, aspect_ratio: float = None, dtype: torch.dtype = None, device: torch.device = None
) -> torch.Tensor:
    """
    Create a normalized UV grid of shape (width, height, 2).

    The grid spans horizontally and vertically according to an aspect ratio,
    ensuring the top-left corner is at (-x_span, -y_span) and the bottom-right
    corner is at (x_span, y_span), normalized by the diagonal of the plane.

    Args:
        width (int): Number of points horizontally.
        height (int): Number of points vertically.
        aspect_ratio (float, optional): Width-to-height ratio. Defaults to width/height.
        dtype (torch.dtype, optional): Data type of the resulting tensor.
        device (torch.device, optional): Device on which the tensor is created.

    Returns:
        torch.Tensor: A (width, height, 2) tensor of UV coordinates.
    """
    # Derive aspect ratio if not explicitly provided
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)

    # Compute normalized spans for X and Y
    diag_factor = (aspect_ratio**2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor

    # Establish the linspace boundaries
    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height

    # Generate 1D coordinates
    x_coords = torch.linspace(left_x, right_x, steps=width, dtype=dtype, device=device)
    y_coords = torch.linspace(top_y, bottom_y, steps=height, dtype=dtype, device=device)

    # Create 2D meshgrid (width x height) and stack into UV
    uu, vv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    uv_grid = torch.stack((uu, vv), dim=-1)

    return uv_grid

def is_point_in_frustum_batch(points_3d, K, R_c2w, t_c2w, near, far, image_width, image_height):
    N = points_3d.shape[0]
    M = K.shape[0]
    
    R_w2c = R_c2w.transpose(-2, -1)  # (M, 3, 3)
    t_w2c = -torch.bmm(R_w2c, t_c2w.unsqueeze(-1)).squeeze(-1)  # (M, 3)
    
    points_expanded = points_3d.unsqueeze(1).expand(N, M, 3)  # (N, M, 3)
    R_expanded = R_w2c.unsqueeze(0).expand(N, M, 3, 3)  # (N, M, 3, 3)
    t_expanded = t_w2c.unsqueeze(0).expand(N, M, 3)  # (N, M, 3)

    points_cam = torch.matmul(R_expanded, points_expanded.unsqueeze(-1)).squeeze(-1) + t_expanded
    
    z = points_cam[:, :, 2]
    depth_mask = (z >= near) & (z <= far) & (z > 1e-8)
    K_expanded = K.unsqueeze(0).expand(N, M, 3, 3)  # (N, M, 3, 3)
    points_2d_homo = torch.matmul(K_expanded, points_cam.unsqueeze(-1)).squeeze(-1)  # (N, M, 3)
    
    z_homo = points_2d_homo[:, :, 2]
    valid_projection = (z_homo.abs() > 1e-8) & depth_mask
    
    points_2d = torch.zeros_like(points_2d_homo[:, :, :2])
    if valid_projection.any():
        points_2d[valid_projection] = points_2d_homo[valid_projection][:, :2] / z_homo[valid_projection].unsqueeze(-1)
    
    
    image_mask = (points_2d[:, :, 0] >= 0) & (points_2d[:, :, 0] < image_width) & \
                 (points_2d[:, :, 1] >= 0) & (points_2d[:, :, 1] < image_height)
    
    final_mask = valid_projection & image_mask
    
    return final_mask, points_2d.int()
