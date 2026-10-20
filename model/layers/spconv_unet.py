# Contains helpers adapted from OpenPCDet (OpenMMLab), Apache-2.0.
# Modified for MCSplat inference; see ../../THIRD_PARTY_NOTICES.md.
from functools import partial

import torch
import torch.nn as nn
import spconv
if float(spconv.__version__[2:]) >= 2.2:
    spconv.constants.SPCONV_USE_DIRECT_TABLE = False
    
try:
    import spconv.pytorch as spconv
except:
    import spconv as spconv

def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out

def get_voxel_centers(voxel_coords, downsample_times, voxel_size, point_cloud_range):
    """
    Args:
        voxel_coords: (N, 3)
        downsample_times:
        voxel_size:
        point_cloud_range:

    Returns:

    """
    assert voxel_coords.shape[1] == 3
    voxel_centers = voxel_coords[:, [2, 1, 0]].float()  # (xyz)
    if isinstance(voxel_size, torch.Tensor) is False:
        voxel_size = torch.tensor(voxel_size, device=voxel_centers.device).float() * downsample_times
    else:
        voxel_size = voxel_size.float() * downsample_times
    if isinstance(point_cloud_range, torch.Tensor) is False:
        pc_range = torch.tensor(point_cloud_range[0:3], device=voxel_centers.device).float()
    else:
        pc_range = point_cloud_range[0:3].float()
    voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
    return voxel_centers


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, indice_key=None, norm_fn=None):
        super(SparseBasicBlock, self).__init__()
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x.features

        assert x.features.dim() == 2, 'x.features.dim()=%d' % x.features.dim()

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity)
        out = replace_feature(out, self.relu(out.features))

        return out


def project_world_points_to_images(world_points, camera_intrinsics, camera_poses, image_height, image_width):
    """
    Project 3D points in world coordinates to multiple camera image planes
    
    Args:
        world_points: (N, 3) 3D points in world coordinate system
        camera_intrinsics: (M, 3, 3) camera intrinsic matrices for M cameras
        camera_poses: (M, 4, 4) or (M, 3, 4) cam2world transformation matrices for M cameras
        image_height: height of the images
        image_width: width of the images
    
    Returns:
        projected_points: (M, N, 2) projected pixel coordinates (u, v) for all cameras
        depths: (M, N) depth values for all cameras
        valid_masks: (M, N) boolean masks for valid points for each camera
    """
    device = world_points.device
    M = camera_intrinsics.shape[0]
    N = world_points.shape[0]
    
    # Ensure inputs are correct tensor types
    world_points = world_points.float()
    camera_intrinsics = camera_intrinsics.float()
    camera_poses = camera_poses.float()
    
    # Handle 3x4 camera poses by padding to 4x4
    if camera_poses.shape[-2] == 3:
        bottom_rows = torch.tensor([[[0, 0, 0, 1]]], dtype=torch.float32, device=device).repeat(M, 1, 1)
        camera_poses = torch.cat([camera_poses, bottom_rows], dim=-2)
    
    # Compute world2cam transformation matrices for all cameras
    world2cam_matrices = torch.inverse(camera_poses)  # (M, 4, 4)
    R_world2cam = world2cam_matrices[:, :3, :3]       # (M, 3, 3)
    t_world2cam = world2cam_matrices[:, :3, 3]        # (M, 3)
    
    # Transform world coordinates to camera coordinates for all cameras
    # Broadcasting: (M, 3, 3) @ (N, 3) + (M, 3) -> (M, N, 3)
    world_points_expanded = world_points.unsqueeze(0).expand(M, -1, -1)  # (M, N, 3)
    cam_points = torch.bmm(R_world2cam, world_points_expanded.transpose(-1, -2)).transpose(-1, -2) + t_world2cam.unsqueeze(1)  # (M, N, 3)
    
    # Extract camera coordinates
    x_cam = cam_points[:, :, 0]  # (M, N)
    y_cam = cam_points[:, :, 1]  # (M, N)
    z_cam = cam_points[:, :, 2]  # (M, N)
    
    # Extract intrinsic parameters for all cameras
    fx = camera_intrinsics[:, 0, 0].unsqueeze(1)  # (M, 1)
    fy = camera_intrinsics[:, 1, 1].unsqueeze(1)  # (M, 1)
    cx = camera_intrinsics[:, 0, 2].unsqueeze(1)  # (M, 1)
    cy = camera_intrinsics[:, 1, 2].unsqueeze(1)  # (M, 1)
    
    # Perform perspective projection for all cameras
    u = (x_cam * fx / z_cam) + cx  # (M, N)
    v = (y_cam * fy / z_cam) + cy  # (M, N)
    
    # Create validity masks
    depth_mask = z_cam > 0  # (M, N)
    boundary_mask = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)  # (M, N)
    valid_masks = depth_mask & boundary_mask  # (M, N)
    
    # Stack projected coordinates
    projected_points = torch.stack([u, v], dim=-1)  # (M, N, 2)
    
    return projected_points, z_cam, valid_masks

    
