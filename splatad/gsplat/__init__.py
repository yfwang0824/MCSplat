import warnings

from .cuda._torch_impl import accumulate
from .cuda._wrapper import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    persp_proj,
    quat_scale_to_covar_preci,
    rasterize_to_indices_in_range,
    rasterize_to_pixels,
    rasterize_to_points,
    spherical_harmonics,
    world_to_cam,
    map_points_to_lidar_tiles,
    points_mapping_offset_encode,
    populate_image_from_points,
)
from .rendering import (
    rasterization,
    rasterization_inria_wrapper,
)
from .version import __version__


all = [
    "rasterization",
    "rasterization_inria_wrapper",
    "spherical_harmonics",
    "isect_offset_encode",
    "isect_tiles",
    "isect_lidar_tiles",
    "map_points_to_lidar_tiles",
    "points_mapping_offset_encode",
    "populate_image_from_points",
    "persp_proj",
    "fully_fused_projection",
    "quat_scale_to_covar_preci",
    "rasterize_to_pixels",
    "rasterize_to_points",
    "world_to_cam",
    "accumulate",
    "rasterize_to_indices_in_range",
    "rasterize_to_indices_in_range_lidar",
    "__version__",
]
