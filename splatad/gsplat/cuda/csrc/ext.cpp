#include "bindings.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_sh_fwd", &compute_sh_fwd_tensor);
    m.def("compute_sh_bwd", &compute_sh_bwd_tensor);

    m.def("quat_scale_to_covar_preci_fwd", &quat_scale_to_covar_preci_fwd_tensor);
    m.def("quat_scale_to_covar_preci_bwd", &quat_scale_to_covar_preci_bwd_tensor);

    m.def("persp_proj_fwd", &persp_proj_fwd_tensor);
    m.def("persp_proj_bwd", &persp_proj_bwd_tensor);

    m.def("lidar_proj_fwd", &lidar_proj_fwd_tensor);
    m.def("lidar_proj_bwd", &lidar_proj_bwd_tensor);

    m.def("world_to_cam_fwd", &world_to_cam_fwd_tensor);
    m.def("world_to_cam_bwd", &world_to_cam_bwd_tensor);

    m.def("compute_pix_velocity_fwd", &compute_pix_velocity_fwd_tensor);
    m.def("compute_pix_velocity_bwd", &compute_pix_velocity_bwd_tensor);
    
    m.def("compute_lidar_velocity_fwd", &compute_lidar_velocity_fwd_tensor);
    m.def("compute_lidar_velocity_bwd", &compute_lidar_velocity_bwd_tensor);

    m.def("fully_fused_projection_fwd", &fully_fused_projection_fwd_tensor);
    m.def("fully_fused_projection_bwd", &fully_fused_projection_bwd_tensor);

    m.def("fully_fused_lidar_projection_fwd", &fully_fused_lidar_projection_fwd_tensor);
    m.def("fully_fused_lidar_projection_bwd", &fully_fused_lidar_projection_bwd_tensor);

    m.def("isect_tiles", &isect_tiles_tensor);
    m.def("isect_lidar_tiles", &isect_lidar_tiles_tensor);
    m.def("isect_offset_encode", &isect_offset_encode_tensor);

    m.def("map_points_to_lidar_tiles", &map_points_to_lidar_tiles_tensor);
    m.def("points_mapping_offset_encode", &points_mapping_offset_encode_tensor);
    m.def("populate_image_from_points", &populate_image_from_points_tensor);

    m.def("rasterize_to_pixels_fwd", &rasterize_to_pixels_fwd_tensor);
    m.def("rasterize_to_pixels_bwd", &rasterize_to_pixels_bwd_tensor);

    m.def("rasterize_to_points_fwd", &rasterize_to_points_fwd_tensor);
    m.def("rasterize_to_points_bwd", &rasterize_to_points_bwd_tensor);

    m.def("rasterize_to_indices_in_range", &rasterize_to_indices_in_range_tensor);
    m.def("rasterize_to_indices_in_range_lidar", &rasterize_to_indices_in_range_lidar_tensor);

    // packed version
    m.def("fully_fused_projection_packed_fwd", &fully_fused_projection_packed_fwd_tensor);
    m.def("fully_fused_projection_packed_bwd", &fully_fused_projection_packed_bwd_tensor);
    
    m.def("compute_relocation", &compute_relocation_tensor);
}