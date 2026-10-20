"""Save inference camera mosaics and scene videos."""
import os
import tempfile
import cv2
import numpy as np
def _rgb_tensor_to_bgr(t):
    img = t.permute(1, 2, 0).cpu().numpy()
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def save_current_camera_mosaic(rgb_pred, cam_ids, camera_rows, output_path):
    """Save current-view renders in the requested camera row layout."""
    ids = [int(cam_id) for cam_id in cam_ids]
    layout_ids = [cam_id for row in camera_rows for cam_id in row]
    if sorted(ids) != sorted(layout_ids):
        raise ValueError(f"Rendered camera IDs {ids} do not match layout {camera_rows}")
    view_by_cam = {cam_id: view_idx for view_idx, cam_id in enumerate(ids)}
    mosaic = np.concatenate(
        [
            np.concatenate(
                [_rgb_tensor_to_bgr(rgb_pred[view_by_cam[cam_id]]) for cam_id in row],
                axis=1,
            )
            for row in camera_rows
        ],
        axis=0,
    )
    if not cv2.imwrite(output_path, mosaic):
        raise IOError(f"Failed to save rendered camera mosaic: {output_path}")

def create_current_scene_video(work_dir: str, scene: str, image_suffix: str, fps: int = 10):
    """Combine every saved current-view mosaic for one scene into a video."""
    scene_dir = os.path.join(work_dir, "combined", str(scene))
    suffix = f"_{image_suffix}.png"
    if not os.path.isdir(scene_dir):
        return None
    frame_paths = [
        (name[:-len(suffix)], os.path.join(scene_dir, name))
        for name in os.listdir(scene_dir)
        if name.endswith(suffix)
    ]
    if not frame_paths:
        return None

    def frame_sort_key(item):
        frame_id = item[0]
        return (0, int(frame_id)) if frame_id.isdecimal() else (1, frame_id)

    frame_paths.sort(key=frame_sort_key)
    first_image = cv2.imread(frame_paths[0][1])
    if first_image is None:
        raise IOError(f"Cannot read rendered frame: {frame_paths[0][1]}")
    height, width = first_image.shape[:2]
    video_path = os.path.join(scene_dir, f"current_{fps}fps.mp4")
    fd, temporary_path = tempfile.mkstemp(prefix=".current_", suffix=".mp4", dir=scene_dir)
    os.close(fd)
    writer = None
    try:
        writer = cv2.VideoWriter(
            temporary_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise IOError(f"Cannot open video writer: {temporary_path}")
        for index, (_, image_path) in enumerate(frame_paths):
            image = first_image if index == 0 else cv2.imread(image_path)
            if image is None:
                raise IOError(f"Cannot read rendered frame: {image_path}")
            if image.shape != first_image.shape:
                raise ValueError(
                    f"Rendered frame has shape {image.shape}, expected {first_image.shape}: "
                    f"{image_path}"
                )
            writer.write(image)
        writer.release()
        writer = None
        if os.path.getsize(temporary_path) == 0:
            raise IOError(f"Video writer produced an empty file: {temporary_path}")
        os.replace(temporary_path, video_path)
    finally:
        if writer is not None:
            writer.release()
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return video_path, len(frame_paths)
