"""Current-frame inference loader for the preprocessed driving datasets."""

import copy
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import inv, geotrf, rescale_image_depthmap_custom


def stable_hash(value):
    return int(hashlib.md5(str(value).encode("utf-8")).hexdigest(), 16) % (2**16 - 1) + 1


def imread_cv2(path, options=cv2.IMREAD_COLOR):
    if path.lower().endswith(".exr"):
        options = cv2.IMREAD_ANYDEPTH
    image = cv2.imread(path, options)
    if image is None:
        raise IOError(f"Cannot read image or depth map: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image


def camera_rays(intrinsics, camera_to_lidar, resolution):
    width, height = (int(value) for value in resolution)
    x, y = torch.meshgrid(torch.arange(width) + 0.5, torch.arange(height) + 0.5, indexing="xy")
    fx, fy = intrinsics[:, 0, 0], intrinsics[:, 1, 1]
    cx, cy = intrinsics[:, 0, 2], intrinsics[:, 1, 2]
    directions = torch.stack(((x - cx[:, None, None]) / fx[:, None, None],
                              (y - cy[:, None, None]) / fy[:, None, None],
                              torch.ones((len(intrinsics), height, width))), dim=-1)
    rays_d = (directions[..., None, :] * camera_to_lidar[:, None, None, :3, :3]).sum(-1)
    rays_o = camera_to_lidar[:, None, None, :3, 3].expand_as(rays_d)
    return rays_o, rays_d


class DrivingDataset(Dataset):
    def __init__(self, scene_root, cfg, scene_ids=None):
        self.scene_root = str(scene_root)
        self.cfg = cfg
        self.num_cameras = int(cfg.num_cameras)
        self.resolution = list(cfg.resolution)
        self.render_resolution = list(cfg.get("render_resolution", self.resolution))
        scenes = sorted(p.name for p in Path(scene_root).iterdir() if (p / "images").is_dir())
        if scene_ids is not None:
            numeric = {int(s): s for s in scenes if s.isdecimal()}
            selected = set()
            for token in scene_ids:
                if token in scenes:
                    selected.add(token)
                elif token.isdecimal() and int(token) in numeric:
                    selected.add(numeric[int(token)])
                else:
                    raise ValueError(f"Scene ID {token!r} does not exist under {scene_root}")
            scenes = [s for s in scenes if s in selected]
        self.samples = []
        for scene in scenes:
            frames = sorted({p.stem.rsplit("_", 1)[0] for p in (Path(scene_root) / scene / "images").glob("*.png")})
            self.samples.extend((scene, frame) for frame in frames)
        if not self.samples:
            raise ValueError(f"No preprocessed scenes found under {scene_root}; see the data layout in README.md")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        current = self.load_instance(index)
        render = current if self.render_resolution == self.resolution else self.load_instance(index, resolution_override=self.render_resolution)
        intrinsics = torch.as_tensor(current["intrinsics"], dtype=torch.float32)
        poses = current["input_c2ws"]
        rays_o, rays_d = camera_rays(intrinsics, poses, self.resolution)
        current["input_dict_gs"] = {
            "ck": intrinsics, "c2w": poses,
            "fx": intrinsics[:, 0, 0], "fy": intrinsics[:, 1, 1],
            "cx": intrinsics[:, 0, 2], "cy": intrinsics[:, 1, 2],
            "rays_o": rays_o, "rays_d": rays_d,
        }
        output_intrinsics = torch.as_tensor(render["intrinsics"], dtype=torch.float32)
        output_rays_o, output_rays_d = camera_rays(output_intrinsics, poses, self.render_resolution)
        current["output_dict_gs"] = {
            "rgb": render["images"], "c2w": poses,
            "fovx": 2 * torch.atan(output_intrinsics[:, 0, 2] / output_intrinsics[:, 0, 0]),
            "fovy": 2 * torch.atan(output_intrinsics[:, 1, 2] / output_intrinsics[:, 1, 1]),
            "rays_o": output_rays_o, "rays_d": output_rays_d,
            "sky_mask": torch.as_tensor(render["sky_mask"], dtype=torch.bool),
            "dynamics_region": render["dynamics_region"],
            "single_depthmaps": render["single_depthmaps"],
            "valid_mask": torch.as_tensor(render["valid_mask"], dtype=torch.bool),
            "input_valid_mask": torch.as_tensor(current["valid_mask"], dtype=torch.bool),
            "intrinsics": output_intrinsics,
            "view_cam_ids": torch.arange(self.num_cameras),
            "view_frame_ids": torch.full((self.num_cameras,), int(current["frame"]), dtype=torch.long),
        }
        return current

    def load_instance(self, idx, select_cam=None, resolution_override=None):
        scene, frame_name = self.samples[idx]
        actual_root = self.scene_root
        images = []
        camera_poses = []
        intrinsics_list = []
        camera2lidar_list = []
        single_frame_depths = []
        sky_mask_list = []
        valid_mask_list = []
        input_c2ws = []
        dynamics_region_list = []

        dynamics_path = os.path.join(actual_root, scene, "dynamics")
        with open(os.path.join(dynamics_path, "dynamic_infos.json"), "r") as f:
            dynamic_info = json.load(f)["track_id_infos"][str(int(frame_name))]

        for i in range(1, self.num_cameras + 1):
            if select_cam is not None and i not in select_cam:
                continue

            image_path = os.path.join(
                actual_root, scene, "images", f"{frame_name}_{i-1}.png"
            )
            depthmap_path = os.path.join(
                actual_root, scene, f"{frame_name[1:]}_{i}.exr"
            )
            calib_path = os.path.join(actual_root, scene, f"{frame_name[1:]}_{i}.npz")
            with np.load(
                os.path.join(dynamics_path, f"dynamic_mask_{int(frame_name)}_{i}.npz")
            ) as loaded_data:
                dynamics_mask = loaded_data["data"]
            dynamics_region = copy.deepcopy(dynamics_mask)
            for track_id in dynamic_info[str(i)]:
                track_speed = dynamic_info[str(i)][track_id]["speed"]
                track_speed = np.linalg.norm(track_speed)
                if track_speed < 0.2:
                    track_id_int = stable_hash(str(track_id))
                    dynamics_region[dynamics_mask == track_id_int] = 0
            dynamics_region = dynamics_region > 0

            image = imread_cv2(image_path)
            moge_depth_mask = os.path.join(
                actual_root, scene, f"{frame_name[1:]}_{i}_moge_mask.png"
            )
            sky_mask = cv2.imread(moge_depth_mask, cv2.IMREAD_GRAYSCALE) > 0
            depthmap = imread_cv2(depthmap_path)
            calib = np.load(calib_path)

            camera2lidar = np.float32(calib["cam2lidar"])
            intrinsics = np.float32(calib["intrinsics"])
            camera_pose = np.float32(calib["cam2world"])
            single_frame_depth = depthmap.copy()
            resolution = (
                resolution_override
                if resolution_override is not None
                else self.resolution
            )

            valid_mask = np.ones(image.shape[:2], dtype=bool)

            image, _, tmp_intrinsics = rescale_image_depthmap_custom(
                image, None, intrinsics, resolution, force=True
            )
            W, H = resolution
            single_frame_depth_resize = np.zeros((H, W), dtype=np.float32)
            y, x = np.where(single_frame_depth > 0)
            pos2d = np.stack([x, y], axis=-1)  # (N, 2)
            depth_value = single_frame_depth[y, x]
            pos2d = (
                geotrf(tmp_intrinsics @ inv(intrinsics), pos2d)
                .round()
                .astype(np.int16)
            )
            x, y = pos2d.T
            single_frame_depth_resize[
                y.clip(min=0, max=H - 1), x.clip(min=0, max=W - 1)
            ] = depth_value
            sky_mask = cv2.resize(
                sky_mask.astype(np.float32), resolution, interpolation=cv2.INTER_NEAREST
            )
            dynamics_region = cv2.resize(
                dynamics_region.astype(np.float32),
                resolution,
                interpolation=cv2.INTER_NEAREST,
            )
            valid_mask = cv2.resize(
                valid_mask.astype(np.float32), resolution, interpolation=cv2.INTER_NEAREST
            )
            intrinsics = tmp_intrinsics

            images.append(image)
            camera_poses.append(camera_pose)
            intrinsics_list.append(intrinsics)
            camera2lidar_list.append(camera2lidar)
            single_frame_depths.append(single_frame_depth_resize)
            input_c2ws.append(camera2lidar)
            sky_mask_list.append(sky_mask)
            dynamics_region_list.append(dynamics_region)
            valid_mask_list.append(valid_mask)

        to_tensor_transform = transforms.ToTensor()
        tensor_list = []
        for image in images:
            tensor_list.append(to_tensor_transform(image))
        images = torch.stack(tensor_list, dim=0)
        camera_poses = np.stack(camera_poses, axis=0)
        intrinsics_list = np.stack(intrinsics_list, axis=0)
        camera2lidar_list = np.stack(camera2lidar_list, axis=0)
        single_frame_depths = np.stack(single_frame_depths, axis=0)
        input_c2ws = torch.as_tensor(np.stack(input_c2ws, axis=0), dtype=torch.float32)
        sky_mask_list = np.stack(sky_mask_list, axis=0)

        return {
            "images": images,
            "scene": scene,
            "frame": frame_name,
            "camera_pose": camera_poses,
            "intrinsics": intrinsics_list,
            "camera2lidar": camera2lidar_list,
            "single_depthmaps": single_frame_depths,
            "input_c2ws": input_c2ws,
            "sky_mask": sky_mask_list,
            "dynamics_region": np.stack(dynamics_region_list, axis=0),
            "valid_mask": np.stack(valid_mask_list, axis=0),
        }

