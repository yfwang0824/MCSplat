#!/usr/bin/env python3
"""Render MCSplat checkpoints on preprocessed Waymo or nuScenes scenes."""

import argparse
import logging
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("waymo", "nuscenes"))
    parser.add_argument("--data-path", required=True, type=Path, help="Root containing preprocessed scene folders")
    parser.add_argument("--checkpoint", type=Path, help="Defaults to checkpoints/<dataset>/model.safetensors")
    parser.add_argument("--output-dir", "--work-dir-root", dest="output_dir", type=Path)
    parser.add_argument("--scene-ids", "--scene-indices", "--scene_indices", dest="scene_ids", help="Comma-separated scene directory IDs, e.g. 000,001")
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=4)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--fps", type=int, default=10, help="Output video playback frame rate")
    parser.add_argument("--resume", action="store_true", help="Skip existing frame PNGs and rebuild scene videos")
    parser.add_argument("--no-video", action="store_true", help="Save PNG mosaics without encoding videos")
    parser.add_argument("--check-only", action="store_true", help="Validate checkpoint metadata and one input sample without a GPU")
    args = parser.parse_args()
    if args.num_workers < 0 or args.fps <= 0 or (args.max_frames is not None and args.max_frames <= 0):
        parser.error("--num-workers must be nonnegative; --fps and --max-frames must be positive")
    args.data_path = args.data_path.expanduser().resolve()
    args.checkpoint = (args.checkpoint or ROOT / "checkpoints" / args.dataset / "model.safetensors").expanduser().resolve()
    args.output_dir = (args.output_dir or ROOT / "outputs" / args.dataset).expanduser().resolve()
    if not args.data_path.is_dir():
        parser.error(f"Dataset root does not exist: {args.data_path}")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}. Download it as described in README.md.")
    if args.scene_ids is not None:
        args.scene_ids = [s.strip() for s in args.scene_ids.split(",") if s.strip()]
        if not args.scene_ids:
            parser.error("--scene-ids cannot be empty")
    return args


def load_config(dataset):
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(ROOT / "configs" / f"{dataset}.yaml")
    cfg.Model.Gaussian_head.num_cameras = int(cfg.Dataset.num_cameras)
    cfg.Model.Gaussian_head.bilateral_grid_chroma_ref_cameras = list(cfg.Inference.reference_cameras)
    return cfg


def check_checkpoint(path, cfg):
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as weights:
        keys = set(weights.keys())
        for prefix in ("encoder.", "gaussian_head.", "gaussian_head.bilateral_grid_transformer."):
            if not any(k.startswith(prefix) for k in keys):
                raise ValueError(f"Checkpoint is missing {prefix} weights")
        residual = "gaussian_head.bilateral_grid_transformer.residual_proj.weight" in keys
        if residual != (cfg.Model.Gaussian_head.bilateral_grid_chroma_kv_mode == "residual"):
            raise ValueError("Checkpoint color-correction mode does not match the dataset config")
    logging.info("Checkpoint metadata verified: %s", path)


def load_model(checkpoint, cfg, device):
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    from model.gaussian_head import GuassianHead
    from pi3.models.pi3 import Pi3

    model = Pi3()
    model.gaussian_head = GuassianHead(dim_in=2048, cfg=cfg.Model.Gaussian_head)
    model.gaussian_head.image_backbone.mask_token = None
    width, height = cfg.Dataset.get("render_resolution", cfg.Dataset.resolution)
    model.gaussian_head.renderer.resolution = [int(height), int(width)]
    weights = load_file(str(checkpoint), device="cpu")
    # Match the DINO positional-grid interpolation used for the released checkpoints.
    for key, target in model.state_dict().items():
        source = weights.get(key)
        if source is None or source.shape == target.shape or "pos_embed" not in key:
            continue
        if source.ndim != 3 or target.ndim != 3 or source.shape[0] != 1 or target.shape[0] != 1 or source.shape[2] != target.shape[2]:
            raise ValueError(f"Incompatible positional embedding: {key}")
        source_side, target_side = int((source.shape[1] - 1) ** 0.5), int((target.shape[1] - 1) ** 0.5)
        if source_side ** 2 != source.shape[1] - 1 or target_side ** 2 != target.shape[1] - 1:
            raise ValueError(f"Non-square positional embedding: {key}")
        patches = source[:, 1:].reshape(1, source_side, source_side, source.shape[-1]).permute(0, 3, 1, 2)
        patches = F.interpolate(patches.float(), size=(target_side, target_side), mode="bicubic", align_corners=False)
        patches = patches.permute(0, 2, 3, 1).reshape(1, target_side ** 2, source.shape[-1])
        weights[key] = torch.cat((source[:, :1].float(), patches), dim=1).to(target.dtype)
        logging.info("Resized positional embedding: %s", key)
    model.load_state_dict(weights, strict=True)
    del weights
    return model.eval().requires_grad_(False).to(device)


def to_device(value, device):
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value


def render_batch(model, batch, cfg, device, amp_dtype):
    import torch
    import torch.nn.functional as F
    from model.exposure_affine import _prepare_canonical_input_for_backbone, apply_predicted_bilateral_grids

    batch = to_device(batch, device)
    images = batch["images"]
    inputs, outputs = batch["input_dict_gs"], batch["output_dict_gs"]
    for key in ("sky_mask", "single_depthmaps", "intrinsics", "camera2lidar", "dynamics_region"):
        inputs[key] = batch[key]
    opts = cfg.Inference
    images, _, _ = _prepare_canonical_input_for_backbone(
        images, images, inputs, outputs,
        device=device, is_main_process=True, iteration=1, logger=logging,
        use_canonical_input=True, use_lidar_affine_canonical=True,
        use_pi3_depth_for_affine=False, gaussian_only_canonical_mem_opt=False,
        lidar_affine_blend_ref_cameras=tuple(opts.reference_cameras),
        lidar_affine_exclude_dynamic=False, lidar_affine_min_points=int(opts.min_points),
        lidar_affine_use_rgb_refit=True,
        lidar_affine_rgb_residual_thresh=float(opts.rgb_residual_threshold),
        lidar_affine_rgb_refit_min_points=int(opts.rgb_refit_min_points),
        sky_depth_value=float(cfg.Model.Gaussian_head.sky_mask),
    )
    with torch.autocast("cuda", dtype=amp_dtype):
        prediction = model(images)
    with torch.autocast("cuda", enabled=False):
        inputs["sky_mask"] = batch["sky_mask"].to(images.dtype)
        head_outputs = dict(outputs)
        head_outputs["rgb"] = outputs["rgb"].to(images.dtype)
        mask = outputs["input_valid_mask"].float()
        if mask.shape[-2:] != images.shape[-2:]:
            mask = F.interpolate(mask, size=images.shape[-2:], mode="nearest")
        head_outputs["valid_mask"] = mask > 0.5
        prediction["scene"], prediction["frame"] = batch["scene"], batch["frame"]
        prediction["lidar2world"] = batch["camera_pose"] @ batch["camera2lidar"].inverse()
        rendered, _ = model.gaussian_head(prediction, images, model.patch_start_idx,
                                         input_dict_gs=inputs, output_dict_gs=head_outputs)
    rgb = rendered["image"]
    camera_ids = outputs["view_cam_ids"].reshape(-1)
    if rgb.shape[0] != int(cfg.Dataset.num_cameras):
        raise RuntimeError("Rendered camera count does not match the dataset")
    grids = rendered.get("predicted_bilateral_grids")
    if not isinstance(grids, dict) or not grids.get("grids"):
        raise RuntimeError("The checkpoint did not produce color-correction grids")
    apply_mask = torch.ones_like(camera_ids, dtype=torch.bool)
    for reference in opts.reference_cameras:
        apply_mask &= camera_ids != int(reference)
    corrected = rgb.clone()
    corrected[apply_mask] = apply_predicted_bilateral_grids(
        [grid[apply_mask] for grid in grids["grids"]], rgb[apply_mask],
        grids["grid_sizes"], grids["guidance_factors"],
    )
    return corrected, camera_ids.tolist()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    import torch
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm
    from dataset.driving import DrivingDataset

    cfg = load_config(args.dataset)
    check_checkpoint(args.checkpoint, cfg)
    dataset = DrivingDataset(args.data_path, cfg.Dataset, scene_ids=args.scene_ids)
    if args.max_frames is not None:
        dataset = Subset(dataset, range(min(args.max_frames, len(dataset))))
    if args.check_only:
        sample = dataset[0]
        logging.info("Input verified: %s/%s; images=%s; render RGB=%s; selected frames=%d",
                     sample["scene"], sample["frame"], tuple(sample["images"].shape),
                     tuple(sample["output_dict_gs"]["rgb"].shape), len(dataset))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Inference requires a compatible GPU, driver and compiled extensions. Use --check-only to validate data on CPU.")
    from utils.inference import create_current_scene_video, save_current_camera_mosaic

    device = torch.device("cuda")
    torch.manual_seed(42)
    amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = load_model(args.checkpoint, cfg, device)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                        pin_memory=True, persistent_workers=args.num_workers > 0)
    rows = [[3, 1, 0, 2, 4]] if args.dataset == "waymo" else [[1, 0, 2], [3, 5, 4]]
    suffix = "5cam" if args.dataset == "waymo" else "6cam_2x3"
    work_dir = args.output_dir / "render_current"
    current_scene = None
    with torch.no_grad():
        for batch in tqdm(loader, desc="Rendering", unit="frame"):
            scene, frame = batch["scene"][0], batch["frame"][0]
            if current_scene is not None and scene != current_scene and not args.no_video:
                create_current_scene_video(str(work_dir), current_scene, suffix, fps=args.fps)
            current_scene = scene
            path = work_dir / "combined" / scene / f"{frame}_{suffix}.png"
            if args.resume and path.is_file():
                continue
            rgb, camera_ids = render_batch(model, batch, cfg, device, amp_dtype)
            path.parent.mkdir(parents=True, exist_ok=True)
            save_current_camera_mosaic(rgb, camera_ids, rows, str(path))
    if current_scene is not None and not args.no_video:
        create_current_scene_video(str(work_dir), current_scene, suffix, fps=args.fps)
    logging.info("Inference complete: %s", work_dir)


if __name__ == "__main__":
    main()
