# MCSplat: Multi-View Photometric and Geometric Consistent Feed-Forward Gaussian Splatting for Driving Scenes

<p align="center">
  📑 <a href="https://arxiv.org/abs/2610.25345">Paper (arXiv)</a>
  &nbsp;&nbsp;|&nbsp;&nbsp;
  🌐 <a href="https://yfwang0824.github.io/mcsplat-project/">Project Page</a>
  &nbsp;&nbsp;|&nbsp;&nbsp;
  🤗 <a href="https://huggingface.co/Yifeifei/MCSplat/tree/main">Hugging Face</a>
  &nbsp;&nbsp;|&nbsp;&nbsp;
  📦 <a href="https://huggingface.co/datasets/Yifeifei/MCSplat-data">Example Data</a>
</p>

MCSplat is a feed-forward Gaussian splatting framework that reconstructs geometrically consistent driving scenes from sparse multi-view images by disentangling camera-dependent appearance from shared geometry and introducing uncertainty-aware geometric self-supervision.

<p align="center">
  <img src="assets/visualization2.png" alt="Qualitative comparison of ground truth, canonical prediction, UniSplat, and MCSplat" width="100%">
</p>

<p align="center">
  <img src="assets/crosslane-nvs.png" alt="Cross-lane novel view synthesis comparison of UniSplat and MCSplat with a 1.8 m leftward shift" width="100%">
</p>

## Contents

- [1. Conda environment](#1-conda-environment)
- [2. Download pretrained weights](#2-download-pretrained-weights)
- [3. Prepare the data](#3-prepare-the-data)
- [4. Waymo inference](#4-waymo-inference)
- [5. nuScenes inference](#5-nuscenes-inference)
- [6. Options and validation](#6-options-and-validation)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## 1. Conda environment

Recommended environment: **Linux, Python 3.10, PyTorch 2.4.0, and CUDA 12.1**, with a compatible GPU driver, `nvcc`, and GCC/G++ 11 or 12.

From the repository root:

```bash
git clone https://github.com/yfwang0824/MCSplat.git
cd MCSplat

conda env create -f environment.yml
conda activate mcsplat

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"

# Install matching PyTorch and attention wheels.
python -m pip install torch==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install xformers==0.0.27.post2 --no-deps \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install -r requirements.txt
python -m pip install 'spconv-cu120>=2.3,<2.4'

# Build/install simple-knn-v2
python -m pip install --no-build-isolation --no-deps ./submodules/simple-knn-v2

```

Check the environment:

```bash
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available()"
python -c "import torch, spconv.pytorch, xformers; from simple_knn_v2._C import distCUDACross; import gsplat; assert gsplat.__version__ == '1.5.3'; print('gsplat', gsplat.__version__)"
python -c "from pi3.models.layers.pos_embed import RoPE2D, PositionGetter; from pi3.models.layers.camera_head import CameraHead; print('Pi3 modules are available')"
```

Rendering uses the **official `gsplat==1.5.3`** package installed from PyPI. The released inference path renders synchronized camera views using the standard rasterization API; it does not require a separate local rasterizer checkout. If PyTorch or CUDA is changed, rebuild the extensions against the new environment. PyTorch's supported version combinations are documented [here](https://pytorch.org/get-started/previous-versions/), and spconv installation options are documented [here](https://github.com/traveller59/spconv#install).

## 2. Download pretrained weights

```bash
hf download Yifeifei/MCSplat \
  waymo/model.safetensors nuscenes/model.safetensors \
  --local-dir checkpoints
```

Expected paths:

```text
checkpoints/
├── waymo/model.safetensors
└── nuscenes/model.safetensors
```

Each checkpoint is approximately 4.58 GB and includes the Pi3 backbone, Gaussian head and bilateral-grid transformer.

## 3. Prepare the data

For data preprocessing code, please refer to [UniSplat](https://github.com/chenshi3/UniSplat).

Download the preprocessed example scenes from [Hugging Face](https://huggingface.co/datasets/Yifeifei/MCSplat-data): Waymo `008`, `040`, `046`, `070` and nuScenes `600`, `690`.

```bash
hf download Yifeifei/MCSplat-data --repo-type dataset \
  --local-dir data/archives

mkdir -p data/waymo/validation data/nuscenes/val
for scene in 008 040 046 070; do
  tar -xzf "data/archives/waymo/${scene}.tar.gz" -C data/waymo/validation
done
for scene in 600 690; do
  tar -xzf "data/archives/nuscenes/${scene}.tar.gz" -C data/nuscenes/val
done
```

Use `data/waymo/validation` or `data/nuscenes/val` as `--data-path` in the inference commands below. Each archive extracts into its scene directory.

Pass a root whose immediate subdirectories are the scenes to process. All scenes under that root are used unless `--scene-ids` is specified.

```text
DATA_ROOT/
└── 000/                             # scene directory ID
    ├── images/
    │   ├── 000000_0.png             # frame 000000, zero-based camera 0
    │   ├── 000000_1.png
    │   └── ...
    ├── 00000_1.npz                  # calibration; one-based camera 1
    ├── 00000_1.exr                  # sparse depth in meters
    ├── 00000_1_moge_mask.png         # nonzero = non-sky
    ├── ...
    └── dynamics/
        ├── dynamic_infos.json
        ├── dynamic_mask_0_1.npz      # integer frame ID, one-based camera
        └── ...
```

Camera conventions and released resolutions:

| Dataset | Camera IDs | Input size (width × height) | Render size |
| --- | --- | --- | --- |
| Waymo | `0=front`, `1=front left`, `2=front right`, `3=side left`, `4=side right` | 518 × 350 | 518 × 350 |
| nuScenes | `0=front`, `1=front left`, `2=front right`, `3=back left`, `4=back right`, `5=back` | 406 × 224 | 400 × 224 |

## 4. Waymo inference

```bash
python run_inference.py \
  --dataset waymo \
  --data-path data/waymo/validation \
  --output-dir outputs/waymo
```

Run selected scene directory IDs, or verify one frame first:

```bash
python run_inference.py \
  --dataset waymo \
  --data-path data/waymo/validation \
  --scene-ids 008,040 --resume

python run_inference.py \
  --dataset waymo \
  --data-path data/waymo/validation \
  --scene-ids 008 --max-frames 1 --no-video
```

Each output PNG is a horizontal mosaic in camera order `[3, 1, 0, 2, 4]`:

```text
outputs/waymo/render_current/combined/{scene}/{frame}_5cam.png
outputs/waymo/render_current/combined/{scene}/current_10fps.mp4
```

## 5. nuScenes inference

```bash
python run_inference.py \
  --dataset nuscenes \
  --data-path data/nuscenes/val \
  --output-dir outputs/nuscenes
```

Select scenes or run a single-frame check:

```bash
python run_inference.py \
  --dataset nuscenes \
  --data-path data/nuscenes/val \
  --scene-ids 600,690 --resume

python run_inference.py \
  --dataset nuscenes \
  --data-path data/nuscenes/val \
  --scene-ids 600 --max-frames 1 --no-video
```

Each output PNG is a 2×3 mosaic: top row `[1, 0, 2]`, bottom row `[3, 5, 4]`:

```text
outputs/nuscenes/render_current/combined/{scene}/{frame}_6cam_2x3.png
outputs/nuscenes/render_current/combined/{scene}/current_10fps.mp4
```

## 6. Options and validation

`--check-only` checks the checkpoint metadata and reads one selected frame, including its calibration and masks, without loading the network or requiring a GPU:

```bash
python run_inference.py \
  --dataset waymo \
  --data-path data/waymo/validation \
  --scene-ids 008 --check-only
```

| Option | Meaning |
| --- | --- |
| `--checkpoint PATH` | Override the dataset-specific weight path |
| `--scene-ids ID,ID` | Select actual scene directory IDs; numeric IDs accept leading-zero normalization |
| `--num-workers N` | DataLoader workers, default 4; use 0 for troubleshooting |
| `--max-frames N` | Limit the total number of selected frames |
| `--resume` | Skip existing PNGs and rebuild videos from saved scene PNGs |
| `--no-video` | Save PNGs only |
| `--fps N` | Video playback FPS, default 10; this does not resample input timestamps |
| `--output-dir PATH` | Output root, default `outputs/<dataset>` |

The inference path uses batch size 1 and runs the backbone in mixed precision, followed by the Gaussian head in float32. It preserves the checkpoint's reference cameras `[0, 3, 4]` during bilateral color correction. Outputs contain current-frame RGB renders; training, temporal evaluation and metric computation are outside this release.

## Citation

```bibtex
@article{wang2026mcsplat,
  title={MCSplat: Multi-View Photometric and Geometric Consistent Feed-Forward Gaussian Splatting for Driving Scenes},
  author={Wang, Yifei and Tian, Wei and Bao, Wenhua and Zhou, Jia and Wang, Hanshi and Ge, Fudong and Wu, Jianhua and Yuan, Shuai and Zhang, Zhipeng and Liu, Xingliang and others},
  journal={arXiv preprint arXiv:2610.25345},
  year={2026}
}
```

## Acknowledgement

We thank the authors of [UniSplat](https://github.com/chenshi3/UniSplat), [gsplat](https://github.com/nerfstudio-project/gsplat), [Pi3](https://github.com/yyfz/Pi3), [bilateral-driving](https://github.com/BigCiLeng/bilateral-driving), [MoGe](https://github.com/microsoft/MoGe), and [SAM 3](https://github.com/facebookresearch/sam3) for sharing their code and projects with the community.
