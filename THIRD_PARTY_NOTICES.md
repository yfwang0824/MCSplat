# Third-party notices

The root [Apache-2.0 license](LICENSE) applies to original MCSplat contributions.
It does not replace the licenses of the components below. Existing source
headers are retained. Local changes adapt these components for the released
checkpoints and inference pipeline; this is not an unmodified upstream release.

| Component / local scope | Upstream | License / retained notice |
| --- | --- | --- |
| Pi3 backbone in `pi3/`, except separately licensed portions listed below and file-specific notices | [Pi3](https://github.com/yyfz/Pi3) | [BSD-3-Clause](pi3/LICENSE), Copyright (c) 2025, the authors |
| DINOv2 transformer layers in `pi3/models/dinov2/`, `pi3/models/layers/{attention,block}.py` and `model/layers/` | [DINOv2](https://github.com/facebookresearch/dinov2) | Apache-2.0; Meta Platforms, Inc. and affiliates; original source headers retained |
| `position_grid_to_embed`, `make_sincos_pos_embed` and `create_uv_grid` in `model/gaussian_head/utils.py` | [VGGT head utilities](https://github.com/facebookresearch/vggt/blob/main/vggt/heads/utils.py) | [VGGT License and Acceptable Use Policy](licenses/VGGT.txt); Meta Platforms, Inc. and affiliates |
| Geometry/image helpers in `dataset/utils.py` and `LinearPts3d` in `pi3/models/layers/transformer_head.py` | [DUSt3R](https://github.com/naver/dust3r) | [Upstream notice](licenses/DUST3R.txt) and [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/); Copyright (c) 2024-present Naver Corporation. These adapted portions retain CC BY-NC-SA 4.0. |
| Residual and feature-fusion blocks in `model/gaussian_head/head_layers.py` | [DPT](https://github.com/isl-org/DPT) / MiDaS | [MIT](licenses/DPT.txt); Copyright (c) 2021 Intel ISL |
| Sparse-convolution and geometry helpers in `model/layers/spconv_unet.py` | [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) | Apache-2.0; OpenMMLab |
| Type-annotation module `dataset/typing.py` | [threestudio](https://github.com/threestudio-project/threestudio) | Apache-2.0; upstream attribution retained in the module |
| Official `gsplat==1.5.3` renderer, installed as an external dependency | [gsplat](https://github.com/nerfstudio-project/gsplat/tree/v1.5.3) | [Apache-2.0](https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/LICENSE); Nerfstudio Team. Its bundled dependencies retain their own upstream notices. |
| Modified nearest-neighbor CUDA extension in `submodules/simple-knn-v2/` | [simple-knn](https://gitlab.inria.fr/bkerbl/simple-knn) | [Gaussian-Splatting License](submodules/simple-knn-v2/LICENSE.md); Copyright (C) 2023 Inria, GRAPHDECO research group; research and evaluation terms apply |

## Included Pi3 modules

The following files are included at their original Pi3 package paths, with
their source attributions and file-specific notices retained.

- `pi3/models/layers/pos_embed.py`: the file declares CC BY-NC-SA 4.0 and credits
  Naver/CroCo. See the [pinned Pi3 source](https://github.com/yyfz/Pi3/blob/9fa3ddb3f8d53041f8b2738df404f62223bbaa7b/pi3/models/layers/pos_embed.py).
- `pi3/models/layers/camera_head.py`: the file attributes its residual block to
  [MAREPO](https://github.com/nianticlabs/marepo/blob/9a45e2bb07e5bb8cb997620088d352b439b13e0e/transformer/transformer.py).
  The [MAREPO license](https://github.com/nianticlabs/marepo/blob/9a45e2bb07e5bb8cb997620088d352b439b13e0e/LICENSE)
  includes noncommercial-use and redistribution restrictions. The Pi3 root BSD
  license does not override the original code's file-specific terms.
- Optional cuRoPE is distributed separately by [CroCo](https://github.com/naver/croco).
  Follow its upstream license and build instructions if installing it.

The root Apache-2.0 license does not grant unrestricted commercial use of the
complete dependency stack. Pretrained weights, input datasets and installed
Python/CUDA packages retain their own terms.
