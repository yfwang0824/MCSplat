"""Load separately installed Pi3 modules with file-specific upstream licenses."""

import importlib.util
import os
from pathlib import Path
import sys


def _load_pi3_module(relative_path, module_name):
    dependency_root = os.environ.get("MCSPLAT_PI3_ROOT")
    if not dependency_root:
        raise ImportError(
            "Set MCSPLAT_PI3_ROOT to the external Pi3 dependency directory. "
            "See 'External Pi3 dependencies' in README.md."
        )
    source = Path(dependency_root).expanduser() / relative_path
    if not source.is_file():
        raise ImportError(
            f"Pi3 dependency was not found at {source}. "
            "Follow the external-dependency installation steps in README.md."
        )
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load external Pi3 dependency: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


_position_encoding = _load_pi3_module(
    "pi3/models/layers/pos_embed.py", "_mcsplat_external_pi3_position_encoding"
)
RoPE2D = _position_encoding.RoPE2D
PositionGetter = _position_encoding.PositionGetter
CameraHead = _load_pi3_module(
    "pi3/models/layers/camera_head.py", "_mcsplat_external_pi3_camera_head"
).CameraHead

__all__ = ["RoPE2D", "PositionGetter", "CameraHead"]
