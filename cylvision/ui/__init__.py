"""User interface: dark theme, Tk control panel, frame picker, live tuner.

Only :mod:`cylvision.ui.theme` is imported eagerly (it has no window
dependency); the panel, picker and tuner are imported lazily through the
module attributes below so that ``import cylvision.ui`` stays cheap and
headless-safe.
"""
from __future__ import annotations

from typing import Any

from cylvision.ui import theme

__all__ = ["theme", "ControlsPanel", "pick_frame", "compose_canvas", "run_tuner", "render_canvas"]

_LAZY: dict[str, tuple[str, str]] = {
    "ControlsPanel": ("cylvision.ui.controls_panel", "ControlsPanel"),
    "pick_frame": ("cylvision.ui.frame_picker", "pick_frame"),
    "compose_canvas": ("cylvision.ui.tuner", "compose_canvas"),
    "render_canvas": ("cylvision.ui.tuner", "render_canvas"),
    "run_tuner": ("cylvision.ui.tuner", "run_tuner"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        module, attr = _LAZY[name]
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(f"module 'cylvision.ui' has no attribute {name!r}")
