"""Run directory layout: where a processed video keeps its files.

One video = one run directory, ``<root>/<video_stem>/`` by default::

    <run_dir>/
    ├── calib.json              calibration (crop band, clicked graduations, models)
    ├── calibration_check.png   verification figure written by ``scripts/calibrate.py``
    ├── params.json             DetectionParams + ``frame_step`` written by the tuner
    ├── levels.csv              per-frame rows written by the batch (see ``batch.CSV_COLUMNS``)
    ├── meta.json               batch metadata (video, frame range, step, timings, summaries)
    └── figure.png              V(t) figure

``params.json`` holds the detector parameters (``DetectionParams.to_dict``)
plus one extra key, ``frame_step`` (``K``: one frame in ``K`` is analysed
by the batch). Legacy ``params.json`` files (``T_pos`` / ``T_neg`` keys)
are accepted through ``DetectionParams.from_dict``; a legacy ``calib.json``
with a sibling ``crop.json`` is accepted through ``Calibration.from_json``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cylvision.calibration.store import Calibration
from cylvision.detection.interfaces import DetectionParams

CALIB_FILE = "calib.json"
PARAMS_FILE = "params.json"
META_FILE = "meta.json"
CSV_FILE = "levels.csv"
FIGURE_FILE = "figure.png"
CHECK_FILE = "calibration_check.png"

_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_stem(name: str) -> str:
    """File-system-safe version of a video stem (Windows-reserved chars removed)."""
    stem = Path(name).stem if "." in name else name
    stem = _UNSAFE.sub("_", stem).strip(" .")
    return stem or "run"


@dataclass
class RunDir:
    """Paths of one run directory. ``root`` is the directory itself."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    # -- paths ---------------------------------------------------------------
    @property
    def calib_path(self) -> Path:
        return self.root / CALIB_FILE

    @property
    def params_path(self) -> Path:
        return self.root / PARAMS_FILE

    @property
    def meta_path(self) -> Path:
        return self.root / META_FILE

    @property
    def csv_path(self) -> Path:
        return self.root / CSV_FILE

    @property
    def figure_path(self) -> Path:
        return self.root / FIGURE_FILE

    @property
    def check_png(self) -> Path:
        return self.root / CHECK_FILE

    def ensure(self) -> "RunDir":
        """Create the directory (and parents) if needed; returns ``self``."""
        self.root.mkdir(parents=True, exist_ok=True)
        return self

    def has_calibration(self) -> bool:
        return self.calib_path.exists()

    def has_params(self) -> bool:
        return self.params_path.exists()

    # -- params --------------------------------------------------------------
    def save_params(self, params: DetectionParams, frame_step: int = 1) -> Path:
        """Write ``params.json`` (``DetectionParams`` dict + ``frame_step``)."""
        self.ensure()
        d: dict[str, Any] = params.to_dict()
        d["frame_step"] = max(1, int(frame_step))
        self.params_path.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
        return self.params_path

    def load_params(self) -> tuple[DetectionParams, int]:
        """Read ``params.json`` -> ``(DetectionParams, frame_step)``.

        Legacy keys are translated; a missing ``frame_step`` defaults to 1.
        """
        d = json.loads(self.params_path.read_text(encoding="utf-8"))
        step = max(1, int(d.get("frame_step", 1)))
        return DetectionParams.from_dict(d), step

    # -- calibration ---------------------------------------------------------
    def load_calibration(self) -> Calibration:
        """Read ``calib.json`` (new layout, or legacy + sibling ``crop.json``)."""
        return Calibration.from_json(self.calib_path)

    def save_calibration(self, calib: Calibration) -> Path:
        self.ensure()
        calib.to_json(self.calib_path)
        return self.calib_path

    # -- meta ----------------------------------------------------------------
    def save_meta(self, meta: dict[str, Any]) -> Path:
        self.ensure()
        self.meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=_json_default),
                                  encoding="utf-8")
        return self.meta_path

    def load_meta(self) -> dict[str, Any]:
        return json.loads(self.meta_path.read_text(encoding="utf-8"))


def _json_default(obj: Any) -> Any:
    """Fallback serialiser for numpy scalars and paths."""
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"not json-serialisable: {type(obj).__name__}")


def find_run(root: Path | str, video_path: Path | str) -> RunDir:
    """``RunDir`` of ``video_path`` under ``root`` (``<root>/<safe video stem>/``).

    The directory is not created; call :meth:`RunDir.ensure` before writing.
    """
    return RunDir(Path(root) / safe_stem(Path(video_path).name))


__all__ = [
    "CALIB_FILE", "PARAMS_FILE", "META_FILE", "CSV_FILE", "FIGURE_FILE", "CHECK_FILE",
    "RunDir", "find_run", "safe_stem",
]
