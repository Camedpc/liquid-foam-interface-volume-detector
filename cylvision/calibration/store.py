"""Calibration container, cylinder presets and JSON persistence.

A *calibration* ties one video (or image) to one graduated cylinder:

* the crop band ``[x_left, x_right]`` (full-height vertical band that
  contains the cylinder) and the gradient mask rows ``y_top`` / ``y_bottom``
  (glass rim and base, whose strong edges must be ignored by the detector);
* the clicked graduations ``(V_mL, y_px, x_px)`` and the ones the user
  skipped because they were out of frame;
* the fitted model parameters (``poly2`` coefficients, Möbius parameters
  and covariance), the per-model residuals and the recommended model;
* optional *back-face* clicks: the same graduations seen through the glass
  on the far side of the cylinder, used by the focal-free curvature model.

The JSON layout keeps every key of the historical ``calib.json`` files
(``roi_x``, ``graduations_*``, ``poly2_coef``, ``mobius_popt``,
``mobius_pcov``, ``residuals_ml``, ``recommended_model``,
``graduations_px_y_back``, ``graduations_px_x_back``,
``back_calibration_frame``) and adds ``x_left``/``x_right``/``y_top``/
``y_bottom`` and a ``cylinder`` block. Old files, where the row bounds lived
in a sibling ``crop.json``, are read transparently by :meth:`Calibration.from_json`
and :func:`load_legacy_run`.

Image conventions: ``y`` grows downward, so volume increases as ``y``
decreases; a row inside the crop is the same row as in the full frame.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from cylvision.calibration.models import (
    ModelName,
    best_model,
    evaluate_residuals,
    fit_calibration_models,
)


# ---------------------------------------------------------------------------
# Cylinder presets
# ---------------------------------------------------------------------------

@dataclass
class CylinderSpec:
    """Description of a graduated cylinder.

    ``graduations_ml`` are the marks the user is asked to click;
    ``medium_step_ml`` / ``thin_step_ml`` reproduce the visual hierarchy of
    the printed scale (used by the verification figure); ``radius_cm`` is
    the inner radius needed by the cylinder-curvature model (``None`` when
    unknown).
    """

    name: str
    capacity_ml: float
    graduations_ml: list[float]
    radius_cm: float | None = None
    thin_step_ml: float = 1.0
    medium_step_ml: float = 5.0

    @property
    def key(self) -> str:
        """Preset key (``"1000"`` for a 1000 mL cylinder)."""
        return str(int(round(self.capacity_ml)))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["graduations_ml"] = [float(v) for v in self.graduations_ml]
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CylinderSpec":
        return cls(
            name=str(d["name"]),
            capacity_ml=float(d["capacity_ml"]),
            graduations_ml=[float(v) for v in d["graduations_ml"]],
            radius_cm=None if d.get("radius_cm") is None else float(d["radius_cm"]),
            thin_step_ml=float(d.get("thin_step_ml", 1.0)),
            medium_step_ml=float(d.get("medium_step_ml", 5.0)),
        )


PRESETS: dict[str, CylinderSpec] = {
    "50": CylinderSpec(
        name="50 mL",
        capacity_ml=50.0,
        graduations_ml=[5.0, 10.0, 20.0, 30.0, 40.0, 50.0],
        radius_cm=1.35,
        thin_step_ml=1.0,
        medium_step_ml=5.0,
    ),
    "1000": CylinderSpec(
        name="1000 mL",
        capacity_ml=1000.0,
        graduations_ml=[float(v) for v in range(100, 1001, 100)],
        radius_cm=3.20,
        thin_step_ml=10.0,
        medium_step_ml=50.0,
    ),
}


def spec_from_graduations(graduations_ml: list[float],
                          name: str | None = None) -> CylinderSpec:
    """Build a spec for a cylinder that has no preset (steps guessed)."""
    grads = sorted(float(v) for v in graduations_ml)
    capacity = grads[-1] if grads else 0.0
    step = (grads[1] - grads[0]) if len(grads) >= 2 else max(capacity, 1.0)
    return CylinderSpec(
        name=name or f"{capacity:g} mL",
        capacity_ml=capacity,
        graduations_ml=grads,
        radius_cm=None,
        thin_step_ml=step / 10.0,
        medium_step_ml=step / 2.0,
    )


def _spec_from_legacy(d: Mapping[str, Any]) -> CylinderSpec:
    key = str(d.get("eprouvette", "")).strip()
    if key in PRESETS:
        return replace(PRESETS[key])
    grads = d.get("graduations_expected_ml") or d.get("graduations_ml") or []
    spec = spec_from_graduations([float(v) for v in grads], name=d.get("eprouvette_label"))
    if "thin_step_ml" in d:
        spec.thin_step_ml = float(d["thin_step_ml"])
    if "medium_step_ml" in d:
        spec.medium_step_ml = float(d["medium_step_ml"])
    return spec


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _opt_list(values: Any) -> list[float | None] | None:
    if values is None:
        return None
    return [None if v is None else float(v) for v in values]


@dataclass
class Calibration:
    """Everything needed to turn a detected row into a volume."""

    image_ref: str
    image_size: tuple[int, int]            # (W, H)
    cylinder: CylinderSpec
    x_left: int
    x_right: int
    y_top: int
    y_bottom: int
    graduations_ml: list[float]
    graduations_px_y: list[float]
    graduations_px_x: list[float]
    skipped_ml: list[float]
    poly2_coef: list[float]
    mobius_popt: list[float]
    mobius_pcov: list[list[float]]
    residuals_ml: dict[str, list[float]]
    recommended_model: ModelName
    back_px_y: list[float | None] | None = None
    back_px_x: list[float | None] | None = None
    back_frame_idx: int | None = None
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    # -- geometry helpers ---------------------------------------------------
    def crop(self) -> tuple[int, int, int, int]:
        """``(x_left, x_right, y_top, y_bottom)``."""
        return int(self.x_left), int(self.x_right), int(self.y_top), int(self.y_bottom)

    def crop_width(self) -> int:
        """Width of the crop band ``frame[:, x_left:x_right+1]``."""
        return int(self.x_right) - int(self.x_left) + 1

    def cx(self) -> int:
        """Cylinder axis column *inside the crop* (``(x_right - x_left) // 2``)."""
        return (int(self.x_right) - int(self.x_left)) // 2

    def has_back_clicks(self, min_points: int = 2) -> bool:
        """True when at least ``min_points`` back-face graduations were clicked."""
        if not self.back_px_y:
            return False
        return sum(v is not None for v in self.back_px_y) >= min_points

    @property
    def width(self) -> int:
        return int(self.image_size[0])

    @property
    def height(self) -> int:
        return int(self.image_size[1])

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict (new keys + legacy aliases)."""
        d: dict[str, Any] = {
            "image_ref": self.image_ref,
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "cylinder": self.cylinder.to_dict(),
            "x_left": int(self.x_left),
            "x_right": int(self.x_right),
            "y_top": int(self.y_top),
            "y_bottom": int(self.y_bottom),
            "graduations_ml": [float(v) for v in self.graduations_ml],
            "graduations_px_y": [float(v) for v in self.graduations_px_y],
            "graduations_px_x": [float(v) for v in self.graduations_px_x],
            "skipped_ml": [float(v) for v in self.skipped_ml],
            "poly2_coef": [float(v) for v in self.poly2_coef],
            "mobius_popt": [float(v) for v in self.mobius_popt],
            "mobius_pcov": [[float(v) for v in row] for row in self.mobius_pcov],
            "residuals_ml": {k: [float(v) for v in vals] for k, vals in self.residuals_ml.items()},
            "recommended_model": self.recommended_model,
            # legacy aliases, kept so older tools still read the file
            "roi_x": [int(self.x_left), int(self.x_right)],
            "eprouvette": self.cylinder.key,
            "graduations_expected_ml": [float(v) for v in self.cylinder.graduations_ml],
            "graduations_skipped_ml": [float(v) for v in self.skipped_ml],
            "thin_step_ml": float(self.cylinder.thin_step_ml),
            "medium_step_ml": float(self.cylinder.medium_step_ml),
        }
        if self.back_px_y is not None:
            d["graduations_px_y_back"] = _opt_list(self.back_px_y)
            d["graduations_px_x_back"] = _opt_list(self.back_px_x)
            d["back_calibration_frame"] = self.back_frame_idx
        return d

    def to_json(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                        encoding="utf-8")

    @staticmethod
    def is_legacy_dict(d: Mapping[str, Any]) -> bool:
        """A legacy ``calib.json`` has ``roi_x`` but no ``x_left``."""
        return "x_left" not in d and "roi_x" in d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any],
                  crop: Mapping[str, Any] | None = None) -> "Calibration":
        """Build from a dict in either the new or the legacy layout.

        ``crop`` is the optional content of a legacy ``crop.json`` providing
        ``y_top`` / ``y_bottom`` (and ``x_left`` / ``x_right``, which win
        over ``roi_x`` when both exist).
        """
        if cls.is_legacy_dict(d):
            return cls.from_legacy_dict(d, crop)
        spec = CylinderSpec.from_dict(d["cylinder"]) if "cylinder" in d else _spec_from_legacy(d)
        W, H = int(d["image_size"][0]), int(d["image_size"][1])
        known = {
            "image_ref", "image_size", "cylinder", "x_left", "x_right", "y_top", "y_bottom",
            "graduations_ml", "graduations_px_y", "graduations_px_x", "skipped_ml",
            "poly2_coef", "mobius_popt", "mobius_pcov", "residuals_ml", "recommended_model",
            "roi_x", "eprouvette", "graduations_expected_ml", "graduations_skipped_ml",
            "thin_step_ml", "medium_step_ml", "eprouvette_label",
            "graduations_px_y_back", "graduations_px_x_back", "back_calibration_frame",
        }
        return cls(
            image_ref=str(d.get("image_ref", "")),
            image_size=(W, H),
            cylinder=spec,
            x_left=int(d["x_left"]),
            x_right=int(d["x_right"]),
            y_top=int(d.get("y_top", 0)),
            y_bottom=int(d.get("y_bottom", H - 1)),
            graduations_ml=[float(v) for v in d["graduations_ml"]],
            graduations_px_y=[float(v) for v in d["graduations_px_y"]],
            graduations_px_x=[float(v) for v in d.get("graduations_px_x", [])],
            skipped_ml=[float(v) for v in d.get("skipped_ml", d.get("graduations_skipped_ml", []))],
            poly2_coef=[float(v) for v in d["poly2_coef"]],
            mobius_popt=[float(v) for v in d["mobius_popt"]],
            mobius_pcov=[[float(v) for v in row] for row in d["mobius_pcov"]],
            residuals_ml={k: [float(v) for v in vals] for k, vals in d["residuals_ml"].items()},
            recommended_model=d.get("recommended_model", "poly2"),
            back_px_y=_opt_list(d.get("graduations_px_y_back")),
            back_px_x=_opt_list(d.get("graduations_px_x_back")),
            back_frame_idx=None if d.get("back_calibration_frame") is None
            else int(d["back_calibration_frame"]),
            extra={k: v for k, v in d.items() if k not in known},
        )

    @classmethod
    def from_legacy_dict(cls, d: Mapping[str, Any],
                         crop: Mapping[str, Any] | None = None) -> "Calibration":
        """Read a historical ``calib.json`` (+ optional ``crop.json`` content)."""
        crop = crop or {}
        W, H = int(d["image_size"][0]), int(d["image_size"][1])
        if "x_left" in crop and "x_right" in crop:
            x_left, x_right = int(crop["x_left"]), int(crop["x_right"])
        else:
            x_left, x_right = int(d["roi_x"][0]), int(d["roi_x"][1])
        y_top = int(crop.get("y_top", 0))
        y_bottom = int(crop.get("y_bottom", H - 1))
        merged = dict(d)
        merged.update({"x_left": x_left, "x_right": x_right,
                       "y_top": y_top, "y_bottom": y_bottom})
        merged.pop("roi_x", None)  # not legacy any more: route through from_dict
        calib = cls.from_dict(merged)
        return calib

    @classmethod
    def from_json(cls, path: Path | str) -> "Calibration":
        """Load a calibration file.

        A legacy file is recognised by its keys; if a ``crop.json`` sits
        next to it, its ``y_top`` / ``y_bottom`` are used.
        """
        path = Path(path)
        d = json.loads(path.read_text(encoding="utf-8"))
        if cls.is_legacy_dict(d):
            sibling = path.with_name("crop.json")
            return load_legacy_run(path, sibling if sibling.exists() else None)
        return cls.from_dict(d)

    def with_back_clicks(self, back_px_y: list[float | None],
                         back_px_x: list[float | None] | None,
                         frame_idx: int | None) -> "Calibration":
        """Copy with the back-face clicks attached (one entry per graduation)."""
        if len(back_px_y) != len(self.graduations_ml):
            raise ValueError("one back click (or None) per clicked graduation is required")
        return replace(
            self,
            back_px_y=_opt_list(back_px_y),
            back_px_x=None if back_px_x is None else _opt_list(back_px_x),
            back_frame_idx=None if frame_idx is None else int(frame_idx),
        )


def load_legacy_run(calib_json: Path | str,
                    crop_json: Path | str | None = None) -> Calibration:
    """Load a legacy ``calib.json`` and, optionally, its ``crop.json``.

    Without ``crop.json`` the crop columns come from ``roi_x`` and the row
    bounds default to the full frame (``y_top = 0``, ``y_bottom = H - 1``).
    """
    d = json.loads(Path(calib_json).read_text(encoding="utf-8"))
    crop = None
    if crop_json is not None:
        crop = json.loads(Path(crop_json).read_text(encoding="utf-8"))
    return Calibration.from_legacy_dict(d, crop)


# ---------------------------------------------------------------------------
# Fitting from clicks
# ---------------------------------------------------------------------------

def fit_calibration(clicks: Mapping[str, Any], spec: CylinderSpec,
                    image_ref: str, image_size: tuple[int, int]) -> Calibration:
    """Fit the three models on a clicks dict and package the result.

    ``clicks`` is the dict returned by
    :func:`cylvision.calibration.clicks.collect_calibration_clicks`:
    ``x_left, x_right, y_top, y_bottom, V_ml, y_px, x_px, skipped_ml``.
    Pure function, no I/O.
    """
    V_ml = np.asarray(clicks["V_ml"], dtype=float)
    y_px = np.asarray(clicks["y_px"], dtype=float)
    x_px = np.asarray(clicks.get("x_px", np.zeros_like(y_px)), dtype=float)
    if V_ml.size < 3:
        raise ValueError(f"at least 3 clicked graduations are required, got {V_ml.size}")

    models = fit_calibration_models(y_px, V_ml)
    residuals = evaluate_residuals(models, y_px, V_ml)
    chosen = best_model(residuals)

    x_left, x_right = int(clicks["x_left"]), int(clicks["x_right"])
    if x_left > x_right:
        x_left, x_right = x_right, x_left
    W, H = int(image_size[0]), int(image_size[1])
    y_top = int(clicks.get("y_top", 0))
    y_bottom = int(clicks.get("y_bottom", H - 1))
    if y_top > y_bottom:
        y_top, y_bottom = y_bottom, y_top

    return Calibration(
        image_ref=str(image_ref),
        image_size=(W, H),
        cylinder=replace(spec),
        x_left=x_left,
        x_right=x_right,
        y_top=y_top,
        y_bottom=y_bottom,
        graduations_ml=V_ml.tolist(),
        graduations_px_y=y_px.tolist(),
        graduations_px_x=x_px.tolist(),
        skipped_ml=[float(v) for v in clicks.get("skipped_ml", [])],
        poly2_coef=models["poly2"]["coef"].tolist(),
        mobius_popt=models["mobius"]["popt"].tolist(),
        mobius_pcov=models["mobius"]["pcov"].tolist(),
        residuals_ml={k: v.tolist() for k, v in residuals.items()},
        recommended_model=chosen,
    )


def residual_summary(calib: Calibration) -> dict[str, dict[str, float]]:
    """``{model: {"max": ..., "rms": ...}}`` in mL, for console reports."""
    out: dict[str, dict[str, float]] = {}
    for name, res in calib.residuals_ml.items():
        r = np.asarray(res, dtype=float)
        out[name] = {
            "max": float(np.max(np.abs(r))) if r.size else 0.0,
            "rms": float(np.sqrt(np.mean(r ** 2))) if r.size else 0.0,
        }
    return out
