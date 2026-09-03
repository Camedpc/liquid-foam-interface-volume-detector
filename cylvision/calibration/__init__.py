"""Calibration of a graduated cylinder: clicks -> models -> JSON -> check figure.

Public API (see the individual modules for the algorithms):

* :mod:`cylvision.calibration.models` -- PCHIP / poly2 / Möbius fits,
  ``best_model``, ``build_model_fn``, ``slope_ml_per_px``.
* :mod:`cylvision.calibration.store` -- ``CylinderSpec``, ``PRESETS``,
  ``Calibration`` (JSON IO, legacy files), ``fit_calibration``.
* :mod:`cylvision.calibration.clicks` -- interactive click collection.
* :mod:`cylvision.calibration.back_clicks` -- optional back-face clicks.
* :mod:`cylvision.calibration.check` -- verification figure.
"""
from __future__ import annotations

from cylvision.calibration.models import (
    MODEL_NAMES,
    ModelFn,
    ModelName,
    best_model,
    build_model_fn,
    evaluate_residuals,
    fit_calibration_models,
    invert_model,
    mobius_eval,
    pchip_fn,
    poly2_eval,
    rms,
    slope_ml_per_px,
)
from cylvision.calibration.store import (
    PRESETS,
    Calibration,
    CylinderSpec,
    fit_calibration,
    load_legacy_run,
    residual_summary,
    spec_from_graduations,
)
from cylvision.calibration.clicks import (
    ARROW_LOCK_MS,
    SHIFT_DAMPEN,
    build_click_prompts,
    clicks_from_points,
    collect_calibration_clicks,
    render_click_overlay,
)
from cylvision.calibration.back_clicks import (
    apply_back_clicks,
    back_fit_preview,
    back_points_to_lists,
    build_back_prompts,
    collect_back_clicks,
    render_back_overlay,
)
from cylvision.calibration.check import (
    annotate_frame,
    predicted_graduation_rows,
    render_verification_figure,
)

__all__ = [
    "MODEL_NAMES", "ModelFn", "ModelName", "best_model", "build_model_fn",
    "evaluate_residuals", "fit_calibration_models", "invert_model", "mobius_eval",
    "pchip_fn", "poly2_eval", "rms", "slope_ml_per_px",
    "PRESETS", "Calibration", "CylinderSpec", "fit_calibration", "load_legacy_run",
    "residual_summary", "spec_from_graduations",
    "ARROW_LOCK_MS", "SHIFT_DAMPEN", "build_click_prompts", "clicks_from_points",
    "collect_calibration_clicks", "render_click_overlay",
    "apply_back_clicks", "back_fit_preview", "back_points_to_lists", "build_back_prompts",
    "collect_back_clicks", "render_back_overlay",
    "annotate_frame", "predicted_graduation_rows", "render_verification_figure",
]
