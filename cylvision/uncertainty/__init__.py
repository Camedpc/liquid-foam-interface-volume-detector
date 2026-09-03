"""Uncertainty budget of a volume reading: geometric (cylinder curvature),
empirical (per-column spread of the detection) and their combination with
the calibration and pixel terms."""
from __future__ import annotations

from .budget import (UncertaintyBudget, budget_markdown_table, build_budget, calibration_rms_ml,
                     default_scale_resolution_ml, render_budget_figure)
from cylvision.calibration.models import slope_ml_per_px

from .curvature import (CameraGeometry, FocalFreeFit, alpha_from_mobius, arc_in_band, band_half_angle,
                        curvature_delta_px, curvature_uncertainty_fn, curvature_uncertainty_ml,
                        fit_focal_free, geometry_from_calibration, invert_no_tilt, project_graduation,
                        render_curvature_figure, render_cylinder_schematic, volume_to_row_fn)
from .empirical import method_uncertainty, method_uncertainty_from_result, summarize_method_uncertainty

__all__ = [
    "CameraGeometry", "FocalFreeFit", "UncertaintyBudget",
    "alpha_from_mobius", "arc_in_band", "band_half_angle", "budget_markdown_table", "build_budget",
    "calibration_rms_ml", "curvature_delta_px", "curvature_uncertainty_fn", "curvature_uncertainty_ml",
    "default_scale_resolution_ml", "fit_focal_free", "geometry_from_calibration", "invert_no_tilt",
    "method_uncertainty", "method_uncertainty_from_result", "project_graduation",
    "render_budget_figure", "render_curvature_figure", "render_cylinder_schematic",
    "slope_ml_per_px", "summarize_method_uncertainty", "volume_to_row_fn",
]
