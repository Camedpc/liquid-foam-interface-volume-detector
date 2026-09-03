"""Interface detection: vertical gradient, two-pass detector, rendering.

Public names re-exported from :mod:`gradient`, :mod:`interfaces` and
:mod:`panels`.
"""
from __future__ import annotations

from .gradient import (
    CHANNELS,
    Channel,
    apply_bottom_mask,
    apply_top_mask,
    compute_gradient,
    to_gray,
)
from .interfaces import (
    POLARITIES,
    DetectionParams,
    InterfaceResult,
    Polarity,
    band_columns,
    detect_from_crop,
    detect_interfaces,
    polarity_sign,
)
from .panels import (
    annotate_interfaces,
    dashed_hline,
    draw_brace_right,
    draw_inward_arrow,
    make_gradient_panel,
    make_highlight_panel,
    make_info_strip,
    make_label_strip,
    make_panels,
    make_specimen_panel,
    make_threshold_panel,
)

__all__ = [
    "CHANNELS", "Channel", "apply_bottom_mask", "apply_top_mask", "compute_gradient", "to_gray",
    "POLARITIES", "DetectionParams", "InterfaceResult", "Polarity", "band_columns",
    "detect_from_crop", "detect_interfaces", "polarity_sign",
    "annotate_interfaces", "dashed_hline", "draw_brace_right", "draw_inward_arrow",
    "make_gradient_panel", "make_highlight_panel", "make_info_strip", "make_label_strip",
    "make_panels", "make_specimen_panel", "make_threshold_panel",
]
