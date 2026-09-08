"""Optical absorbance of a back-lit foam: reference, Beer-Lambert map, foam top, liquid content, heat maps.

* :mod:`cylvision.absorbance.reference` -- ``Reference`` (empty-cylinder
  ``I0``), dead-pixel masks and fill, ``prepare_intensity``.
* :mod:`cylvision.absorbance.beer_lambert` -- ``AbsorbanceParams``,
  ``absorbance_map`` / ``absorbance_profile``, ``foam_top_from_profile``,
  ``mass_balance_gamma`` / ``liquid_content_profile``, ``analyse_crop``.
* :mod:`cylvision.absorbance.batch` -- ``process_video_absorbance``,
  ``finalize_mass_balance``, ``foam_tops_for_ks``, ``AbsorbanceRun``.
* :mod:`cylvision.absorbance.heatmap` -- ``plot_heatmap_A``,
  ``plot_heatmap_c``, ``plot_measurement_figure``.
"""
from __future__ import annotations

from cylvision.absorbance.batch import (
    ABS_CSV_COLUMNS,
    DEFAULT_KS,
    AbsorbanceRun,
    finalize_mass_balance,
    foam_tops_for_ks,
    foam_volume_vs_k,
    process_video_absorbance,
    px_per_cm_from_calibration,
    safe_volume,
    volume_window,
)
from cylvision.absorbance.beer_lambert import (
    D_INT_CM,
    RHO_BEER_G_ML,
    S_CM2,
    AbsorbanceParams,
    AbsorbanceResult,
    PreparedReference,
    absorbance_map,
    absorbance_profile,
    analyse_crop,
    foam_integral_cm,
    foam_top_from_profile,
    liquid_content_profile,
    mass_balance_gamma,
    prepare_reference,
)
from cylvision.absorbance.heatmap import (
    plot_heatmap_A,
    plot_heatmap_c,
    plot_measurement_figure,
    time_edges,
    volume_ticks,
)
from cylvision.absorbance.reference import (
    DEFAULT_N_REF_FRAMES,
    Reference,
    channel_intensity,
    dead_pixel_masks,
    fill_dead_pixels,
    prepare_intensity,
    reference_from_video,
    stack_reference,
)

__all__ = [
    "ABS_CSV_COLUMNS", "DEFAULT_KS", "AbsorbanceRun", "finalize_mass_balance", "foam_tops_for_ks",
    "foam_volume_vs_k", "process_video_absorbance", "px_per_cm_from_calibration", "safe_volume",
    "volume_window",
    "D_INT_CM", "RHO_BEER_G_ML", "S_CM2", "AbsorbanceParams", "AbsorbanceResult", "PreparedReference",
    "absorbance_map", "absorbance_profile", "analyse_crop", "foam_integral_cm", "foam_top_from_profile",
    "liquid_content_profile", "mass_balance_gamma", "prepare_reference",
    "plot_heatmap_A", "plot_heatmap_c", "plot_measurement_figure", "time_edges", "volume_ticks",
    "DEFAULT_N_REF_FRAMES", "Reference", "channel_intensity", "dead_pixel_masks", "fill_dead_pixels",
    "prepare_intensity", "reference_from_video", "stack_reference",
]
