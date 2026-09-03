"""Verification figure for a calibration.

The figure lets the user check, at a glance, that the clicked graduations
and the fitted model are consistent with the physical scale:

1. **Annotated frame** -- the original frame with the crop bounds and one
   horizontal line per graduation of the printed scale, all positioned by
   *inverting the recommended model* ``V(y)``: the clicked marks in green,
   the intermediate ``medium_step`` marks (not clicked) in orange, and the
   ``thin_step`` marks in grey. If the fit is right, every predicted line
   sits on a real mark of the scale, including the marks that were never
   clicked.
2. **Crop band** -- the same lines on the crop alone (full width).
3. **V(y)** -- the clicks as dots and the three fitted curves (PCHIP,
   poly2, Möbius), plus a bar chart of the residual of each model at every
   click. Sub-millilitre residuals mean the click noise dominates; a
   systematic pattern means the wrong model or a mis-click.

Only the Agg canvas of matplotlib is used (no window, no global backend).
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from cylvision.calibration.models import (
    MODEL_NAMES,
    ModelFn,
    build_model_fn,
    evaluate_residuals,
    fit_calibration_models,
    invert_model,
)
from cylvision.calibration.store import Calibration

# RGB colours (the annotated images are converted to RGB for matplotlib)
COLOR_THIN = (190, 190, 190)     # grey: thin marks
COLOR_MEDIUM = (255, 140, 0)     # orange: medium marks, not clicked
COLOR_CLICK = (0, 200, 0)        # green: clicked marks
COLOR_EDGE = (220, 50, 50)       # red: crop columns
COLOR_ROWS = (0, 200, 220)       # cyan: y_top / y_bottom


def predicted_graduation_rows(calib: Calibration, model_fn: ModelFn | None = None
                              ) -> tuple[dict[float, float], dict[float, float]]:
    """Rows of every non-clicked mark, from the recommended model.

    Returns ``(y_medium, y_thin)`` as ``{V_mL: y_px}`` dicts covering the
    range between the lowest and the highest clicked graduation.
    """
    model_fn = model_fn or build_model_fn(calib)
    H = calib.height
    V_click = np.asarray(calib.graduations_ml, dtype=float)
    thin = float(calib.cylinder.thin_step_ml) or 1.0
    medium = float(calib.cylinder.medium_step_ml) or thin
    V_min, V_max = float(V_click.min()), float(V_click.max())
    n = int(round((V_max - V_min) / thin))
    V_all = V_min + thin * np.arange(n + 1)
    y_all = invert_model(model_fn, 0, H, V_all)

    clicked = {round(float(v), 6) for v in V_click}
    y_medium: dict[float, float] = {}
    y_thin: dict[float, float] = {}
    for V, y in zip(V_all, y_all):
        key = round(float(V), 6)
        if key in clicked:
            continue
        ratio = V / medium
        if abs(ratio - round(ratio)) < 1e-6:
            y_medium[key] = float(y)
        else:
            y_thin[key] = float(y)
    return y_medium, y_thin


def annotate_frame(img_bgr: np.ndarray, calib: Calibration,
                   model_fn: ModelFn | None = None, *, full_width: bool = False) -> np.ndarray:
    """Draw crop bounds and predicted graduation lines. Returns RGB.

    With ``full_width=False`` the lines span the crop columns only (frame
    view); with ``full_width=True`` they span the whole image (crop view).
    """
    H, W = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    y_medium, y_thin = predicted_graduation_rows(calib, model_fn)
    x0, x1 = (0, W) if full_width else (int(calib.x_left), int(calib.x_right))
    lx = 3 if full_width else int(calib.x_right) + 6

    if not full_width:
        for x_col in (int(calib.x_left), int(calib.x_right)):
            _halo_line(rgb, (x_col, 0), (x_col, H), COLOR_EDGE, 2)
    for y_row, tag in ((int(calib.y_top), "y_top"), (int(calib.y_bottom), "y_bottom")):
        _dashed_hline(rgb, y_row, x0, x1, COLOR_ROWS, 2)
        # y_top labelled below its line, y_bottom above it (both stay in the image)
        ty = y_row + 22 if tag == "y_top" else y_row - 8
        ty = min(max(ty, 16), H - 4)
        _halo_text(rgb, tag, (x0 + 4, ty), 0.6, COLOR_ROWS, 2)

    for _V, y in y_thin.items():
        yy = int(round(y))
        _halo_line(rgb, (x0, yy), (x1, yy), COLOR_THIN, 1)
    for V, y in y_medium.items():
        yy = int(round(y))
        _halo_line(rgb, (x0, yy), (x1, yy), COLOR_MEDIUM, 2)
        _halo_text(rgb, f"{V:g}", (lx, yy + 5 if not full_width else yy - 3),
                   0.55 if not full_width else 0.40, COLOR_MEDIUM, 2 if not full_width else 1)
    for V, y in zip(calib.graduations_ml, calib.graduations_px_y):
        yy = int(round(y))
        _halo_line(rgb, (x0, yy), (x1, yy), COLOR_CLICK, 2)
        _halo_text(rgb, f"{V:g} mL", (lx, yy + 5 if not full_width else yy - 4),
                   0.65 if not full_width else 0.45, COLOR_CLICK, 2 if not full_width else 1)
    return rgb


def _halo_line(img: np.ndarray, p0: tuple[int, int], p1: tuple[int, int],
               color: tuple[int, int, int], thickness: int) -> None:
    """Coloured line over a dark halo: readable on black and on green screens."""
    cv2.line(img, p0, p1, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)


def _halo_text(img: np.ndarray, text: str, org: tuple[int, int], scale: float,
               color: tuple[int, int, int], thickness: int) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _dashed_hline(img: np.ndarray, y: int, x0: int, x1: int,
                  color: tuple[int, int, int], thickness: int, dash: int = 12) -> None:
    for xs in range(x0, x1, 2 * dash):
        cv2.line(img, (xs, y), (min(xs + dash, x1), y), color, thickness)


def render_verification_figure(img_bgr: np.ndarray, calib: Calibration,
                               out_png: Path | str) -> None:
    """Save the three-panel verification figure to ``out_png``."""
    H, W = img_bgr.shape[:2]
    y_px = np.asarray(calib.graduations_px_y, dtype=float)
    V_ml = np.asarray(calib.graduations_ml, dtype=float)

    models = fit_calibration_models(y_px, V_ml)
    residuals: dict[str, Any] = {k: np.asarray(v, dtype=float)
                                 for k, v in calib.residuals_ml.items()}
    if not all(name in residuals and residuals[name].size == V_ml.size for name in MODEL_NAMES):
        residuals = evaluate_residuals(models, y_px, V_ml)
    model_fn = models[calib.recommended_model]["fn"]

    orig = annotate_frame(img_bgr, calib, model_fn)
    x_min = max(0, int(calib.x_left) - 5)
    x_max = min(W, int(calib.x_right) + 6)
    crop = annotate_frame(img_bgr[:, x_min:x_max], _shifted(calib, x_min), model_fn,
                          full_width=True)

    thin = calib.cylinder.thin_step_ml
    medium = calib.cylinder.medium_step_ml

    fig = Figure(figsize=(16, 8))
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 0.6, 1.4], height_ratios=[3, 1],
                          hspace=0.3, wspace=0.2)

    ax_orig = fig.add_subplot(gs[:, 0])
    ax_orig.imshow(orig)
    ax_orig.set_title(
        f"Annotated frame ({W} x {H} px)\n"
        f"green: clicks  |  orange: {medium:g} mL marks  |  grey: every {thin:g} mL\n"
        f"lines predicted by the '{calib.recommended_model}' model",
        fontsize=10)
    ax_orig.axis("off")

    ax_crop = fig.add_subplot(gs[:, 1])
    ax_crop.imshow(crop)
    ax_crop.set_title(f"Crop band\n{crop.shape[1]} x {crop.shape[0]} px", fontsize=10)
    ax_crop.axis("off")

    ax_fit = fig.add_subplot(gs[0, 2])
    yq = np.linspace(y_px.min() - 15, y_px.max() + 15, 600)
    ax_fit.plot(y_px, V_ml, "ko", ms=7, label="clicks", zorder=5)
    ax_fit.plot(yq, models["pchip"]["fn"](yq), "-", lw=1.6, label="PCHIP")
    ax_fit.plot(yq, models["poly2"]["fn"](yq), "--", lw=1.6, label="poly2")
    ax_fit.plot(yq, models["mobius"]["fn"](yq), ":", lw=1.8, label="Möbius")
    ax_fit.invert_xaxis()
    ax_fit.set_xlabel("y (px)")
    ax_fit.set_ylabel("V (mL)")
    ax_fit.grid(True, alpha=0.3)
    ax_fit.legend(loc="best")
    ax_fit.set_title(f"V(y) - model comparison ({calib.cylinder.name}, "
                     f"recommended: {calib.recommended_model})", fontsize=10)

    ax_res = fig.add_subplot(gs[1, 2])
    width = 0.25
    xs = np.arange(V_ml.size)
    for i, name in enumerate(MODEL_NAMES):
        r = residuals[name]
        rms = float(np.sqrt(np.mean(r ** 2))) if r.size else 0.0
        ax_res.bar(xs + (i - 1) * width, r, width, label=f"{name} (RMS {rms:.2f} mL)")
    ax_res.axhline(0, color="k", lw=0.6)
    ax_res.set_xticks(xs)
    ax_res.set_xticklabels([f"{v:g}" for v in V_ml], fontsize=8)
    ax_res.set_xlabel("graduation (mL)")
    ax_res.set_ylabel("residual (mL)")
    ax_res.grid(True, axis="y", alpha=0.3)
    ax_res.legend(loc="best", fontsize=8)
    ax_res.set_title("Residuals at the calibration points", fontsize=10)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=130, bbox_inches="tight")


def _shifted(calib: Calibration, dx: int) -> Calibration:
    """Copy of ``calib`` whose x coordinates are expressed inside a sub-image."""
    return replace(
        calib,
        x_left=int(calib.x_left) - dx,
        x_right=int(calib.x_right) - dx,
        graduations_px_x=[float(x) - dx for x in calib.graduations_px_x],
    )
