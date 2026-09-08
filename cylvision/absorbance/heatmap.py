"""Figures of the absorbance analysis: ``A(t, V)``, ``c(t, V)`` and the measurement panel.

Rendered with the ``Agg`` backend through an explicit ``Figure`` +
``FigureCanvasAgg`` pair (no ``pyplot`` global state), like
:mod:`cylvision.pipeline.plot`.

Vertical axis: the profiles are per image row; the axis is drawn in rows
(so the maps are not resampled) but labelled in **mL** through the
calibration model, inverted with :func:`cylvision.calibration.invert_model`
so that the volume grows upwards. Only the calibrated window
(``y_range``) is shown: outside the clicked graduations the model
extrapolates.

Overlays follow the README code: **red** = liquid / foam interface
(gradient detector), **teal** = foam / air interface (absorbance
threshold, the chosen ``k``); the other ``k`` values are thin lines on a
cool colour ramp.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from cylvision.calibration.models import invert_model

ModelFn = Callable[[Any], Any]

LIQUID_COLOR = "#e8453c"
FOAM_COLOR = "#19c3b8"
INK = "#1f1f1f"
INK_SOFT = "#5c5c5c"
GRID = "#e4e4e4"
DARK_BG = "#101014"
DARK_INK = "#e9e9ee"
DARK_INK_SOFT = "#a9a9b4"


# ---------------------------------------------------------------------------
# Axis helpers
# ---------------------------------------------------------------------------

def volume_ticks(model_fn: ModelFn, y_min: float, y_max: float,
                 step_ml: float = 50.0) -> tuple[np.ndarray, list[str]]:
    """Rows and labels of the volume ticks (multiples of ``step_ml``) inside ``[y_min, y_max]``."""
    V_lo, V_hi = (float(np.asarray(model_fn(np.array([float(y)])), dtype=float).ravel()[0])
                  for y in (y_max, y_min))
    if V_lo > V_hi:
        V_lo, V_hi = V_hi, V_lo
    V_ticks = np.arange(step_ml * math.ceil(V_lo / step_ml), V_hi + 1e-6, step_ml)
    if V_ticks.size == 0:
        return np.zeros(0), []
    y_ticks = invert_model(model_fn, y_min, y_max, V_ticks)
    keep = (y_ticks >= y_min) & (y_ticks <= y_max)
    return y_ticks[keep], [f"{v:g}" for v in V_ticks[keep]]


def _volume_axis(ax: Any, model_fn: ModelFn, y_min: float, y_max: float, *,
                 color: str, step_ml: float = 50.0) -> None:
    y_ticks, labels = volume_ticks(model_fn, y_min, y_max, step_ml)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(labels)
    ax.set_ylim(y_max, y_min)  # rows grow downwards: volume grows upwards
    ax.set_ylabel("volume on the scale (mL)", color=color, fontsize=10)
    ax.tick_params(axis="y", colors=color, labelsize=9)


def time_edges(t: np.ndarray) -> np.ndarray:
    """Cell edges for ``pcolormesh`` from sample centres (uniform end cells)."""
    t = np.asarray(t, dtype=float)
    if t.size == 1:
        return np.array([t[0] - 0.5, t[0] + 0.5])
    e = np.empty(t.size + 1)
    e[1:-1] = 0.5 * (t[1:] + t[:-1])
    e[0] = t[0] - 0.5 * (t[1] - t[0])
    e[-1] = t[-1] + 0.5 * (t[-1] - t[-2])
    return e


def _finite_rows(M: np.ndarray) -> np.ndarray:
    return np.isfinite(M).any(axis=1)


def _k_colors(ks: Sequence[int]) -> dict[int, tuple[float, float, float, float]]:
    cmap = colormaps["cool"]
    n = max(1, len(ks) - 1)
    return {k: tuple(cmap(i / n)) for i, k in enumerate(ks)}


def _overlay_interfaces(ax: Any, t: np.ndarray, y_liquid: np.ndarray | None,
                        foam_tops: Mapping[int, np.ndarray] | None, k_main: int | None,
                        *, liquid_label: str, main_label: str) -> list[Any]:
    handles: list[Any] = []
    if foam_tops:
        ks = sorted(foam_tops)
        colors = _k_colors(ks)
        for k in ks:
            if k == k_main:
                continue
            (h,) = ax.plot(t, foam_tops[k], color=colors[k], linewidth=0.8, alpha=0.85,
                           label=f"k = {k}")
            handles.append(h)
        if k_main is not None and k_main in foam_tops:
            (h,) = ax.plot(t, foam_tops[k_main], color=FOAM_COLOR, linewidth=2.2,
                           label=main_label)
            handles.insert(0, h)
    if y_liquid is not None:
        (h,) = ax.plot(t, y_liquid, color=LIQUID_COLOR, linewidth=2.2, label=liquid_label)
        handles.insert(0, h)
    return handles


# ---------------------------------------------------------------------------
# Heat maps
# ---------------------------------------------------------------------------

def plot_heatmap_A(A: np.ndarray, t: np.ndarray, *, model_fn: ModelFn,
                   y_range: tuple[int, int], out_png: Path | str,
                   y_liquid: np.ndarray | None = None,
                   foam_tops: Mapping[int, np.ndarray] | None = None,
                   k_main: int | None = None, title: str = "", vmax: float | None = None,
                   V_inf_ml: float | None = None, time_label: str = "time since the start of the pour (s)",
                   dpi: int = 150) -> Path:
    """Dark ``inferno`` map of ``A(t, V)``; ``A`` is ``(n_frames, H)``, one profile per row.

    ``y_range = (y_min, y_max)`` is the window of rows shown; ``foam_tops``
    maps ``k`` to the foam-top row per frame (NaN when not found) and
    ``k_main`` is the one used by the analysis (drawn thick, teal).
    ``vmax`` defaults to the 99th percentile of the finite values.
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    y_min, y_max = int(y_range[0]), int(y_range[1])
    ok = _finite_rows(A)
    A_s, t_s = A[ok], np.asarray(t, dtype=float)[ok]
    M = A_s[:, y_min:y_max].T
    finite = np.isfinite(M)
    if vmax is None:
        vmax = float(np.percentile(M[finite], 99)) if finite.any() else 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
    cmap = colormaps["inferno"].copy()
    cmap.set_bad(color="#1c1c22")

    fig = Figure(figsize=(13, 6.8), dpi=dpi, facecolor=DARK_BG)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_facecolor(DARK_BG)
    im = ax.pcolormesh(time_edges(t_s), np.arange(y_min, y_max + 1, dtype=float),
                       np.ma.masked_invalid(M), cmap=cmap, vmin=0.0, vmax=vmax, shading="flat",
                       rasterized=True)
    cbar = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.035)
    cbar.set_label("effective absorbance A = −ln(I / I₀)", color=DARK_INK, fontsize=10)
    cbar.ax.tick_params(colors=DARK_INK_SOFT, labelsize=9)
    cbar.outline.set_edgecolor(DARK_INK_SOFT)

    handles = _overlay_interfaces(
        ax, t_s, None if y_liquid is None else np.asarray(y_liquid, dtype=float)[ok],
        None if foam_tops is None else {k: np.asarray(v, dtype=float)[ok] for k, v in foam_tops.items()},
        k_main, liquid_label="liquid / foam (gradient detector)",
        main_label=f"foam / air, threshold A_max / {k_main} (used)")
    if V_inf_ml is not None and np.isfinite(V_inf_ml):
        y_inf = float(invert_model(model_fn, y_min, y_max, [V_inf_ml])[0])
        (h,) = ax.plot([t_s[0], t_s[-1]], [y_inf, y_inf], color=DARK_INK, linestyle=":",
                       linewidth=1.1, alpha=0.8, label=f"V∞ = {V_inf_ml:.1f} mL (final liquid)")
        handles.append(h)

    _volume_axis(ax, model_fn, y_min, y_max, color=DARK_INK_SOFT)
    ax.set_xlim(float(t_s[0]), float(t_s[-1]))
    ax.set_xlabel(time_label, color=DARK_INK_SOFT, fontsize=10)
    ax.tick_params(axis="x", colors=DARK_INK_SOFT, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color(DARK_INK_SOFT)
    if title:
        ax.set_title(title, color=DARK_INK, fontsize=12, loc="left", pad=10)
    if handles:
        ks_txt = ""
        if foam_tops:
            ks_sorted = sorted(int(k) for k in foam_tops)
            ks_txt = f"k = {ks_sorted[0]}..{ks_sorted[-1]}"
        leg = ax.legend(handles=handles, loc="lower right", fontsize=8, ncol=2, framealpha=0.85,
                        facecolor="#202028", edgecolor=DARK_INK_SOFT, labelcolor=DARK_INK,
                        title=f"foam / air row for {ks_txt}  (threshold T = A_max / k)")
        leg.get_title().set_color(DARK_INK)
        leg.get_title().set_fontsize(8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, facecolor=DARK_BG)
    return out_png


def plot_heatmap_c(c: np.ndarray, t: np.ndarray, *, model_fn: ModelFn,
                   y_range: tuple[int, int], out_png: Path | str,
                   y_liquid: np.ndarray | None = None, y_foam_top: np.ndarray | None = None,
                   title: str = "", vmax: float | None = None,
                   time_label: str = "time since the start of the pour (s)",
                   dpi: int = 150) -> Path:
    """Inverted-grey map of the liquid content ``c(t, V)`` on white (NaN = no foam = white)."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    y_min, y_max = int(y_range[0]), int(y_range[1])
    ok = _finite_rows(c)
    c_s, t_s = c[ok], np.asarray(t, dtype=float)[ok]
    M = c_s[:, y_min:y_max].T
    finite = np.isfinite(M)
    if vmax is None:
        vmax = float(np.percentile(M[finite], 99)) if finite.any() else 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
    cmap = colormaps["gray_r"].copy()
    cmap.set_bad(color="white")

    fig = Figure(figsize=(13, 6.8), dpi=dpi, facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_facecolor("white")
    im = ax.pcolormesh(time_edges(t_s), np.arange(y_min, y_max + 1, dtype=float),
                       np.ma.masked_invalid(M), cmap=cmap, vmin=0.0, vmax=vmax, shading="flat",
                       rasterized=True)
    cbar = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.035)
    cbar.set_label("liquid volume fraction of the foam  c = γ A", color=INK, fontsize=10)
    cbar.ax.tick_params(colors=INK_SOFT, labelsize=9)

    handles = []
    if y_foam_top is not None:
        (h,) = ax.plot(t_s, np.asarray(y_foam_top, dtype=float)[ok], color=FOAM_COLOR, linewidth=1.8,
                       label="foam / air (absorbance threshold)")
        handles.append(h)
    if y_liquid is not None:
        (h,) = ax.plot(t_s, np.asarray(y_liquid, dtype=float)[ok], color=LIQUID_COLOR, linewidth=1.8,
                       label="liquid / foam (gradient detector)")
        handles.append(h)
    _volume_axis(ax, model_fn, y_min, y_max, color=INK_SOFT)
    ax.set_xlim(float(t_s[0]), float(t_s[-1]))
    ax.set_xlabel(time_label, color=INK_SOFT, fontsize=10)
    ax.tick_params(axis="x", colors=INK_SOFT, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    if title:
        ax.set_title(title, color=INK, fontsize=12, loc="left", pad=10)
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=8.5, framealpha=0.9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, facecolor="white")
    return out_png


# ---------------------------------------------------------------------------
# "How it is measured" panel
# ---------------------------------------------------------------------------

def _fmt_ml(model_fn: ModelFn, y: float | None) -> str:
    if y is None or not np.isfinite(y):
        return "n/a"
    v = float(np.asarray(model_fn(np.array([float(y)])), dtype=float).ravel()[0])
    return f"{v:.0f} mL"


def plot_measurement_figure(ref_intensity: np.ndarray, live_intensity: np.ndarray, A: np.ndarray,
                            A_z: np.ndarray, *, y_liquid: float | None, y_foam_top: int | None,
                            A_max: float, threshold: float, band: tuple[int, int],
                            model_fn: ModelFn, y_range: tuple[int, int], out_png: Path | str,
                            k: float, title: str = "", vmax_A: float | None = None,
                            dpi: int = 150) -> Path:
    """Four panels: reference ``I0`` | live ``I`` | ``A(x, y)`` | ``A(z)`` with the threshold.

    ``band = (x_lo, x_hi)`` are the columns averaged into ``A(z)``; both
    interfaces are drawn on every panel (red = liquid / foam, teal = foam /
    air) and the threshold ``T = A_max / k`` is the dashed vertical line of
    the profile panel.
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    y_min, y_max = int(y_range[0]), int(y_range[1])
    H, W = A.shape
    if vmax_A is None:
        sub = A[y_min:y_max]
        vmax_A = float(np.percentile(sub[np.isfinite(sub)], 99.5)) if np.isfinite(sub).any() else 1.0
    img_w_in = 1.45
    fig = Figure(figsize=(3 * img_w_in + 5.6, 7.6), dpi=dpi, facecolor="white")
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(1, 6, width_ratios=[img_w_in, img_w_in, img_w_in, 0.2, 0.5, 3.4],
                          wspace=0.08, left=0.075, right=0.985, top=0.9, bottom=0.075)
    ax_ref = fig.add_subplot(gs[0, 0])
    ax_live = fig.add_subplot(gs[0, 1], sharey=ax_ref)
    ax_A = fig.add_subplot(gs[0, 2], sharey=ax_ref)
    ax_cb = fig.add_subplot(gs[0, 3])
    ax_p = fig.add_subplot(gs[0, 5], sharey=ax_ref)

    extent = (0, W, y_max, y_min)
    grey_max = float(max(np.nanmax(ref_intensity), 1.0))
    ax_ref.imshow(ref_intensity[y_min:y_max], cmap="gray", vmin=0, vmax=grey_max, extent=extent, aspect="auto",
                  interpolation="nearest")
    ax_live.imshow(live_intensity[y_min:y_max], cmap="gray", vmin=0, vmax=grey_max, extent=extent, aspect="auto",
                   interpolation="nearest")
    im = ax_A.imshow(A[y_min:y_max], cmap="inferno", vmin=0, vmax=vmax_A, extent=extent, aspect="auto",
                     interpolation="nearest")
    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label("A", color=INK_SOFT, fontsize=9)
    cb.ax.tick_params(colors=INK_SOFT, labelsize=8)
    for ax, name in ((ax_ref, "reference I₀ (empty cylinder)"), (ax_live, "live frame I"),
                     (ax_A, "A(x, y) = −ln(I / I₀)")):
        ax.set_title(name, fontsize=9.5, color=INK, pad=6)
        ax.set_xticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
    x_lo, x_hi = band
    for ax in (ax_live, ax_A):
        ax.axvline(x_lo, color="#ffd21f", linewidth=0.9, linestyle="--", alpha=0.9)
        ax.axvline(x_hi, color="#ffd21f", linewidth=0.9, linestyle="--", alpha=0.9)

    # Profile panel.
    rows = np.arange(H)
    ax_p.plot(A_z, rows, color=INK, linewidth=1.3,
              label=f"A(z), mean over |x − cₓ| ≤ {(x_hi - x_lo) // 2} px (dashed band)")
    ax_p.axvline(threshold, color=FOAM_COLOR, linestyle="--", linewidth=1.2,
                 label=f"threshold T = A_max / {k:g} = {threshold:.2f}")
    if y_foam_top is not None and y_liquid is not None:
        yl = int(round(float(y_liquid)))
        ax_p.fill_betweenx(rows[y_foam_top:yl], 0, np.nan_to_num(A_z[y_foam_top:yl]), color="#f6c7a0",
                           alpha=0.55, linewidth=0, label="foam: ∫A dz enters the mass balance")
    for ax in (ax_ref, ax_live, ax_A, ax_p):
        if y_liquid is not None:
            ax.axhline(float(y_liquid), color=LIQUID_COLOR, linewidth=1.8)
        if y_foam_top is not None:
            ax.axhline(float(y_foam_top), color=FOAM_COLOR, linewidth=1.8)
    ax_p.set_xlim(min(0.0, float(np.nanmin(A_z[y_min:y_max]))) - 0.02, max(vmax_A, A_max) * 1.08)
    ax_p.set_xlabel("effective absorbance A(z)", color=INK_SOFT, fontsize=10)
    ax_p.grid(True, color=GRID, linewidth=0.7)
    ax_p.set_axisbelow(True)
    ax_p.tick_params(colors=INK_SOFT, labelsize=9)
    ax_p.tick_params(axis="y", labelleft=False, length=0)
    for sp in ax_p.spines.values():
        sp.set_color(GRID)
    ax_p.set_title("radial profile A(z) and the foam-top threshold", fontsize=9.5, color=INK, pad=6)
    x_text = max(vmax_A, A_max) * 1.05
    if y_liquid is not None:
        ax_p.annotate(f"liquid / foam (gradient detector)  y = {float(y_liquid):.1f} px  ·  "
                      f"{_fmt_ml(model_fn, y_liquid)} of liquid",
                      (x_text, float(y_liquid)), xytext=(-4, -6), textcoords="offset points", ha="right",
                      va="top", fontsize=8, color=LIQUID_COLOR)
    if y_foam_top is not None:
        ax_p.annotate(f"foam / air (first row with A < T)  y = {int(y_foam_top)} px  ·  "
                      f"{_fmt_ml(model_fn, y_foam_top)} liquid + foam",
                      (x_text, float(y_foam_top)), xytext=(-4, 6), textcoords="offset points", ha="right",
                      va="bottom", fontsize=8, color=FOAM_COLOR)
    if y_liquid is not None and np.isfinite(A_z[:int(round(float(y_liquid)))]).any():
        y_amax = int(np.nanargmax(A_z[:int(round(float(y_liquid)))]))
        ax_p.annotate(f"A_max = {A_max:.2f}", (A_max, y_amax), xytext=(-8, 0), textcoords="offset points",
                      ha="right", va="center", fontsize=8, color=INK_SOFT,
                      arrowprops={"arrowstyle": "-", "color": INK_SOFT, "lw": 0.6})
    ax_p.legend(loc="upper right", fontsize=7.5, framealpha=0.9, labelcolor=INK)

    _volume_axis(ax_ref, model_fn, y_min, y_max, color=INK_SOFT)
    for ax in (ax_live, ax_A):
        ax.tick_params(axis="y", labelleft=False, length=0)
    if title:
        fig.suptitle(title, fontsize=12, color=INK, x=0.075, ha="left", y=0.975)
    fig.savefig(out_png, dpi=dpi, facecolor="white")
    return out_png


__all__ = [
    "LIQUID_COLOR", "FOAM_COLOR", "volume_ticks", "time_edges", "plot_heatmap_A", "plot_heatmap_c",
    "plot_measurement_figure",
]
