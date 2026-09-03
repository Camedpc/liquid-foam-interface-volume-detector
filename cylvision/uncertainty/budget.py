"""Uncertainty budget of a volume reading.

A level read from a frame goes through four independent steps, each with its
own uncertainty (all expressed in mL, standard uncertainties, ``k = 1``):

``u_calibration``
    How well the row -> volume model reproduces the clicked graduations.  It
    is the RMS residual of the recommended model, floored by the reading
    resolution of the scale: a click is placed on a graduation ring whose
    thin step is ``s`` mL, so the clicked volume itself is only known within a
    uniform ``±s/2`` window, ``u = s / sqrt(12)`` (2.9 mL for the 10 mL step of
    a 1000 mL cylinder).  With an interpolating model (PCHIP) the RMS is zero
    by construction and the floor is what remains.

``u_pixel(V)``
    Quantisation of the row.  A row is an integer; the true interface lies
    anywhere in the pixel, ``u = 1 px / sqrt(12)``, converted with the local
    slope ``|dV/dy|`` (mL/px).  Sub-pixel averaging over the band makes the
    real figure smaller, so this is conservative.

``u_curvature(V)``
    Geometric: the interface is a horizontal circle whose projection is an
    arc of ellipse, sampled over the averaging band.  See
    :mod:`cylvision.uncertainty.curvature`; zero on the camera horizon,
    ~1 mL at the ends of a 1000 mL cylinder.

``u_method``
    Empirical: spread of the per-column detections in a frame, summarised
    (median) over the run.  See :mod:`cylvision.uncertainty.empirical`.

They are combined in quadrature (independent contributions)::

    u_total(V)^2 = u_calibration^2 + u_pixel(V)^2 + u_curvature(V)^2 + u_method^2

Not included, and why: lens distortion (a few percent at the frame edges of
a webcam, absorbed by the calibration which is fitted on the same image
region as the readings), sensor noise (already
inside ``u_method``), the tolerance on the physical radius (enters
``u_curvature`` only, a second-order effect on a ~1 mL term) and the time
stamp of a frame (``1 / (2 sqrt(3) fps)``, a time uncertainty, not a volume).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from cylvision.calibration.models import slope_ml_per_px

from .curvature import BG, GRID, INK, MUTED, SPINE, CameraGeometry, curvature_uncertainty_fn, volume_to_row_fn
from .empirical import summarize_method_uncertainty

_SQRT12 = float(np.sqrt(12.0))

# Fixed categorical palette (one slot per contribution, never re-assigned).
COLORS = {
    "calibration": "#2a78d6",
    "pixel": "#eb6834",
    "curvature": "#1baf7a",
    "method": "#eda100",
    "total": INK,
}
LABELS = {
    "calibration": r"$u_{\mathrm{calibration}}$",
    "pixel": r"$u_{\mathrm{pixel}}(V)$",
    "curvature": r"$u_{\mathrm{curvature}}(V)$",
    "method": r"$u_{\mathrm{method}}$",
    "total": r"$u_{\mathrm{total}}(V)$",
}

# Thin graduation step (mL) of the usual cylinders, by capacity.
THIN_STEP_BY_CAPACITY_ML = {50.0: 1.0, 100.0: 1.0, 250.0: 2.0, 500.0: 5.0, 1000.0: 10.0, 2000.0: 20.0}


def default_scale_resolution_ml(spec: Any) -> float:
    """Thin graduation step of the cylinder (mL): ``spec.thin_step_ml`` if present,
    else a lookup on the capacity, else a tenth of the clicked graduation step."""
    step = getattr(spec, "thin_step_ml", None)
    if step is not None:
        return float(step)
    cap = getattr(spec, "capacity_ml", None)
    if cap is not None and float(cap) in THIN_STEP_BY_CAPACITY_ML:
        return THIN_STEP_BY_CAPACITY_ML[float(cap)]
    grads = np.asarray(getattr(spec, "graduations_ml", []) or [], dtype=float)
    if grads.size >= 2:
        return float(np.min(np.diff(np.sort(grads))) / 10.0)
    return 1.0


def calibration_rms_ml(calib: Any) -> float:
    """RMS of the residuals of the recommended model (0 if unavailable)."""
    residuals = getattr(calib, "residuals_ml", None) or {}
    model = getattr(calib, "recommended_model", None)
    res = residuals.get(model) if model is not None else None
    if not res:
        return 0.0
    arr = np.asarray(res, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.sqrt(np.mean(arr ** 2))) if arr.size else 0.0


@dataclass
class UncertaintyBudget:
    """Combined uncertainty of a volume reading (see module docstring)."""

    u_calibration_ml: float
    u_pixel_ml_fn: Callable[[float], float]
    u_curvature_ml_fn: Callable[[float], float]
    u_method_ml: float
    u_method_available: bool = True
    notes: dict[str, str] = field(default_factory=dict)

    def components(self, V: float) -> dict[str, float]:
        return {
            "calibration": float(self.u_calibration_ml),
            "pixel": float(self.u_pixel_ml_fn(V)),
            "curvature": float(self.u_curvature_ml_fn(V)),
            "method": float(self.u_method_ml),
        }

    def total(self, V: float) -> float:
        """Quadrature sum of the four contributions at volume ``V``."""
        c = self.components(V)
        return float(np.sqrt(sum(v * v for v in c.values())))

    def table(self, V_values: Iterable[float]) -> list[dict]:
        rows = []
        for V in V_values:
            c = self.components(V)
            rows.append({"V_ml": float(V), "u_calibration_ml": c["calibration"], "u_pixel_ml": c["pixel"],
                         "u_curvature_ml": c["curvature"], "u_method_ml": c["method"],
                         "u_total_ml": self.total(V)})
        return rows


def build_budget(calib: Any, geom: CameraGeometry, params: Any, model_fn: Callable[..., Any],
                 rows: Sequence[dict] | None = None, *, scale_resolution_ml: float | None = None,
                 r_px: int | None = None, method_key: str = "u_method_lower_mL") -> UncertaintyBudget:
    """Assemble the budget of a run.

    ``params.r_lower`` gives the band half-width unless ``r_px`` is passed;
    ``rows`` are the per-frame batch rows (their ``method_key`` column is
    summarised by its median).  Without rows ``u_method`` is a 0 placeholder
    and the budget says so in ``notes``.
    """
    notes: dict[str, str] = {}
    res_ml = default_scale_resolution_ml(getattr(calib, "cylinder", None)) if scale_resolution_ml is None \
        else float(scale_resolution_ml)
    rms = calibration_rms_ml(calib)
    floor = res_ml / _SQRT12
    u_cal = max(rms, floor)
    notes["calibration"] = (f"RMS residual of the '{getattr(calib, 'recommended_model', '?')}' model = {rms:.2f} mL; "
                            f"scale resolution {res_ml:g} mL / sqrt(12) = {floor:.2f} mL; kept the larger.")

    y_of_V = volume_to_row_fn(calib, model_fn)

    def u_pixel(V: float) -> float:
        y = float(np.asarray(y_of_V(V)).ravel()[0])
        return slope_ml_per_px(model_fn, y) / _SQRT12

    notes["pixel"] = "1 px / sqrt(12) converted with the local slope |dV/dy|."

    r = int(r_px) if r_px is not None else int(getattr(params, "r_lower", 80))
    u_curv = curvature_uncertainty_fn(calib, geom, r, model_fn)
    notes["curvature"] = f"geometric model [{geom.source}], band half-width r = {r} px, u = Delta / (2 sqrt(3))."

    if rows:
        summary = summarize_method_uncertainty(rows, key=method_key)
        available = summary["n"] > 0
        u_method = summary["median"] if available else 0.0
        notes["method"] = (f"median of the per-frame '{method_key}' over {summary['n']} frames "
                           f"(mean {summary['mean']:.2f}, p95 {summary['p95']:.2f} mL)." if available
                           else f"no finite '{method_key}' value in the rows: placeholder 0.")
    else:
        available = False
        u_method = 0.0
        notes["method"] = "no batch rows given: placeholder 0 (run the batch to measure it)."

    return UncertaintyBudget(u_calibration_ml=float(u_cal), u_pixel_ml_fn=u_pixel, u_curvature_ml_fn=u_curv,
                             u_method_ml=float(u_method), u_method_available=available, notes=notes)


def budget_markdown_table(budget: UncertaintyBudget, V_values: Iterable[float]) -> str:
    """GitHub-flavoured Markdown table of the contributions at the given volumes."""
    head = ("| V (mL) | u_calibration (mL) | u_pixel (mL) | u_curvature (mL) | u_method (mL) | u_total (mL) |\n"
            "|---:|---:|---:|---:|---:|---:|")
    lines = [head]
    for r in budget.table(V_values):
        method = f"{r['u_method_ml']:.2f}" if budget.u_method_available else f"{r['u_method_ml']:.2f} *"
        lines.append(f"| {r['V_ml']:.0f} | {r['u_calibration_ml']:.2f} | {r['u_pixel_ml']:.2f} | "
                     f"{r['u_curvature_ml']:.2f} | {method} | {r['u_total_ml']:.2f} |")
    if not budget.u_method_available:
        lines.append("")
        lines.append("\\* placeholder: no per-frame method uncertainty available for this run.")
    return "\n".join(lines)


def _spread_labels(y: Sequence[float], min_gap: float) -> list[float]:
    """Label rows pushed apart so that two labels are never closer than ``min_gap``.

    The labels are sorted by their anchor, then each one is moved up just
    enough to clear its lower neighbour; the result is returned in the input
    order (a plain one-pass de-overlap, enough for a handful of end labels).
    """
    order = sorted(range(len(y)), key=lambda i: y[i])
    out = [float(v) for v in y]
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < min_gap:
            out[b] = out[a] + min_gap
    return out


def render_budget_figure(budget: UncertaintyBudget, V_range: Sequence[float], out_png: Path | str) -> None:
    """Two panels: each contribution and the total vs V; share of the variance vs V."""
    V = np.asarray(V_range, dtype=float)
    if V.size == 2:
        V = np.linspace(V[0], V[1], 200)
    comp = {k: np.array([budget.components(v)[k] for v in V]) for k in ("calibration", "pixel", "curvature", "method")}
    total = np.sqrt(sum(c ** 2 for c in comp.values()))

    fig = Figure(figsize=(12.5, 4.8), dpi=100, facecolor=BG)
    FigureCanvasAgg(fig)
    ax1, ax2 = fig.subplots(1, 2, gridspec_kw=dict(width_ratios=[1.15, 1.0], wspace=0.22))

    for k, arr in comp.items():
        ax1.plot(V, arr, color=COLORS[k], lw=2.0, label=LABELS[k])
    ax1.plot(V, total, color=COLORS["total"], lw=2.6, label=LABELS["total"])
    y_top = float(total.max()) * 1.18
    ends = [(float(arr[-1]), k, COLORS[k], "normal") for k, arr in comp.items()]
    ends.append((float(total[-1]), "total", INK, "bold"))
    for y_lab, (_y, k, color, weight) in zip(_spread_labels([e[0] for e in ends], 0.045 * y_top), ends):
        ax1.text(V[-1] + 0.01 * (V[-1] - V[0]), y_lab, k, fontsize=8, color=color, va="center", fontweight=weight)
    ax1.set_xlim(V[0], V[-1] + 0.12 * (V[-1] - V[0]))
    ax1.set_ylim(0, y_top)
    ax1.set_xlabel("volume V  (mL)", fontsize=10, color=INK)
    ax1.set_ylabel("standard uncertainty  (mL)", fontsize=10, color=INK)
    ax1.set_title("Contributions and quadrature total", fontsize=11, color=INK, loc="left")
    ax1.legend(loc="upper center", fontsize=8.5, ncol=3, frameon=False)

    shares = [comp[k] ** 2 / np.maximum(total ** 2, 1e-12) for k in comp]
    ax2.stackplot(V, *shares, colors=[COLORS[k] for k in comp], labels=[k for k in comp], alpha=0.9)
    ax2.set_xlim(V[0], V[-1])
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_yticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"])
    ax2.set_xlabel("volume V  (mL)", fontsize=10, color=INK)
    ax2.set_ylabel("share of the variance  $u_i^2 / u_{\\mathrm{total}}^2$", fontsize=10, color=INK)
    ax2.set_title("Who dominates where", fontsize=11, color=INK, loc="left")
    ax2.legend(loc="lower right", fontsize=8.5, frameon=True, facecolor=BG, edgecolor=SPINE)

    for ax in (ax1, ax2):
        ax.set_facecolor(BG)
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK, labelsize=9, color=SPINE)
        for name, s in ax.spines.items():
            s.set_color(SPINE)
            s.set_visible(name not in ("top", "right"))
    if not budget.u_method_available:
        fig.text(0.01, 0.01, "u_method is a placeholder (0): no batch rows were given.", fontsize=8, color=MUTED)
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=170, facecolor=BG, bbox_inches="tight", pad_inches=0.15)
