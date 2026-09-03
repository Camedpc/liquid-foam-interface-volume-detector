"""V(t) figure: liquid, total and foam volumes against time.

The figure is rendered with the ``Agg`` backend through an explicit
``Figure`` + ``FigureCanvasAgg`` pair, so library code never touches the
``pyplot`` global state (no window, no leaked figures in a batch).

Design: one shared volume axis (mL) for the three series, thin lines with
small markers (rows are sparse when ``frame_step > 1``, so a line alone can
look empty), a legend plus a direct label at the end of each series, and an
optional shaded band of ``V ± u_total(V)`` from an ``UncertaintyBudget``.
Series colours are fixed by role: liquid = blue, total = orange, foam = aqua.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

SERIES: tuple[tuple[str, str, str], ...] = (
    # (column, label, colour)
    ("V_lower_mL", "liquid (lower interface)", "#2a78d6"),
    ("V_total_mL", "liquid + foam (upper interface)", "#eb6834"),
    ("V_foam_mL", "foam", "#1baf7a"),
)
_INK = "#1f1f1f"
_INK_SOFT = "#5c5c5c"
_GRID = "#e4e4e4"
_SURFACE = "#ffffff"


def _column(rows: Sequence[dict[str, Any]], key: str) -> np.ndarray:
    out = np.empty(len(rows), dtype=float)
    for i, r in enumerate(rows):
        v = r.get(key)
        try:
            out[i] = float(v) if v is not None else math.nan
        except (TypeError, ValueError):
            out[i] = math.nan
    return out


def _style_axis(ax: Any) -> None:
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_INK_SOFT, labelsize=9)
    ax.grid(True, axis="y", color=_GRID, linewidth=0.8)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)


def plot_levels(rows: Sequence[dict[str, Any]], *, title: str, out_png: Path | str,
                budget: Any | None = None, time_label: str = "time (s)",
                dpi: int = 150) -> Path:
    """Save the V(t) figure of a batch to ``out_png``.

    ``budget`` (an ``UncertaintyBudget`` or anything with ``total(V) -> u``)
    adds a shaded ``±u`` band around each series. Returns the written path.
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    t = _column(rows, "t_sec")

    fig = Figure(figsize=(11, 5.5), dpi=dpi, facecolor=_SURFACE)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    _style_axis(ax)

    handles = []
    end_labels: list[tuple[float, float, str, str]] = []
    for key, label, color in SERIES:
        V = _column(rows, key)
        ok = np.isfinite(t) & np.isfinite(V)
        if not ok.any():
            continue
        tt, VV = t[ok], V[ok]
        if budget is not None:
            try:
                u = np.array([float(budget.total(float(v))) for v in VV], dtype=float)
                ax.fill_between(tt, VV - u, VV + u, color=color, alpha=0.18, linewidth=0,
                                label=f"{label}: ±u")
            except Exception:
                pass
        (h,) = ax.plot(tt, VV, color=color, linewidth=1.6, marker=".", markersize=3.5,
                       markeredgewidth=0, label=label)
        handles.append(h)
        end_labels.append((float(tt[-1]), float(VV[-1]), label.split(" (")[0], color))

    ax.set_xlabel(time_label, color=_INK_SOFT, fontsize=10)
    ax.set_ylabel("volume (mL)", color=_INK_SOFT, fontsize=10)
    ax.set_title(title, color=_INK, fontsize=12, loc="left", pad=12)

    if end_labels:
        x_max = max(x for x, _, _, _ in end_labels)
        x_min = float(np.nanmin(t)) if np.isfinite(t).any() else 0.0
        ax.set_xlim(right=x_max + 0.08 * max(1.0, x_max - x_min))
        for x, y, text, color in end_labels:
            ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=8.5, color=_INK_SOFT,
                        bbox={"boxstyle": "round,pad=0.15", "fc": _SURFACE, "ec": "none", "alpha": 0.8})
    if handles:
        ax.legend(handles=handles, loc="best", fontsize=9, frameon=False, labelcolor=_INK)
    else:
        ax.text(0.5, 0.5, "no valid row", transform=ax.transAxes, ha="center", va="center",
                color=_INK_SOFT)

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, facecolor=_SURFACE)
    return out_png


__all__ = ["SERIES", "plot_levels"]
