"""Tk dark control panel of the tuner (gradient detector only).

The panel replaces OpenCV trackbars by a themed Tkinter window with four
colour-coded sections and two action buttons:

* **VIEW** (blue) -- canvas layout: specimen panel on/off, inline / stacked.
* **CYLINDER** (mauve) -- ``y_top``, ``y_bottom`` (gradient masks), ``cx``
  (axis column inside the crop) and the video ``frame`` slider.
* **GRADIENT** (teal) -- the detector: ``polarity`` first (it decides the
  sign of both interfaces), then the thresholds ``T_lower`` / ``T_upper``,
  the bands ``r_lower`` / ``r_upper``, the smoothing ``blur_sigma`` /
  ``blur_h``, the connected-component filter ``min_h_upper`` and the
  ``channel`` used as grey level.
* **SAMPLING** (peach) -- ``frame_step`` (one frame in K is analysed by the
  batch) with presets.
* Buttons ``Run batch`` / ``Abort`` and a footer with the hotkeys and a
  one-line summary of the current values.

Integration with the OpenCV loop
--------------------------------
Tkinter and OpenCV each own an event loop. Instead of ``root.mainloop()``
the tuner calls :meth:`ControlsPanel.update` once per iteration of its
``cv2.waitKey`` loop; :meth:`ControlsPanel.is_alive` detects a window
closed by the user. Sliders store integers (``tk.IntVar``); real-valued
parameters use a scaled slider (``blur_sigma x10``) and are converted in
:meth:`ControlsPanel.get_params`.

The ``1/2 columns`` button rebuilds every widget in place (the Tk root is
created once and never destroyed until :meth:`destroy`), replaying the
current values. :meth:`grab_screenshot` returns the panel as a BGR image
(used by the README figure generator).
"""
from __future__ import annotations

import os
import tkinter as tk
from collections.abc import Callable, Sequence
from tkinter import ttk

import numpy as np

from cylvision.detection.interfaces import POLARITIES, DetectionParams
from cylvision.ui import theme

PANEL_TITLE = "liquid-foam-interface-volume-detector  -  control panel"

POLARITY_LABELS: tuple[str, str] = ("lower darker", "lower brighter")
CHANNEL_OPTIONS: tuple[str, ...] = ("gray", "G", "R", "B")
FRAME_STEP_PRESETS: tuple[tuple[str, int], ...] = (("all", 1), ("1/2", 2), ("1/3", 3), ("1/5", 5), ("1/10", 10))
FRAME_STEP_MAX = 60
T_MAX = 500
BLUR_SIGMA_X10_MAX = 100
BLUR_H_MAX = 101
MIN_H_UPPER_MAX = 50

HOTKEYS: tuple[tuple[str, str, str], ...] = (
    ("Enter", "run batch", theme.PALETTE["green"]),
    ("Esc", "abort", theme.PALETTE["red"]),
    ("t", "specimen", theme.SECTION_VIEW),
    ("l", "inline", theme.SECTION_VIEW),
    ("f", "fullscreen", theme.SECTION_VIEW),
    ("<- ->", "frame", theme.SECTION_CYLINDER),
    ("Home", "first frame", theme.SECTION_CYLINDER),
)

Formatter = Callable[[int], str]


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


class ControlsPanel:
    """Tk window holding every knob of the gradient tuner (see module doc)."""

    def __init__(self, *, params: DetectionParams, H: int, W_crop: int,
                 n_frames: int | None, init_frame_idx: int = 0,
                 include_sampling: bool = True, show_batch_button: bool = True,
                 two_columns: bool = False, frame_step: int = 1,
                 show_specimen: bool = True, inline: bool = True) -> None:
        self._H = max(2, int(H))
        self._W_crop = max(2, int(W_crop))
        self._r_max = max(1, self._W_crop // 2)
        self._n_frames = None if n_frames is None else max(1, int(n_frames))
        self._init_frame_idx = int(init_frame_idx)
        self._include_sampling = bool(include_sampling)
        self._show_batch_button = bool(show_batch_button)
        self._two_columns = bool(two_columns)
        self._init_params = params
        self._init_frame_step = max(1, int(frame_step))
        self._init_show_specimen = bool(show_specimen)
        self._init_inline = bool(inline)

        self._alive = True
        self._vars: dict[str, tk.IntVar] = {}
        self._value_labels: dict[str, tk.Label] = {}
        self._formatters: dict[str, Formatter] = {}
        self._radio_groups: dict[str, list[tk.Button]] = {}
        self._radio_accents: dict[str, str] = {}
        self._radio_callbacks: dict[str, Callable[[int], None]] = {}
        self._summary_label: tk.Label | None = None
        self._last_summary = ""
        self._batch_requested = False
        self._abort_requested = False

        self.root = tk.Tk()
        self.root.title(PANEL_TITLE)
        self.root.configure(bg=theme.BG)
        try:
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        except tk.TclError:
            pass
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")  # the only built-in theme that honours custom colours
        except tk.TclError:
            pass
        self._configure_styles(style)

        self._build_widgets()
        self._apply_params(params)
        self._fit_geometry()
        # A single column taller than the screen would hide the buttons:
        # fall back to two columns automatically.
        if not self._two_columns and self._too_tall():
            self.toggle_layout()

    def _too_tall(self) -> bool:
        try:
            self.root.update_idletasks()
            return self.root.winfo_reqheight() > self.root.winfo_screenheight() - 90
        except tk.TclError:
            return False

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------
    def _build_widgets(self) -> None:
        outer = tk.Frame(self.root, bg=theme.BG, padx=14, pady=10)
        outer.pack(fill="both", expand=True)

        header = tk.Frame(outer, bg=theme.BG)
        header.pack(fill="x", pady=(0, 6))
        tk.Label(header, text="Interface detection", bg=theme.BG, fg=theme.TEXT,
                 font=theme.FONT_TITLE).pack(side="left")
        self._flat_button(header, "⇄ 1 column" if self._two_columns else "⇄ 2 columns",
                          self.toggle_layout).pack(side="right")

        # Footer (summary + hotkeys) and action buttons are packed at the
        # bottom FIRST so that they stay visible when the window is capped
        # to the screen height; the sections fill the space above them.
        self._summary_label = tk.Label(outer, text="", bg=theme.BG, fg=theme.SUBTEXT,
                                       font=theme.FONT_VALUE, anchor="w", justify="left")
        self._summary_label.pack(side="bottom", fill="x", pady=(6, 0))
        self._last_summary = ""
        footer = tk.Frame(outer, bg=theme.BG, pady=4)
        footer.pack(side="bottom", fill="x", pady=(10, 0))
        for key, label, col in HOTKEYS:
            chip = tk.Frame(footer, bg=theme.SURFACE, padx=6, pady=2)
            chip.pack(side="left", padx=3)
            tk.Label(chip, text=key, bg=theme.SURFACE, fg=col, font=theme.FONT_HOTKEY).pack(side="left", padx=(0, 4))
            tk.Label(chip, text=label, bg=theme.SURFACE, fg=theme.SUBTEXT, font=theme.FONT_HOTKEY).pack(side="left")
        if self._show_batch_button:
            actions = tk.Frame(outer, bg=theme.BG, pady=6)
            actions.pack(side="bottom", fill="x", pady=(10, 0))
            tk.Button(actions, text="▶  Run batch", bg=theme.PALETTE["green"], fg=theme.BG,
                      activebackground=theme.PALETTE["green"], activeforeground=theme.BG,
                      bd=0, padx=14, pady=6, font=theme.FONT_TITLE, relief="flat", cursor="hand2",
                      command=self._on_run_batch).pack(side="left", padx=(0, 6))
            tk.Button(actions, text="✖  Abort", bg=theme.PALETTE["red"], fg=theme.BG,
                      activebackground=theme.PALETTE["red"], activeforeground=theme.BG,
                      bd=0, padx=14, pady=6, font=theme.FONT_LABEL, relief="flat", cursor="hand2",
                      command=self._on_abort).pack(side="left")

        if self._two_columns:
            cols = tk.Frame(outer, bg=theme.BG)
            cols.pack(fill="x", anchor="n")
            left = tk.Frame(cols, bg=theme.BG)
            left.pack(side="left", fill="both", expand=True, padx=(0, 8), anchor="n")
            right = tk.Frame(cols, bg=theme.BG)
            right.pack(side="left", fill="both", expand=True, padx=(8, 0), anchor="n")
        else:
            left = right = outer

        # --- VIEW -------------------------------------------------------
        view = self._section(left, "VIEW  ·  canvas layout", theme.SECTION_VIEW)
        self._add_radio(view, "show_specimen", "specimen panel", ("off", "on"),
                        init_index=1 if self._init_show_specimen else 0, accent=theme.SECTION_VIEW)
        self._add_radio(view, "inline", "panels", ("stacked", "inline (1 row)"),
                        init_index=1 if self._init_inline else 0, accent=theme.SECTION_VIEW)

        # --- CYLINDER ---------------------------------------------------
        cyl = self._section(left, "CYLINDER  ·  masks and axis", theme.SECTION_CYLINDER)
        self._add_slider(cyl, "y_top", "y_top (px)", 0, self._H - 1, 0)
        self._add_slider(cyl, "y_bottom", "y_bottom (px)", 0, self._H - 1, self._H - 1)
        self._add_slider(cyl, "cx", "cx (px, in crop)", 0, self._W_crop - 1, self._W_crop // 2)
        if self._n_frames is not None:
            self._add_slider(cyl, "frame", "frame", 0, max(1, self._n_frames - 1),
                             _clamp(self._init_frame_idx, 0, self._n_frames - 1))

        # --- GRADIENT ---------------------------------------------------
        grad = self._section(right, "GRADIENT  ·  interface detection", theme.SECTION_GRADIENT)
        self._add_radio(grad, "polarity", "polarity", POLARITY_LABELS, init_index=0,
                        accent=theme.SECTION_GRADIENT)
        self._add_slider(grad, "T_lower", "T_lower (liquid/foam)", 0, T_MAX, 40)
        self._add_slider(grad, "T_upper", "T_upper (foam/air)", 0, T_MAX, 47)
        self._add_slider(grad, "r_lower", "r_lower (px)", 0, self._r_max, min(80, self._r_max))
        self._add_slider(grad, "r_upper", "r_upper (px)", 0, self._r_max, min(80, self._r_max))
        self._add_slider(grad, "blur_sigma_x10", "blur_sigma × 10", 0, BLUR_SIGMA_X10_MAX, 50,
                         fmt=lambda v: f"{v / 10:.1f}")
        self._add_slider(grad, "blur_h", "blur_h (px)", 0, BLUR_H_MAX, 38)
        self._add_slider(grad, "min_h_upper", "min_h_upper (px)", 0, MIN_H_UPPER_MAX, 3)
        self._add_radio(grad, "channel", "channel", CHANNEL_OPTIONS, init_index=0,
                        accent=theme.SECTION_GRADIENT)

        # --- SAMPLING ---------------------------------------------------
        if self._include_sampling:
            samp = self._section(right, "SAMPLING  ·  batch frame step", theme.SECTION_SAMPLING)
            self._add_radio(samp, "frame_step_preset", "preset",
                            tuple(name for name, _ in FRAME_STEP_PRESETS), init_index=0,
                            accent=theme.SECTION_SAMPLING,
                            on_select=lambda i: self.set("frame_step", FRAME_STEP_PRESETS[i % len(FRAME_STEP_PRESETS)][1]))
            self._add_slider(samp, "frame_step", "1 frame in K", 1, FRAME_STEP_MAX,
                             _clamp(self._init_frame_step, 1, FRAME_STEP_MAX),
                             fmt=lambda v: f"1/{int(v)}")

    def _fit_geometry(self) -> None:
        self.root.update_idletasks()
        min_w = 900 if self._two_columns else 520
        w = max(self.root.winfo_reqwidth(), min_w)
        h = min(self.root.winfo_reqheight(), max(200, self.root.winfo_screenheight() - 90))
        self.root.geometry(f"{w}x{h}")
        self.root.minsize(w, min(h, 400))

    def toggle_layout(self) -> None:
        """Switch between 1 and 2 columns, keeping every value."""
        if not self._alive:
            return
        state = {n: int(v.get()) for n, v in self._vars.items()}
        for child in list(self.root.winfo_children()):
            try:
                child.destroy()
            except tk.TclError:
                pass
        self._vars.clear()
        self._value_labels.clear()
        self._formatters.clear()
        self._radio_groups.clear()
        self._radio_accents.clear()
        self._radio_callbacks.clear()
        self._batch_requested = False
        self._abort_requested = False
        self._two_columns = not self._two_columns
        self._build_widgets()
        for name, value in state.items():
            if name in self._vars:
                self.set(name, value)
        self._fit_geometry()
        self._refresh_summary(force=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get(self, name: str) -> int:
        """Raw integer value of a slider / radio index."""
        return int(self._vars[name].get())

    def set(self, name: str, value: int) -> None:
        if name not in self._vars:
            return
        self._vars[name].set(int(value))
        lbl = self._value_labels.get(name)
        fmt = self._formatters.get(name)
        if lbl is not None and fmt is not None:
            try:
                lbl.config(text=fmt(int(value)))
            except tk.TclError:
                pass
        if name in self._radio_groups:
            self._refresh_radio(name)

    def has(self, name: str) -> bool:
        return name in self._vars

    def get_params(self) -> DetectionParams:
        """Current sliders as a :class:`DetectionParams`."""
        y_bottom = self.get("y_bottom")
        return DetectionParams(
            polarity=POLARITIES[self.get("polarity") % len(POLARITIES)],  # type: ignore[arg-type]
            T_lower=float(self.get("T_lower")),
            T_upper=float(self.get("T_upper")),
            r_lower=self.get("r_lower"),
            r_upper=self.get("r_upper"),
            blur_sigma=self.get("blur_sigma_x10") / 10.0,
            blur_h=self.get("blur_h"),
            min_h_upper=self.get("min_h_upper"),
            y_top=self.get("y_top"),
            y_bottom=y_bottom,
            cx=self.get("cx"),
            channel=CHANNEL_OPTIONS[self.get("channel") % len(CHANNEL_OPTIONS)],
        )

    def set_params(self, params: DetectionParams) -> None:
        """Push a :class:`DetectionParams` into the sliders (values clamped)."""
        self._apply_params(params)
        self._refresh_summary(force=True)

    def _apply_params(self, p: DetectionParams) -> None:
        self.set("polarity", POLARITIES.index(p.polarity) if p.polarity in POLARITIES else 0)
        self.set("T_lower", _clamp(int(round(p.T_lower)), 0, T_MAX))
        self.set("T_upper", _clamp(int(round(p.T_upper)), 0, T_MAX))
        self.set("r_lower", _clamp(p.r_lower, 0, self._r_max))
        self.set("r_upper", _clamp(p.r_upper, 0, self._r_max))
        self.set("blur_sigma_x10", _clamp(int(round(p.blur_sigma * 10)), 0, BLUR_SIGMA_X10_MAX))
        self.set("blur_h", _clamp(p.blur_h, 0, BLUR_H_MAX))
        self.set("min_h_upper", _clamp(p.min_h_upper, 0, MIN_H_UPPER_MAX))
        self.set("y_top", _clamp(p.y_top, 0, self._H - 1))
        self.set("y_bottom", _clamp(self._H - 1 if p.y_bottom is None else p.y_bottom, 0, self._H - 1))
        self.set("cx", _clamp(p.resolved_cx(self._W_crop), 0, self._W_crop - 1))
        chan = p.channel if p.channel in CHANNEL_OPTIONS else "gray"
        self.set("channel", CHANNEL_OPTIONS.index(chan))

    def frame_idx(self) -> int:
        return self.get("frame") if self.has("frame") else self._init_frame_idx

    def set_frame_idx(self, i: int) -> None:
        if self.has("frame") and self._n_frames is not None:
            self.set("frame", _clamp(i, 0, self._n_frames - 1))

    def frame_step(self) -> int:
        return max(1, self.get("frame_step")) if self.has("frame_step") else self._init_frame_step

    def view_mode(self) -> dict[str, bool]:
        return {"show_specimen": bool(self.get("show_specimen")), "inline": bool(self.get("inline"))}

    def set_view_mode(self, *, show_specimen: bool | None = None, inline: bool | None = None) -> None:
        if show_specimen is not None:
            self.set("show_specimen", 1 if show_specimen else 0)
        if inline is not None:
            self.set("inline", 1 if inline else 0)

    def is_batch_requested(self) -> bool:
        return self._batch_requested

    def is_abort_requested(self) -> bool:
        return self._abort_requested or not self._alive

    def reset_requests(self) -> None:
        self._batch_requested = False
        self._abort_requested = False

    def request_batch(self) -> None:
        self._batch_requested = True

    def request_abort(self) -> None:
        self._abort_requested = True

    def update(self) -> None:
        """Pump the Tk event loop once (call every OpenCV loop iteration)."""
        if not self._alive:
            return
        try:
            self._refresh_summary()
            self.root.update()
        except tk.TclError:
            self._alive = False

    def is_alive(self) -> bool:
        return self._alive

    def destroy(self) -> None:
        if self._alive:
            try:
                self.root.destroy()
            except tk.TclError:
                pass
        self._alive = False

    def summary(self) -> str:
        """One-line summary of the current values (what the footer shows)."""
        p = self.get_params()
        parts = [
            p.polarity.replace("_", " "),
            f"T {p.T_lower:g}/{p.T_upper:g}", f"r {p.r_lower}/{p.r_upper}",
            f"sigma {p.blur_sigma:.1f}", f"blur_h {p.blur_h}", f"min_h {p.min_h_upper}",
            f"y {p.y_top}..{p.y_bottom}", f"cx {p.cx}", p.channel,
        ]
        if self.has("frame"):
            parts.append(f"frame {self.frame_idx()}")
        if self.has("frame_step"):
            parts.append(f"step 1/{self.frame_step()}")
        return "  ·  ".join(parts)

    def grab_screenshot(self) -> np.ndarray | None:
        """Screenshot of the panel window as a BGR ``uint8`` array (``None`` if unavailable).

        Uses ``PIL.ImageGrab``; the window must be mapped on screen. The Tk
        coordinates are rescaled to the physical screen when the process is
        not DPI-aware.
        """
        if not self._alive:
            return None
        try:
            from PIL import ImageGrab
        except ImportError:
            return None
        try:
            self.root.update()
            x = self.root.winfo_rootx()
            y = self.root.winfo_rooty()
            w = self.root.winfo_width()
            h = self.root.winfo_height()
            full = ImageGrab.grab(all_screens=True)
            sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
            vroot_w = max(sw, self.root.winfo_vrootwidth())
            k = full.size[0] / float(vroot_w) if vroot_w > 0 else 1.0
            if abs(k - 1.0) < 0.02:
                k = 1.0
            vx, vy = self.root.winfo_vrootx(), self.root.winfo_vrooty()
            left = int(round((x - vx) * k))
            top = int(round((y - vy) * k))
            right = int(round((x - vx + w) * k))
            bottom = int(round((y - vy + h) * k))
            box = (max(0, left), max(0, top), min(full.size[0], right), min(full.size[1], bottom))
            if box[2] <= box[0] or box[3] <= box[1]:
                return None
            img = np.asarray(full.crop(box).convert("RGB"))
            return np.ascontiguousarray(img[:, :, ::-1])
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _refresh_summary(self, force: bool = False) -> None:
        if self._summary_label is None:
            return
        text = self.summary()
        if force or text != self._last_summary:
            self._last_summary = text
            try:
                self._summary_label.config(text=text)
            except tk.TclError:
                pass

    def _on_close(self) -> None:
        self._alive = False
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _on_run_batch(self) -> None:
        self._batch_requested = True

    def _on_abort(self) -> None:
        self._abort_requested = True

    @staticmethod
    def _configure_styles(style: ttk.Style) -> None:
        style.configure("Dark.Horizontal.TScale", background=theme.BG, troughcolor=theme.SURFACE,
                        sliderthickness=14, sliderlength=18, borderwidth=0,
                        lightcolor=theme.SURFACE_HI, darkcolor=theme.SURFACE_HI)
        style.map("Dark.Horizontal.TScale", background=[("active", theme.SURFACE_HI)],
                  troughcolor=[("active", theme.SURFACE_HI)])

    @staticmethod
    def _flat_button(parent: tk.Widget, text: str, command: Callable[[], None]) -> tk.Button:
        return tk.Button(parent, text=text, bg=theme.SURFACE, fg=theme.SUBTEXT,
                         activebackground=theme.SURFACE_HI, activeforeground=theme.TEXT,
                         bd=0, padx=10, pady=3, font=theme.FONT_LABEL, relief="flat",
                         cursor="hand2", command=command)

    @staticmethod
    def _section(parent: tk.Widget, title: str, accent: str) -> tk.Frame:
        wrap = tk.Frame(parent, bg=theme.BG, pady=2)
        wrap.pack(fill="x", pady=(6, 0))
        header = tk.Frame(wrap, bg=theme.BG)
        header.pack(fill="x")
        bar = tk.Frame(header, bg=accent, width=4, height=18)
        bar.pack(side="left", padx=(0, 8))
        bar.pack_propagate(False)
        tk.Label(header, text=title, bg=theme.BG, fg=accent, font=theme.FONT_TITLE).pack(side="left")
        tk.Frame(wrap, bg=theme.SURFACE, height=1).pack(fill="x", pady=(4, 6))
        body = tk.Frame(wrap, bg=theme.BG)
        body.pack(fill="x")
        return body

    def _add_slider(self, parent: tk.Widget, name: str, label: str, lo: int, hi: int,
                    init: int, fmt: Formatter | None = None) -> None:
        if fmt is None:
            fmt = lambda v: f"{int(v):d}"  # noqa: E731
        var = tk.IntVar(value=int(init))
        self._vars[name] = var
        self._formatters[name] = fmt
        row = tk.Frame(parent, bg=theme.BG)
        row.pack(fill="x", pady=1)
        tk.Label(row, text=label, bg=theme.BG, fg=theme.TEXT, font=theme.FONT_LABEL,
                 width=20, anchor="w").pack(side="left")
        val = tk.Label(row, text=fmt(int(init)), bg=theme.BG, fg=theme.SUBTEXT,
                       font=theme.FONT_VALUE, width=8, anchor="e")
        val.pack(side="right")
        self._value_labels[name] = val
        ttk.Scale(row, from_=lo, to=hi, variable=var, orient="horizontal",
                  style="Dark.Horizontal.TScale",
                  command=lambda v, n=name: self._on_scale(n, v)).pack(side="left", fill="x", expand=True, padx=8)

    def _on_scale(self, name: str, value: str) -> None:
        try:
            iv = int(round(float(value)))
        except ValueError:
            return
        self._vars[name].set(iv)
        lbl = self._value_labels.get(name)
        if lbl is not None:
            lbl.config(text=self._formatters[name](iv))

    def _add_radio(self, parent: tk.Widget, name: str, label: str, options: Sequence[str],
                   init_index: int = 0, accent: str | None = None,
                   on_select: Callable[[int], None] | None = None) -> None:
        var = tk.IntVar(value=int(init_index))
        self._vars[name] = var
        self._radio_accents[name] = accent or theme.SECTION_GRADIENT
        if on_select is not None:
            self._radio_callbacks[name] = on_select
        row = tk.Frame(parent, bg=theme.BG)
        row.pack(fill="x", pady=3)
        tk.Label(row, text=label, bg=theme.BG, fg=theme.TEXT, font=theme.FONT_LABEL,
                 width=20, anchor="w").pack(side="left")
        wrap = tk.Frame(row, bg=theme.BG)
        wrap.pack(side="left", fill="x", expand=True, padx=8)
        buttons: list[tk.Button] = []
        for i, opt in enumerate(options):
            btn = tk.Button(wrap, text=opt, bg=theme.SURFACE, fg=theme.SUBTEXT,
                            activebackground=theme.SURFACE_HI, activeforeground=theme.TEXT,
                            bd=0, padx=10, pady=2, font=theme.FONT_LABEL, relief="flat", cursor="hand2",
                            command=lambda i=i: self._select_radio(name, i))
            btn.pack(side="left", padx=(0, 4))
            buttons.append(btn)
        self._radio_groups[name] = buttons
        self._refresh_radio(name)

    def _select_radio(self, name: str, index: int) -> None:
        self._vars[name].set(int(index))
        self._refresh_radio(name)
        cb = self._radio_callbacks.get(name)
        if cb is not None:
            try:
                cb(int(index))
            except Exception:
                pass  # a failing callback must not kill the panel

    def _refresh_radio(self, name: str) -> None:
        sel = int(self._vars[name].get())
        accent = self._radio_accents.get(name, theme.SECTION_GRADIENT)
        for i, btn in enumerate(self._radio_groups.get(name, [])):
            if i == sel:
                btn.configure(bg=accent, fg=theme.BG, activebackground=accent, activeforeground=theme.BG)
            else:
                btn.configure(bg=theme.SURFACE, fg=theme.SUBTEXT, activebackground=theme.SURFACE_HI,
                              activeforeground=theme.TEXT)


def headless() -> bool:
    """True when ``CYLVISION_HEADLESS`` is set (no window may be opened)."""
    return bool(os.environ.get("CYLVISION_HEADLESS"))


__all__ = [
    "PANEL_TITLE", "POLARITY_LABELS", "CHANNEL_OPTIONS", "FRAME_STEP_PRESETS", "FRAME_STEP_MAX",
    "T_MAX", "BLUR_SIGMA_X10_MAX", "BLUR_H_MAX", "MIN_H_UPPER_MAX", "HOTKEYS",
    "ControlsPanel", "headless",
]
