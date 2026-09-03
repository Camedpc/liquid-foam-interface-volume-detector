# liquid-foam-interface-volume-detector — internal engineering spec

This file is the contract shared by everyone working on the repo. It fixes
the package layout, the public API of every module, naming conventions and
the physical conventions (image axes, gradient sign, polarity). Read it
fully before writing code. It is NOT shipped to end users (README.md is).

## 0. Source material

The code is a clean re-organisation of an existing research pipeline
(beer-foam experiment, IPT 2026). The legacy code lives in the original
research pipeline folder (not shipped; read-only, never modify it). Its
location, like the run data, is passed to the scripts on the command line
and never hard-coded.

Legacy file → new home:

| legacy file | what to port | new module |
|---|---|---|
| `interface_tuner.py` (compute_gradient, apply_top_mask, apply_bottom_mask, detect_interfaces, make_panels, make_info_strip, draw_* helpers) | gradient core + rendering | `cylvision/detection/gradient.py`, `interfaces.py`, `panels.py` |
| `beer_lambert_density.detect_interfaces_swapped` | the sign-swap idea → becomes the `polarity` parameter | `cylvision/detection/interfaces.py` |
| `models.py` | 3 calibration models | `cylvision/calibration/models.py` |
| `calibration.py`, `volume_timeseries.collect_calibration_clicks` (lines ~478-820), `volume_timeseries.build_model_fn`, `fit_and_dump_calibration`, `calibration.show_verification` | click flow, JSON, check figure | `cylvision/calibration/clicks.py`, `store.py`, `check.py` |
| `magnifier.py` | loupe | `cylvision/magnifier.py` |
| `interface_tuner_video_wide.py` (open_video, seek_to, fit_to_screen, pick_calibration_frame, compose_wide_canvas, get_screen_size) + `volume_timeseries.pick_frame_with_title` | video IO + tuner UI | `cylvision/io.py`, `cylvision/ui/tuner.py`, `cylvision/ui/frame_picker.py` |
| `controls_panel.py` (Tk dark panel — keep ONLY the gradient / cylinder / view / sampling sections, drop every Beer-Lambert, concentration, dead-pixel, gamma, V_inf, collapse section) | control panel | `cylvision/ui/controls_panel.py` |
| `volume_timeseries.process_batch`, `_process_one_frame_legacy`, `write_csv`, `plot_curves`, `subsample_video_ffmpeg` (optional) | batch | `cylvision/pipeline/batch.py`, `plot.py` |
| `curvature_uncertainty.py`, `slide_curvature.py` | geometric (cylinder curvature) uncertainty + figures | `cylvision/uncertainty/curvature.py` |
| `interface_uncertainty_video.py` (steps 1-4 of its docstring), `interface_uncertainty_video_compare.compute_uncertainty` | empirical / method uncertainty | `cylvision/uncertainty/empirical.py` |
| `generate_uncertainty_legend.py`, `plot_run09_uncertainty.py` | budget figure ideas | `cylvision/uncertainty/budget.py` |
| `back_calibration.py`, `rectify_cylinder.py` | optional back-graduation clicks + focal-free model | `cylvision/calibration/back_clicks.py`, `cylvision/uncertainty/curvature.py` (focal-free branch) |

Everything Beer-Lambert (absorbance, concentration, I0, dead pixels, gamma,
V_inf, collapse detection, heatmaps) is OUT OF SCOPE. Do not port it.

## 1. Repository layout

```
liquid-foam-interface-volume-detector/
├── README.md                      end-user doc (English, very visual)
├── SPEC.md                        this file
├── LICENSE, pyproject.toml, requirements.txt, .gitignore
├── cylvision/
│   ├── __init__.py
│   ├── io.py                      unicode-safe image IO, video source
│   ├── magnifier.py               zoom loupe for pixel-accurate clicks
│   ├── calibration/
│   │   ├── models.py              PCHIP / poly2 / Möbius fits, best_model, build_model_fn
│   │   ├── clicks.py              interactive ROI + graduation clicks (OpenCV window)
│   │   ├── back_clicks.py         optional back-face graduation clicks
│   │   ├── store.py               Calibration dataclass, cylinder presets, JSON IO
│   │   └── check.py               verification figure (annotated frame + fit + residuals)
│   ├── detection/
│   │   ├── gradient.py            compute_gradient, apply_top_mask, apply_bottom_mask
│   │   ├── interfaces.py          DetectionParams, Polarity, InterfaceResult, detect_interfaces, detect_from_crop
│   │   └── panels.py              make_panels (highlight / gradient / thresholds), info strip, drawing helpers
│   ├── ui/
│   │   ├── theme.py               colour palette (dark) shared by Tk + OpenCV canvases
│   │   ├── controls_panel.py      Tk dark control panel (gradient-only)
│   │   ├── frame_picker.py        pick a frame in a video (start/end/calibration frame)
│   │   └── tuner.py               live tuner: video frame + panels + control panel → DetectionParams
│   ├── pipeline/
│   │   ├── run_store.py           run directory layout (crop.json, calib.json, params.json, meta.json, levels.csv)
│   │   ├── batch.py               process a video → per-frame rows (y_lower, y_upper, V_lower, V_total, spreads)
│   │   └── plot.py                V(t) figure with uncertainty bands
│   └── uncertainty/
│       ├── curvature.py           cylinder-projection geometry + curvature uncertainty + figures
│       ├── empirical.py           per-frame method uncertainty from per-column spread
│       └── budget.py              combine calibration / pixel / curvature / method → u_total(V), tables and figure
├── scripts/                       thin CLIs (argparse) around the package
│   ├── calibrate.py               video/image → calib.json + calibration_check.png
│   ├── tune.py                    open the live tuner on a video, save params.json
│   ├── run_batch.py               calibration + params → levels.csv + figure
│   ├── uncertainty_report.py      run dir → curvature figure + budget table/figure
│   └── make_readme_figures.py     regenerate every docs/images/*.png from real runs
├── docs/
│   ├── images/                    README screenshots (real data only)
│   └── uncertainty.md             full derivation of the uncertainty budget (English, LaTeX via GitHub math)
└── tests/                         pytest, synthetic images only (no video needed)
```

## 2. Conventions (mandatory)

- Python ≥ 3.10, `from __future__ import annotations`, type hints everywhere.
- **All comments, docstrings, UI labels, log messages, figure titles: English.**
- No mention of Claude, AI assistants, or private local data paths anywhere in
  the shipped code or docs. `scripts/make_readme_figures.py` takes the data
  root as a CLI argument (`--runs-root`, `--video`), never hard-coded.
- Module docstring at the top of every file explaining the physics/algorithm.
- No global state. Parameters travel in dataclasses (`DetectionParams`,
  `Calibration`, `CameraGeometry`).
- OpenCV BGR in memory; save figures with matplotlib (`Agg` backend inside
  library code — only scripts/ui may open windows).
- Unicode paths: always go through `cylvision.io.imread_unicode` /
  `imwrite_unicode` (numpy fromfile + cv2.imdecode/imencode).
- Windows key codes for arrows (`cv2.waitKeyEx`): keep the `a`/`d`/`0`
  fallbacks as in the legacy code.
- Keep functions pure where possible; rendering functions take arrays and
  return arrays.

## 3. Physical / image conventions

- Frame axes: `x` to the right, `y` DOWN (OpenCV). A graduated cylinder is
  vertical in the image; volume increases as `y` decreases.
- **Crop = full-height horizontal band** `frame[:, x_left:x_right+1]`.
  Hence a `y` inside the crop equals the `y` in the original frame, which is
  the `y` the calibration model maps to volume. `cx` is the cylinder axis
  column *inside the crop* (`cx = (x_right - x_left) // 2` by default).
- `y_top` / `y_bottom`: rows outside `[y_top, y_bottom]` have their gradient
  zeroed (glass rim at the top, base/table at the bottom). They mask the
  gradient, they do not crop.
- `Gy = Sobel_y(blur(gray))` = ∂I/∂y. Positive `Gy` = image gets brighter
  going DOWN.
- Two interfaces, named by position, not by substance:
  - **lower interface** = liquid/foam (beer/foam in the original experiment)
  - **upper interface** = foam/air
- **Polarity** (`Polarity = Literal["lower_darker", "lower_brighter"]`):
  - `"lower_darker"`: the phase below an interface is darker than the phase
    above it (front-lit cylinder on a black background: bright foam over
    dark beer). Lower interface → `Gy < 0` peak; upper interface → `Gy > 0`.
  - `"lower_brighter"`: the phase below is brighter (back-lit green screen:
    beer transmits light, foam scatters it and looks dark; air above the
    foam is bright). Lower interface → `Gy > 0`; upper interface → `Gy < 0`.
  - Implementation: `sign = -1 if polarity == "lower_darker" else +1`;
    `S = sign * Gy`; the lower interface is the per-column **argmax of S**
    validated by `S > T_lower`; the upper interface is found **bottom-up
    from y_lower** as the first row where `-S > T_upper`, with the
    connected-component height filter `min_h_upper`. This keeps the maths of
    the legacy `detect_interfaces` unchanged (it was written for
    `lower_darker` with `-Gy` conventions) and makes the swap explicit.
  - The two interfaces ALWAYS have opposite gradient signs, whatever the
    lighting. One parameter flips both.
- Radial bands: averaging over columns `|x - cx| <= r_lower` (resp.
  `r_upper`). `r >= W//2` means the whole crop width.

## 4. Public API (signatures are binding)

### `cylvision/io.py`
```python
def imread_unicode(path: Path | str) -> np.ndarray | None
def imwrite_unicode(path: Path | str, img: np.ndarray) -> bool
def open_video(path: Path | str) -> tuple[cv2.VideoCapture, int, float]   # cap, n_frames, fps
def read_frame(cap: cv2.VideoCapture, idx: int) -> np.ndarray | None
class VideoSource:            # context-manager wrapper
    path, n_frames, fps, width, height
    def frame(self, idx: int) -> np.ndarray | None
    def frames(self, start, stop, step=1) -> Iterator[tuple[int, np.ndarray]]
```

### `cylvision/magnifier.py`
Port `make_magnifier`, `overlay_magnifier`, `draw_crosshair` unchanged (zoom 8).

### `cylvision/calibration/models.py`
```python
ModelName = Literal["pchip", "poly2", "mobius"]
def fit_calibration_models(y_px, V_ml) -> dict[str, dict]      # {"pchip": {"fn": callable}, "poly2": {"fn", "coef"}, "mobius": {"fn", "popt", "pcov"}}
def evaluate_residuals(models, y_px, V_ml) -> dict[str, np.ndarray]
def best_model(residuals, rms_threshold_ml: float = 0.3) -> ModelName
def build_model_fn(calib: "Calibration") -> Callable[[ArrayLike], np.ndarray]   # y_px -> V_mL using calib.recommended_model
def slope_ml_per_px(model_fn, y_px: float, h: float = 1.0) -> float          # |dV/dy| central difference
```

### `cylvision/calibration/store.py`
```python
@dataclass
class CylinderSpec:
    name: str                 # "1000 mL"
    capacity_ml: float
    graduations_ml: list[float]     # graduations to click (e.g. 100..1000 step 100)
    radius_cm: float | None         # inner radius, used by the curvature model
PRESETS: dict[str, CylinderSpec]    # "50", "1000" (values from legacy EPROUVETTES + R_PHYS_CM_DEFAULT = {"50": 1.35, "1000": 3.20})

@dataclass
class Calibration:
    image_ref: str
    image_size: tuple[int, int]            # (W, H)
    cylinder: CylinderSpec
    x_left: int; x_right: int; y_top: int; y_bottom: int
    graduations_ml: list[float]; graduations_px_y: list[float]; graduations_px_x: list[float]
    skipped_ml: list[float]
    poly2_coef: list[float]; mobius_popt: list[float]; mobius_pcov: list[list[float]]
    residuals_ml: dict[str, list[float]]
    recommended_model: ModelName
    back_px_y: list[float | None] | None = None   # optional back-face clicks
    back_px_x: list[float | None] | None = None
    back_frame_idx: int | None = None
    def to_json(self, path) / @classmethod from_json(cls, path)
    def crop(self) -> tuple[int, int, int, int]      # x_left, x_right, y_top, y_bottom
    def cx(self) -> int                               # axis column inside crop
def fit_calibration(clicks: dict, spec: CylinderSpec, image_ref: str, image_size) -> Calibration
```
JSON must stay compatible with the legacy `calib.json` keys where they
exist (`graduations_ml`, `graduations_px_y`, `graduations_px_x`, `roi_x`,
`poly2_coef`, `mobius_popt`, `mobius_pcov`, `residuals_ml`,
`recommended_model`, `graduations_px_y_back`, `graduations_px_x_back`,
`back_calibration_frame`) so that `Calibration.from_json` can also read a
legacy file (`from_legacy_json` helper fine). Extra keys: `x_left`,
`x_right`, `y_top`, `y_bottom`, `cylinder` (spec dict).

### `cylvision/calibration/clicks.py`
```python
def collect_calibration_clicks(img: np.ndarray, graduations_ml: list[float], *, window_title: str = ...) -> dict | None
    # returns {"x_left","x_right","y_top","y_bottom","V_ml":[...],"y_px":[...],"x_px":[...],"skipped_ml":[...]} or None if aborted
```
Port the full legacy interaction (mouse, Shift = damped precision mode,
arrow nudge with 400 ms lock, Space/click = place, `f` fullscreen, `s`
skip, `z` undo, Enter validate, Esc abort, magnifier overlay, coloured
prompts: green vertical edges, cyan top/bottom, orange graduations).

### `cylvision/calibration/check.py`
```python
def render_verification_figure(img_bgr, calib: Calibration, out_png: Path) -> None
```
Three panels like the legacy figure (annotated original with every
graduation line predicted by the model, the crop, V(y) with 3 models +
residual bars). English labels.

### `cylvision/detection/gradient.py`
```python
def compute_gradient(crop_gray: np.ndarray, blur_sigma: float, blur_h: int = 0) -> np.ndarray   # float32 Gy
def apply_top_mask(Gy, y_top: int) -> np.ndarray
def apply_bottom_mask(Gy, y_bottom: int | None) -> np.ndarray
def to_gray(crop_bgr, channel: Literal["gray","B","G","R"] = "gray") -> np.ndarray
```

### `cylvision/detection/interfaces.py`
```python
Polarity = Literal["lower_darker", "lower_brighter"]
POLARITIES: tuple[str, ...]

@dataclass
class DetectionParams:
    polarity: Polarity = "lower_darker"
    T_lower: float = 40.0        # gradient threshold, lower (liquid/foam) interface
    T_upper: float = 47.0        # gradient threshold, upper (foam/air) interface
    r_lower: int = 80            # half-width of the averaging band around cx (px)
    r_upper: int = 80
    blur_sigma: float = 5.0      # Gaussian sigma (px)
    blur_h: int = 38             # horizontal box-filter width (px), 0 = off
    min_h_upper: int = 3         # connected-component min height for the upper interface (px), 0/1 = off
    y_top: int = 0               # gradient mask (rows above are ignored)
    y_bottom: int | None = None  # gradient mask (rows below are ignored)
    cx: int | None = None        # axis column inside the crop, None = W//2
    channel: str = "gray"
    def to_dict(self) / from_dict(d)          # json-friendly; accept legacy keys (T_pos/T_neg/r_pos/r_neg/min_h_pos) in from_dict with polarity guessed = "lower_darker"

@dataclass
class InterfaceResult:
    y_lower: float | None          # mean row of the lower interface (crop == frame coordinates)
    y_upper: float | None
    lower_rows: np.ndarray         # per-column argmax row (int32, W)
    lower_valid: np.ndarray        # bool (W): above threshold AND inside band
    upper_rows: np.ndarray
    upper_valid: np.ndarray
    band_lower: tuple[int, int]    # x_lo, x_hi inclusive
    band_upper: tuple[int, int]
    n_lower: int; n_upper: int
    std_lower: float | None        # std of lower_rows over valid columns
    upper_lowest: int | None; upper_low_blob: float | None   # legacy extras (keep)
    S: np.ndarray                  # the signed gradient used (sign * Gy, masked)

def detect_interfaces(Gy: np.ndarray, params: DetectionParams) -> InterfaceResult
def detect_from_crop(crop_bgr_or_gray: np.ndarray, params: DetectionParams) -> tuple[InterfaceResult, np.ndarray]   # (result, Gy)
```
Behaviour identical to legacy `detect_interfaces` for `lower_darker`
(argmin Gy ↔ argmax S). Docstring must explain the two-pass ordered
detection (lower first, then upper bottom-up from y_lower) and why
(porous foam → micro-gradients would fool a raw argmax), the
connected-component filter, the bands, and the polarity.

### `cylvision/detection/panels.py`
```python
def make_panels(crop_bgr, Gy, result: InterfaceResult, params: DetectionParams, *, panel_w: int | None = None) -> np.ndarray
    # horizontal stack: [highlight (crop + coloured overlays)] [signed gradient heatmap] [threshold mask +/-] — each with a label strip on top
def make_info_strip(width: int, result, params, model_fn=None, frame_idx=None, t_sec=None, fps=None) -> np.ndarray
def make_specimen_panel(frame_bgr, calib_crop, result, ...) -> np.ndarray     # full frame with the interface markers, brace between the two interfaces
def annotate_interfaces(img_bgr, result, params, *, zoom: float = 1.0, offset_x: int = 0) -> np.ndarray   # dots per column + mean lines + band markers (colours from theme)
```
Colours from `cylvision.ui.theme` (red = lower interface, cyan/teal =
upper interface, orange = band markers, yellow/cyan dashed = bounds).

### `cylvision/ui/theme.py`
Catppuccin-Mocha-like dark palette as in legacy `controls_panel.py`
(`BG`, `SURFACE`, `TEXT`, `SUBTEXT`, section colours: cylinder = mauve,
gradient = teal, view = blue, sampling = peach) + BGR tuples for OpenCV.

### `cylvision/ui/controls_panel.py`
```python
class ControlsPanel:
    def __init__(self, *, params: DetectionParams, H: int, W_crop: int, n_frames: int | None, init_frame_idx: int = 0,
                 include_sampling: bool = True, show_batch_button: bool = True, two_columns: bool = False)
    def update(self) -> None            # pump Tk events once per cv2 loop iteration
    def is_alive(self) -> bool
    def get_params(self) -> DetectionParams
    def set_params(self, params: DetectionParams) -> None
    def frame_idx(self) -> int ; def set_frame_idx(self, i) ; def frame_step(self) -> int
    def view_mode(self) -> dict          # {"show_specimen": bool, "inline": bool}
    def is_batch_requested(self) / is_abort_requested(self)
    def toggle_layout(self)
```
Sections and widgets (every one documented in README):
- VIEW: radio `specimen panel on/off`, radio `inline (1 row) / stacked`.
- CYLINDER (mauve): sliders `y_top`, `y_bottom`, `cx`, `frame` (video only).
- GRADIENT (teal): radio `polarity` (lower darker / lower brighter) — first
  widget of the section; sliders `T_lower`, `T_upper`, `r_lower`,
  `r_upper`, `blur_sigma ×10`, `blur_h`, `min_h_upper`; radio `channel`
  (gray / G / R / B).
- SAMPLING (peach): slider `frame_step` 1..60 + radio presets (all, 1/2, 1/3, 1/5, 1/10).
- Buttons: `Run batch`, `Abort`, `⇄ 1/2 columns`, plus a footer showing the
  current values as a one-line summary.

### `cylvision/ui/frame_picker.py`
```python
def pick_frame(cap, n_frames, fps, *, title: str, init_idx: int = 0) -> int | None
```
Trackbar + arrows/`a`/`d`/`0` nav, `Enter` validate, `Esc` abort; screen-fit.

### `cylvision/ui/tuner.py`
```python
def run_tuner(video: VideoSource, calib: Calibration, params: DetectionParams, *, model_fn=None, start_idx: int = 0, two_columns: bool = False) -> tuple[DetectionParams, int, bool]
    # returns (params, frame_step, batch_requested)
def compose_canvas(frame_bgr, calib, params, model_fn, *, show_specimen, inline, screen_size) -> np.ndarray   # pure, used by both the tuner and make_readme_figures
```

### `cylvision/pipeline/run_store.py`
```python
@dataclass
class RunDir:     # <root>/<video_stem>/
    root: Path
    calib_path, params_path, meta_path, csv_path, figure_path, check_png
    def save_params(self, params, frame_step) / load_params()
def find_run(root: Path, video_path: Path) -> RunDir
```

### `cylvision/pipeline/batch.py`
```python
CSV_COLUMNS = ["frame_idx", "t_sec", "y_lower_px", "std_lower_px", "n_lower", "y_upper_px", "n_upper",
               "V_lower_mL", "V_total_mL", "V_foam_mL", "u_method_lower_mL", "u_method_upper_mL"]
def process_frame(frame_bgr, calib, params, model_fn, *, tol_ml: float = 7.0) -> dict
def process_video(video: VideoSource, calib, params, model_fn, *, start: int, end: int, frame_step: int = 1, progress: bool = True, on_row=None) -> list[dict]
def write_csv(rows, path) ; def read_csv(path) -> list[dict]
```
`V_lower` = model_fn(y_lower) (liquid volume), `V_total` = model_fn(y_upper)
(liquid + foam), `V_foam = V_total − V_lower`. Rows for skipped frames are
NaN rows (keeps time alignment). `u_method_*` comes from
`cylvision.uncertainty.empirical.method_uncertainty`.

### `cylvision/pipeline/plot.py`
```python
def plot_levels(rows, *, title: str, out_png: Path, budget: "UncertaintyBudget | None" = None) -> None
```
V_lower(t), V_total(t), V_foam(t) with shaded ±u bands when a budget is given.

### `cylvision/uncertainty/curvature.py`
```python
@dataclass
class CameraGeometry:
    R_cm: float; a_px: float; rho: float; D_cm: float; f_px: float; cy_px: float; alpha_rad: float
    cx_px: float
    source: Literal["no_tilt", "mobius_tilt", "focal_free"]
def geometry_from_calibration(calib: Calibration, *, alpha_deg: float | None = None, use_back_clicks: bool = True) -> CameraGeometry
def project_graduation(theta: np.ndarray, v_click_px: float, geom: CameraGeometry) -> tuple[np.ndarray, np.ndarray]   # (u_px, v_px) of the horizontal circle at level v_click
def curvature_delta_px(v_click_px: float, geom: CameraGeometry, r_px: int) -> float     # full vertical span of the arc inside |u - cx| <= r_px
def curvature_uncertainty_ml(calib, geom, r_px, model_fn) -> list[dict]   # per graduation: V, v_click, delta_px, delta_ml, u_ml = delta_ml/(2*sqrt(3))
def curvature_uncertainty_fn(calib, geom, r_px, model_fn) -> Callable[[float], float]   # V -> u_curvature (interpolated)
def render_curvature_figure(frame_bgr, calib, geom, r_px, model_fn, out_png, *, style="light") -> None
    # 4 panels: (1) frame crop with projected ellipses per graduation + band; (2) u_curvature(V) with Δ full-span wedge and camera horizon;
    # (3) top-view schematic (circle, camera, visible arc, band cone); (4) side-view schematic (camera, pitch α, rays to graduations).
def render_cylinder_schematic(geom, calib, r_px, out_png) -> None    # standalone clean 2-panel schematic (top view + side view) for the README
```
Maths: port the legacy docstring (pinhole camera, tilt α, Möbius link,
`v(θ,V) − v_click(V) = f R (1−cosθ) B / (D0 Dθ)`, no-tilt inversion from
R and the calibration slope, focal-free inversion when back clicks
exist). Δ = vertical span of the arc over the band; u = Δ / (2√3)
(uniform-distribution half-width).

### `cylvision/uncertainty/empirical.py`
```python
def method_uncertainty(rows_px: np.ndarray, valid: np.ndarray, y_mean: float, model_fn, *, tol_ml: float = 7.0) -> dict
    # {"y_up": float, "y_lo": float, "V_up": ..., "V_lo": ..., "range_ml": ..., "u_ml": range/(2*sqrt(3)), "n_kept": int}
    # keep only columns whose V(y) is within ±tol_ml of V(y_mean) (outlier guard), then min/max
def summarize_method_uncertainty(rows: list[dict], key="u_method_lower_mL") -> dict   # mean/median/p95 over a run
```

### `cylvision/uncertainty/budget.py`
```python
@dataclass
class UncertaintyBudget:
    u_calibration_ml: float        # RMS of the recommended model residuals (or the reading resolution of the scale/√12 if larger)
    u_pixel_ml_fn: Callable        # V -> slope(V) * 1 px / sqrt(12)
    u_curvature_ml_fn: Callable    # from curvature.curvature_uncertainty_fn
    u_method_ml: float             # run-level summary (median) of the empirical spread
    def total(self, V: float) -> float      # quadrature sum
    def table(self, V_values) -> list[dict]
def build_budget(calib, geom, params, model_fn, rows=None) -> UncertaintyBudget
def render_budget_figure(budget, V_range, out_png) -> None       # stacked contributions vs V
def budget_markdown_table(budget, V_values) -> str
```

## 5. Scripts

Each script: `argparse`, `--help` in English, exit codes, no hard-coded
paths. `make_readme_figures.py --video <path> --run-dir <dir> --out docs/images`
must regenerate every image referenced by README.md from real data, using
only public package functions (nothing interactive except an optional
`--panel-screenshot` flag that opens the Tk panel briefly and grabs it).

## 6. Tests (pytest, no video, no display)

- synthetic crop generator: 3 horizontal bands (air / foam / liquid) with
  noise, both polarities → `detect_from_crop` finds both interfaces within
  ±2 px, for both polarities, and swapping the polarity on the same image
  fails (asserts the parameter matters).
- models: exact PCHIP at clicks, poly2/mobius residuals small on a synthetic
  tilted projection, `build_model_fn` round trip through JSON.
- `DetectionParams.from_dict` on a legacy params.json dict.
- curvature: Δ = 0 at the camera horizon, symmetric growth away from it,
  u = Δ/(2√3).
- empirical: known per-column rows → expected range.

## 7. README.md (written last, by a dedicated pass)

English, GitHub-flavoured, math via `$…$`. Sections: hero image → what it
does → install → 60-second quick start → the 3 tools (calibrate / tune /
batch) each with real screenshots → **control panel, parameter by
parameter** (one row per widget: name, what it does, when to change it,
before/after image) → polarity explained with the two real lighting setups
→ output files → **uncertainty budget** (calibration, pixel, curvature with
cylinder schematics, empirical method) with formulas and figures → tests →
license. Every image in `docs/images/` comes from a real run.
