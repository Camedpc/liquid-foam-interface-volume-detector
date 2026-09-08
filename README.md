# liquid-foam-interface-volume-detector

**Point a camera at a graduated cylinder; get the liquid level, the foam level and their volumes, frame by frame, with an uncertainty budget.**

<p align="center"><img src="docs/images/hero.png" width="100%"></p>

*Raw frame → detected, on the two lighting setups. The tool finds the **liquid/foam interface** (red line) and the **foam/air interface** (teal line), shades the foam band between them (peach) and the liquid below (blue), and converts both rows to volumes through the printed scale — the readouts next to the braces. Left pair: back-lit green screen, frame 108 (t = 172 s after the start of the pour) — foam 431 mL, liquid 222 mL, total 653 mL. Right pair: front-lit black background, frame 2472 (t = 124 s) — foam 581 mL, liquid 149 mL, total 730 mL.*

<p align="center"><img src="docs/images/detection_timeline.gif" width="34%"></p>

*The green-screen run through the detector, from the pour to after the foam collapse (15 frames, 1 frame = 60 source frames, times from the start of the pour): the readout in the header gives t and the three volumes of every frame; the static strip below shows the same frames for viewers whose browser does not play GIFs.*

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests: pytest](https://img.shields.io/badge/tests-pytest%20%C2%B7%2078%20passed-brightgreen.svg)](#tests)
[![OpenCV](https://img.shields.io/badge/built%20with-OpenCV%20%C2%B7%20NumPy%20%C2%B7%20SciPy%20%C2%B7%20Matplotlib%20%C2%B7%20Pillow-lightgrey.svg)](requirements.txt)

## The foam, over time

<p align="center"><img src="docs/images/timeline_green.png" width="100%"></p>

*Seven frames of the green-screen run (1 frame = 60 source frames; t = 0 at the start of the pour, sub frame 22). While the beer is still being poured (frame 30, t = 16 s) only the foam top is visible; at the foam peak (frame 39, t = 34 s) the band is 658 mL over 83 mL of liquid. It then shrinks to 431 mL (t = 172 s), 305 mL (356 s), 259 mL (557 s) and 200 mL (11.9 min) while the liquid drains out of it and settles at 257 mL; between frames 380 and 400 (12.6 min) an 83 mL slab of foam collapses and the band drops to 117 mL. Every panel is the specimen view of the tuner: the two flat mean lines with their triangles at the cylinder walls, the shaded foam and liquid zones, the braces and the readouts.*

Every overlay in this README uses the same code: **red** = lower interface (liquid/foam), **teal** = upper interface (foam/air), **peach wash** = foam, **blue wash** = liquid. The lines are drawn at the output resolution (3 px flat, a 1 px shadow underneath) so they stay crisp whatever the size of the figure; small dots along each line are the per-column detections that the mean line averages. The readouts use the TrueType fonts shipped with matplotlib (DejaVu Sans for the zone words, DejaVu Sans Mono for the numbers).

## Background

This project was built for the **internal selection of the École Polytechnique team for the IPT** (International Physicists' Tournament). Each candidate had to present, in six minutes, an experimental study of one of the tournament problems, with a real uncertainty analysis. The problem chosen here was *Beer Foam*: how the volumes of beer and foam evolve in a 1000 mL graduated cylinder during and after a pour. Reading those two levels by eye on hours of video was not an option, so this tool reads the printed scale and the two interfaces automatically, frame by frame, and quantifies how much each reading can be trusted.

Nothing in it is specific to beer: any vertical graduated cylinder filmed by a fixed camera, with one or two horizontal interfaces (liquid/foam, liquid/air, two immiscible liquids), can be read the same way.

## What it does

- **Calibrates the scale from clicks** — click the graduations once on a still frame; three row-to-volume models (PCHIP, quadratic, Möbius) are fitted and checked.
- **Detects two interfaces per frame** from the vertical intensity gradient $G_y = \partial I/\partial y$: the *lower* interface (liquid/foam) and the *upper* one (foam/air), each averaged over a band of columns around the cylinder axis.
- **Batches a whole video** into a CSV (row, volume and per-frame spread of each interface) and a $V(t)$ figure, one frame in $K$.
- **Quantifies the uncertainty** of every reading: calibration, pixel quantisation, cylinder curvature (a graduation is a circle, not a line) and the empirical spread of the detector, combined in quadrature.

```mermaid
flowchart LR
    A["calibrate.py<br/>clicks → calib.json"] --> B["tune.py<br/>live tuner → params.json"]
    B --> C["run_batch.py<br/>video → levels.csv + figure.png"]
    C --> D["uncertainty_report.py<br/>budget table + figures"]
```

## Install

```bash
git clone https://github.com/Camedpc/liquid-foam-interface-volume-detector.git
cd liquid-foam-interface-volume-detector
pip install -r requirements.txt      # opencv-python, numpy, scipy, matplotlib, pillow
pip install -e .                     # optional: import cylvision from anywhere
```

Python 3.10 or newer. The interactive tools open OpenCV and Tkinter windows; the library and the tests do not need a display.

## Quick start

```bash
# 1. click the graduations on one frame  ->  runs/pour/calib.json + calibration_check.png
python scripts/calibrate.py --video pour.mp4 --cylinder 1000 --back --out runs/pour

# 2. tune the detector live               ->  runs/pour/params.json
python scripts/tune.py --video pour.mp4 --run-dir runs/pour --polarity lower_brighter

# 3. process the video                    ->  runs/pour/levels.csv + meta.json + figure.png (±u bands)
python scripts/run_batch.py --video pour.mp4 --run-dir runs/pour --step 5

# 4. (optional) uncertainty budget         ->  runs/pour/uncertainty_*.png + uncertainty.json
python scripts/uncertainty_report.py --run-dir runs/pour --video pour.mp4 --frame 108
```

If the video was already sub-sampled (for example one frame in 60 of a long recording, keeping the source fps in the metadata), pass `--frame-scale 60` to `tune.py` and `run_batch.py` so that the frame counter and `t_sec` show the source time.

## How the detection works

The detector only looks at the **sign of $G_y$** along the cylinder axis. Image rows grow downwards, so $G_y > 0$ where the picture gets brighter going down. Which sign belongs to which interface depends on the lighting, and one parameter, `polarity`, tells it:

<table>
<tr>
<td align="center" width="70%"><img src="docs/images/setup_black_background.png" width="100%"><br><em>Front-lit, black background (frame 2472, t = 124.2 s): bright white foam over dark beer, dark air above.</em></td>
<td align="center" width="30%"><img src="docs/images/setup_green_screen.png" width="60%"><br><em>Back-lit green screen (frame 108, t = 172 s after the pour): the beer transmits the light, the foam scatters it and looks dark. The recording is a 470 × 1766 crop around the cylinder.</em></td>
</tr>
</table>

| `polarity` | setup | lower interface (liquid/foam) | upper interface (foam/air) |
|---|---|---|---|
| `lower_darker` | front-lit, black background | dark beer under bright foam → $G_y < 0$ | bright foam under dark air → $G_y > 0$ |
| `lower_brighter` | back-lit green screen | bright beer under dark foam → $G_y > 0$ | dark foam under bright air → $G_y < 0$ |

The two interfaces always have opposite signs, so the code works on $S = \pm G_y$ (sign $-1$ for `lower_darker`, $+1$ for `lower_brighter`): the lower interface is always a positive peak of $S$ (drawn in red), the upper one a negative peak (drawn in teal).

**Two ordered passes.** The lower interface is found first: per column, the row of the maximum of $S$, validated by $S > T_\text{lower}$, averaged over $|x - c_x| \le r_\text{lower}$. The upper interface is then searched **bottom-up from $y_\text{lower}$**: the first row where $-S > T_\text{upper}$, averaged over $|x - c_x| \le r_\text{upper}$. Foam is porous and every bubble edge creates a small gradient of the "upper" sign, so a raw per-column argmax would pick a random bubble somewhere inside the foam; scanning upwards from the liquid returns the *lowest* transition above threshold, which is the foam top. A **connected-component filter** (`min_h_upper`) removes the isolated 1–2 px bubble edges from the teal mask before the scan, keeping only blobs at least that many rows tall.

<table>
<tr>
<td align="center" width="50%"><img src="docs/images/polarity_lower_darker.png" width="88%"><br><em><code>lower_darker</code> on the black run: red blob at the beer/foam interface (y = 845.7 px, 161/161 columns, 148.6 mL), teal blobs at the foam top (y = 159.8 px, 203/227 columns, 729.9 mL); the foam band (581 mL) is shaded between them.</em></td>
<td align="center" width="50%"><img src="docs/images/polarity_lower_brighter.png" width="80%"><br><em><code>lower_brighter</code> on the green run: same colours, flipped physics. Lower y = 1258.2 px ± 1.7 px (221.8 mL, 267/343 columns), upper y = 431.7 px (653.0 mL, 237/343 columns), foam 431 mL. The teal blobs above the foam top are the bubbly upper layer of the foam; the bottom-up scan stops on the lowest one.</em></td>
</tr>
</table>

The three analysis panels shown above and in the tuner:

| panel | what it shows |
|---|---|
| **Highlight** | the crop with the shaded foam / liquid zones and their volumes, the per-column detections (dots), the mean rows (red = lower, teal = upper), the band edges (orange/yellow), the axis `cx` (thin mauve line) and the `y_top` / `y_bottom` masks (magenta) |
| **Signed gradient** | the heat map of $S = \pm G_y$ after Gaussian and horizontal smoothing: red = lower-interface sign, teal = upper-interface sign |
| **Threshold mask** | $S > T_\text{lower}$ in red and $-S > T_\text{upper}$ in teal — what the detector is allowed to pick |

<p align="center"><img src="docs/images/polarity_wrong.png" width="90%"></p>

*The same green-screen frame with the wrong polarity (left) and the right one (right). With `lower_darker` the detector locks on the wrong sign: nothing passes the "lower" threshold (0 columns) and the real beer/foam interface is reported as the "upper" one (y = 1267.1 px, 217 mL, 341 columns) — no foam band, no readout. With `lower_brighter`: liquid 222 mL, total 653 mL, foam 431 mL.*

## Tool 1 — Scale calibration (`scripts/calibrate.py`)

One still frame, a handful of clicks. The script asks, in order, for the **left** and **right** edges of the cylinder (the crop band), the **top** and **bottom** rows (`y_top` / `y_bottom`, the gradient masks), then every graduation of the preset. Pixel-accurate pointing is the whole point, so the cursor is a *logical* cursor with a ×8 loupe.

<p align="center"><img src="docs/images/calibration_clicks.png" width="45%"></p>

*Prompt 8 of 14 on the green run: the four ROI clicks are placed (left 73, right 415, y_top 4, y_bottom 1534) and the 100, 200 and 300 mL graduations are marked in orange; the cursor sits at (238, 915) on the 400 mL ring, magnified ×8 in the top-left loupe. The 700–1000 mL rings lie above the frame and are skipped with `s`.*

Three models are fitted on the clicks $(y_i, V_i)$:

| model | form | behaviour |
|---|---|---|
| PCHIP | monotone cubic interpolation | exact at the clicks, follows any irregularity of the printed scale |
| poly2 | $V = a y^2 + b y + c$ | smooth, absorbs one noisy click |
| Möbius | $V = \dfrac{a y + b}{c y + 1}$ | the exact pinhole + tilt law; $c = 0$ means no camera pitch |

`best_model` keeps **poly2 when its RMS residual is ≤ 0.3 mL**, otherwise **PCHIP**. On both runs below poly2 lands at 0.43–0.56 mL, so the calibration honours the actual rings with PCHIP.

<table>
<tr>
<td align="center" width="50%"><img src="docs/images/calibration_check_green.png" width="100%"><br><em>Green run (470 × 1766 frame, crop 343 × 1766): 6 clicks from 100 to 600 mL (the rings above are out of frame). RMS: PCHIP 0.00, poly2 0.43, Möbius 0.44 mL.</em></td>
<td align="center" width="50%"><img src="docs/images/calibration_check_black.png" width="100%"><br><em>Black run (1920 × 1080 frame, crop 237 × 1080): 8 clicks from 100 to 800 mL. RMS: PCHIP 0.00, poly2 0.56, Möbius 0.60 mL.</em></td>
</tr>
</table>

The three panels of `calibration_check.png`: the annotated frame with every graduation line predicted by the recommended model (green = clicks, orange = 50 mL marks, grey = 10 mL marks), the crop band alone, and $V(y)$ for the three models with their residuals.

<details>
<summary><b>Click keys, <code>calib.json</code> keys and cylinder presets</b></summary>

| key / action | effect |
|---|---|
| mouse | moves the cursor 1:1 |
| **Shift** + mouse | damped pointer: one logical pixel every few mouse pixels |
| arrows | nudge by exactly 1 px (the mouse is ignored for 400 ms afterwards) |
| **Space** or left click | place the point at the logical cursor |
| `f` | toggle a 1:1 fullscreen window |
| `s` | skip a graduation that lies outside the frame |
| `z` | undo the last point |
| **Enter** | validate (all prompts answered, at least three graduations) |
| **Esc** | abort |

`calib.json` keys: `image_ref`, `image_size`, `cylinder` (spec), `x_left`, `x_right`, `y_top`, `y_bottom`, `graduations_ml`, `graduations_px_y`, `graduations_px_x`, `skipped_ml`, `poly2_coef`, `mobius_popt`, `mobius_pcov`, `residuals_ml`, `recommended_model`, and with `--back`: `graduations_px_y_back`, `graduations_px_x_back`, `back_calibration_frame`. Legacy files with a sibling `crop.json` are read transparently.

**Presets** (`cylvision/calibration/store.py`):

| preset | graduations to click (mL) | inner radius | scale step |
|---|---|---|---|
| `50` | 5, 10, 20, 30, 40, 50 | 1.35 cm | 1 mL |
| `1000` | 100, 200, …, 1000 | 3.20 cm | 10 mL |

Any other cylinder: `--cylinder custom --graduations 25,50,75,100 --radius-cm 1.8 --name "100 mL"`. The radius only feeds the curvature term of the uncertainty budget; `--back` (click the same rings on the back face) makes the budget independent of it.

</details>

## Tool 2 — Interface tuner (`scripts/tune.py`)

The tuner shows one frame through the detector and lets you move every parameter while watching the result: the specimen panel (full frame, shaded zones, braces with the volumes), the three analysis panels and the info strip.

<p align="center"><img src="docs/images/tuner_canvas_inline.png" width="100%"></p>

*Inline layout on the green run (frame 108, t = 172.2 s): specimen, highlight, signed gradient, threshold mask and the info strip. Liquid 221.8 mL, liquid + foam 653.0 mL, foam 431.2 mL. The recording has no margin right of the cylinder, so the specimen panel drops its braces and readouts here; the highlight panel carries them inside the zones.*

<p align="center"><img src="docs/images/tuner_canvas_stacked.png" width="100%"></p>

*Stacked layout on the black run (frame 2472): specimen and highlight on the first row, gradient and mask on the second, info strip below. Liquid 148.6 mL, liquid + foam 729.9 mL, foam 581.3 mL.*

The **info strip** repeats the parameters and prints, for each interface, the mean row, its per-column spread, the number of valid columns in the band, and the volumes `V_liquid`, `V_total`, `V_foam` through the calibration.

### Control panel

<p align="center"><img src="docs/images/control_panel.png" width="90%"></p>

*The Tk panel in two-column layout, loaded with the parameters of an earlier green-screen recording (`lower brighter`, T 16/16, r 64/64, σ 5.7, blur_h 38, min_h 3, y 5..1869, cx 157, gray, every frame). The run used in the figures below is tuned to T 40/8, r 171/171, min_h 15, y 4..1534, cx 171.*

**▶ Run batch** saves `params.json` (parameters + `frame_step`) and prints the batch command; **✖ Abort** leaves the run directory untouched; **⇄ 1/2 columns** rebuilds the panel in one or two columns. The footer shows the hotkeys of the OpenCV window and a one-line summary of the current values.

<details>
<summary><b>Every widget of the panel (VIEW, CYLINDER, GRADIENT, SAMPLING) and the hotkeys</b></summary>

#### VIEW (blue)

| widget | what it does | when to change it |
|---|---|---|
| `specimen panel` — radio [off \| **on**] | shows the full frame with the two interfaces, the shaded zones and the braces | turn off on a small screen to give the analysis panels more room |
| `panels` — radio [stacked \| **inline (1 row)**] | one row of panels, or two rows | stacked suits a short, wide crop (landscape video) |

#### CYLINDER (mauve)

| widget | range · default | what it does | when to change it |
|---|---|---|---|
| `y_top` | 0..H−1 · from `calib.json` | zero the gradient above this row (glass rim) | if the rim or the pouring stream shows up in the mask |
| `y_bottom` | 0..H−1 · from `calib.json` | zero the gradient below this row (base, table) | if the base edge shows up in the mask |
| `cx` | 0..W_crop−1 · W_crop // 2 | axis column *inside the crop*, centre of both bands | if the crop is not centred on the cylinder |
| `frame` (video only) | 0..N−1 | the frame shown | to check the parameters on a pour, a plateau, a collapse |

#### GRADIENT (teal)

| widget | range · default | what it does | when to change it |
|---|---|---|---|
| `polarity` — radio [**lower darker** \| lower brighter] | | sign convention of both interfaces (see above) | front-lit black background → lower darker; back-lit → lower brighter |
| `T_lower` | 0..500 · 40 | threshold on $S$ for the liquid/foam interface | raise until the red mask keeps only the interface |
| `T_upper` | 0..500 · 47 | threshold on $-S$ for the foam/air interface | raise if the teal mask has blobs inside the foam |
| `r_lower` | 0..r_max · min(80, r_max) | half-width of the averaging band of the lower interface | narrow if the walls bend the interface; `r ≥ W_crop // 2` = whole width |
| `r_upper` | 0..r_max · min(80, r_max) | same for the upper interface | independent of `r_lower` |
| `blur_sigma × 10` | 0..100 · 5.0 | Gaussian smoothing (σ in px) before the Sobel derivative | raise if bubble edges compete with the interface |
| `blur_h` | 0..101 · 38 | horizontal box filter width (px), 0 = off | keep on for foam; it averages each row along $x$ |
| `min_h_upper` | 0..50 · 3 | minimum blob height of the teal mask (0/1 = off) | raise during a pour with big bubbles |
| `channel` — radio [**gray** \| G \| R \| B] | | grey level fed to the detector | pick the channel that carries the contrast of your lighting |

#### SAMPLING (peach)

| widget | range · default | what it does |
|---|---|---|
| `preset` — radio [**all** \| 1/2 \| 1/3 \| 1/5 \| 1/10] | | sets `frame_step` to 1, 2, 3, 5 or 10 |
| `frame_step` — "1 frame in K" | 1..60 · 1 | the batch analyses one frame in K |

#### Hotkeys of the OpenCV window

| key | effect |
|---|---|
| **Enter** | accept the parameters and run the batch |
| **Esc** | abort |
| `t` | toggle the specimen panel |
| `l` | toggle inline / stacked |
| `f` | fullscreen |
| ← → (or `a` / `d`) | previous / next frame |
| **Home** (or `0`) | first frame |
| **Space** | play / pause |

</details>

### What each parameter does to the detection

Each figure shows the same frame with one parameter changed; the highlight panel carries the shaded zones and volumes so the effect is visible at a glance, and the caption strip gives the numbers.

<p align="center"><img src="docs/images/param_T_lower.png" width="80%"></p>

*`T_lower` 4 vs 40 (green run). Both values give almost the same lower interface (1257.3 vs 1258.2 px) because the argmax already lands on the true interface, but at 4 every one of the 343 columns is validated and the red mask is littered with weak gradients in the foam and below the beer — every one of them a candidate the moment the interface weakens. 40 keeps only the interface (267 columns, the walls excluded).*

<p align="center"><img src="docs/images/param_T_upper.png" width="80%"></p>

*`T_upper` 4 vs 14 (black run). The bottom-up scan stops on the first row above threshold: at 4 it stops on bubble reflections right above the beer, y = 465.9 px (465 mL) instead of the foam top at 159.8 px (730 mL) — the shaded "foam" band collapses to 316 mL instead of 581 mL. The teal mask on the left shows the ladder of false candidates it climbed.*

<p align="center"><img src="docs/images/param_r_lower.png" width="100%"></p>

*`r_lower` 10, 64 and 171 (green run). A narrow band uses 21 columns (1260.0 px, 220.9 mL); a medium one 129 columns (1259.4 px, 221.2 mL); the tuned band spans the whole crop (171 = half the crop width), 267 columns, 1258.2 px (221.8 mL). On this recording the interface is flat across the cylinder and the threshold already drops the wall columns, so the three agree within 1 mL; the narrow band is nevertheless the one to use when the meniscus bends the interface near the glass.*

<p align="center"><img src="docs/images/param_blur_sigma.png" width="80%"></p>

*`blur_sigma` 0.5 vs 5.7 (green run). With almost no smoothing every bubble edge passes the upper threshold: the teal mask is a cloud of strokes and the bottom-up scan stops on the first of them, 523.6 px (605 mL) instead of the foam top at 431.7 px (653 mL) — the shaded foam band loses 48 mL. The lower interface, a strong edge, survives either way (1258.4 vs 1258.2 px).*

<p align="center"><img src="docs/images/param_blur_h.png" width="80%"></p>

*`blur_h` 0 vs 38 (black run). The horizontal box filter averages each row along $x$, keeping horizontal interfaces and washing out the vertical texture of the foam: the teal blobs in the foam disappear, the upper interface moves from 170.1 px (721 mL, 217 columns) to 159.8 px (730 mL, 203 columns). The lower interface does not move (845.7 px).*

<p align="center"><img src="docs/images/param_min_h_upper.png" width="80%"></p>

*`min_h_upper` 0 vs 15 (green run, frame 108, zoom on the foam top). Without the filter the bottom-up scan stops on a thin bubble edge inside the foam, y = 461.0 px (638 mL, 305 columns); with blobs at least 15 rows tall it reaches the foam top at 431.7 px (653 mL, 237 columns) — a 29 px, 15 mL difference that grows during the pour, when the foam is full of large bubbles.*

<p align="center"><img src="docs/images/param_y_top.png" width="80%"></p>

*`y_top` 200 vs 34 (black run). The rim of both test cylinders lies outside the frame, so the mask is shown the other way round: a `y_top` below the foam top (magenta line at 200) masks the true upper interface and the detector reports nothing (0 columns, no foam band); at 34 the foam top is found at y = 159.8 px (730 mL). The lower interface is unaffected (845.7 px in both cases).*

<p align="center"><img src="docs/images/param_y_bottom.png" width="100%"></p>

*`y_bottom` 1079 (no mask), 800 (too high) and 1031 (tuned), black run. Without the mask the bright edge of the base appears as a teal blob at the bottom of the mask — harmless here, but it is what the upper-interface scan would fall on whenever the lower interface is lost. At 800 the mask covers the beer/foam interface itself (0 columns, nothing shaded). The effect of the mask on these particular frames is otherwise nil: 845.7 / 159.8 px in both valid cases.*

<p align="center"><img src="docs/images/param_cx.png" width="80%"></p>

*`cx` 40 vs 171 (green run). With the axis shifted to the left wall the band is clipped to the columns near the glass (170 lower / 124 upper columns instead of 267 / 237): lower 1259.1 px (221.4 mL) and upper 434.3 px (651.7 mL) instead of 1258.2 / 431.7 px. The shift is small here because the interfaces are flat, but half the band is wasted outside the crop.*

<p align="center"><img src="docs/images/param_channel.png" width="100%"></p>

*`channel` gray / G / R (green run). Gray (tuned) and G give the same lower interface (1258.2 vs 1258.0 px); G has more contrast on a green screen and validates 338 upper columns instead of 237, but its stronger bubble edges stop the scan 32 px below the foam top (463.6 px, 636 mL, instead of 431.7 px, 653 mL). R carries almost no signal: the gradient panel is noise and nothing passes the thresholds.*

## Tool 3 — Batch (`scripts/run_batch.py`)

Reads `calib.json` and `params.json`, processes frames `--start .. --end` one in `--step` (default: the `frame_step` saved by the tuner) and writes `levels.csv`, `meta.json` and `figure.png` into the run directory.

<table>
<tr>
<td align="center" width="50%"><img src="docs/images/batch_figure_green.png" width="100%"><br><em>Green run, frames 0..442 (1 frame = 60 source frames, t = 0 at the pour), 443 rows, the lower interface found on 403 of them and the upper on 422. Liquid + foam peaks at 787 mL, the liquid rises to a 257 mL plateau as it drains out of the foam, the foam decays from 660 mL to 99 mL in 14 min with a visible step at the collapse (t ≈ 12.5 min). The stray points before t = 0 and at the very end are the empty cylinder and the last frame.</em></td>
<td align="center" width="50%"><img src="docs/images/batch_figure_black.png" width="100%"><br><em>Black run, frames 972..40000 step 60, 651 frames analysed. Liquid + foam peaks at 776.4 mL and ends at 278.9 mL; the liquid plateau is 253.1 mL; the foam decays from 746.7 mL to 25.8 mL in 33 min.</em></td>
</tr>
</table>

Both figures carry the $\pm u_\text{total}$ band of the uncertainty budget (about 3 mL), which is barely wider than the lines at this scale. The liquid + foam curve of the green run gets jumpy after t ≈ 700 s as the foam top turns into a sparse layer of large bubbles; the per-frame `u_method_upper_mL` column records it.

<p align="center"><img src="docs/images/frame_step_illustration.png" width="100%"></p>

*Green run, $V_\text{lower}(t)$ at `frame_step` 1 (line) and 10 (circles): sub-sampling ten times does not change the curve. The spike at the start is the pour crossing the band.*

<details>
<summary><b>Files of a run directory and columns of <code>levels.csv</code></b></summary>

| file | written by | content |
|---|---|---|
| `calib.json` | `calibrate.py` | crop band, masks, clicks, fitted models |
| `calibration_check.png` | `calibrate.py` | the verification figure above |
| `params.json` | `tune.py` | `DetectionParams` + `frame_step` |
| `levels.csv` | `run_batch.py` | one row per frame (NaN rows for skipped frames keep the time axis aligned) |
| `meta.json` | `run_batch.py` | video, frame range, step, parameters, timings, summaries, the Markdown budget table (`uncertainty_budget`) |
| `figure.png` | `run_batch.py` | $V_\text{lower}(t)$, $V_\text{total}(t)$, $V_\text{foam}(t)$ with the $\pm u_\text{total}$ bands of the uncertainty budget (drawn whenever the calibration carries the cylinder radius; otherwise the batch says so and draws the lines alone) |
| `uncertainty_curvature.png`, `uncertainty_schematic.png`, `uncertainty_budget.png` | `uncertainty_report.py` | the figures of the budget section |
| `uncertainty.json` | `uncertainty_report.py` | geometry, per-volume table, notes, method summary |

`levels.csv` columns:

| column | meaning |
|---|---|
| `frame_idx`, `t_sec` | frame index and time (× `--frame-scale` for a pre-subsampled video) |
| `y_lower_px`, `std_lower_px`, `n_lower` | mean row of the liquid/foam interface, its per-column standard deviation, valid columns |
| `y_upper_px`, `n_upper` | mean row of the foam/air interface, valid columns |
| `V_lower_mL` | liquid volume, $V(y_\text{lower})$ |
| `V_total_mL` | liquid + foam, $V(y_\text{upper})$ |
| `V_foam_mL` | $V_\text{total} - V_\text{lower}$ |
| `u_method_lower_mL`, `u_method_upper_mL` | per-frame method uncertainty of each interface (see below) |

</details>

## Uncertainty budget

Every volume comes with a standard uncertainty ($k = 1$) built from four independent terms. The full derivation is in [`docs/uncertainty.md`](docs/uncertainty.md); this is the summary. Whenever a quantity is only known to lie in an interval of width $w$, the uniform-distribution rule $u = w / \sqrt{12}$ is used.

**Calibration.** The RMS residual of the recommended model, floored by the resolution of the printed scale ($\delta_s$ = the thin step, 10 mL on a 1000 mL cylinder):

$$u_\text{cal} = \max\left(\sqrt{\tfrac{1}{n}\sum_i \varepsilon_i^2},\; \frac{\delta_s}{\sqrt{12}}\right) = 2.89\ \text{mL on the 1000 mL runs.}$$

**Pixel quantisation.** A row is an integer; with the local slope $s(V) = |dV/dy|$ (≈ 0.57 mL/px here):

$$u_\text{px}(V) = \frac{1\ \text{px}}{\sqrt{12}}\; s(V) \approx 0.16\ \text{mL}.$$

**Cylinder curvature.** A graduation, or the interface, is a horizontal *circle* of radius $R$; a camera at distance $D$ sees it as an arc of ellipse, so the columns of the band sample different rows. The full vertical span of the arc inside the band, $\Delta(V)$, is treated as a uniform reading error:

$$u_\text{curv}(V) = \frac{\Delta(V)}{2\sqrt{3}}\; s(V), \qquad \Delta \approx |v_\text{click} - c_y| \left(\frac{r}{f}\right)^2 \frac{1-\rho}{2\rho}\quad (\rho = R/D).$$

$\Delta$ vanishes on the **camera horizon** (the row $c_y$ where the circle is seen edge-on), grows linearly away from it and quadratically with the band half-width $r$. The geometry ($\rho$, $c_y$, $f$, pitch $\alpha$) is recovered from the calibration alone: with front **and back** clicks the two rows of the same ring give $\rho = (s_f - s_b)/(s_f + s_b)$ and the horizon is where the two lines cross — no focal length, no physical radius (`focal_free`); with front clicks only, the radius preset and the Möbius $c$ coefficient are used instead (`mobius_tilt`).

<p align="center"><img src="docs/images/uncertainty_schematic.png" width="100%"></p>

*Green run geometry (R = 3.20 cm, D = 35.1 cm, f = 1716 px, ρ = 0.091, α = 0, focal-free): (a) top view, the ±22.0° band samples a piece of the ±84.8° visible arc; (b) side view, the camera looks along the 500 mL ring; (c) the 1000 mL ring projected as an ellipse, Δ = 6.1 px across the band.*

<p align="center"><img src="docs/images/uncertainty_curvature.png" width="100%"></p>

*Left: the projected circle of every graduation over the real frame (front clicks in blue, back clicks in red), curving up below the horizon (y ≈ 955 px, V ≈ 520 mL) and down above it. Centre: the Δ wedge and $u_\text{curv}(V)$: 0.04 mL at 500 mL, 0.87 mL at 100 mL, 0.99 mL at 1000 mL. Right: top and side views of the recovered geometry.*

**Method.** For each frame and interface, the valid per-column rows are converted to volumes, columns farther than $t$ = 7 mL from the mean are dropped (a column that jumped to a bubble is a failure of the method, not a property of the interface), and the spread of the rest gives

$$u_\text{method,frame} = \frac{|V(y_\text{up}) - V(y_\text{lo})|}{2\sqrt{3}}.$$

The run-level value is the **median** over the frames.

<p align="center"><img src="docs/images/uncertainty_method.png" width="100%"></p>

*Lower interface of the green run (frame 108, zoom ×3, foam wash above the line, liquid wash below): 267 per-column detections (red dots) around the mean row 1258.24 px (221.8 mL), dashed bounds 1253 px (224.5 mL, yellow) and 1261 px (220.4 mL, cyan), range 4.08 mL, $u_\text{method}$ = 1.18 mL on this frame; the median over the 403 frames of the batch is 1.04 mL (mean 1.13, p95 1.74 mL).*

**Total**, in quadrature:

$$u_\text{total}(V) = \sqrt{u_\text{cal}^2 + u_\text{px}(V)^2 + u_\text{curv}(V)^2 + u_\text{method}^2}.$$

<p align="center"><img src="docs/images/uncertainty_budget.png" width="100%"></p>

*Green-screen geometry: contributions and total versus volume (left) and the share of the variance (right). The scale resolution dominates everywhere; curvature only matters at the ends of the cylinder; the pixel term is negligible. The three budget figures and the table below come from an earlier green-screen recording of the same setup, calibrated with front and back clicks (the run of the figures above has front clicks only, so its budget would use the `mobius_tilt` geometry).*

| V (mL) | u_cal | u_px | u_curv | u_method | **u_total** |
|---:|---:|---:|---:|---:|---:|
| 100 | 2.89 | 0.16 | 0.87 | 0.99 | **3.18** |
| 250 | 2.89 | 0.17 | 0.56 | 0.99 | **3.11** |
| 500 | 2.89 | 0.17 | 0.04 | 0.99 | **3.06** |
| 750 | 2.89 | 0.16 | 0.48 | 0.99 | **3.09** |
| 1000 | 2.89 | 0.16 | 0.99 | 0.99 | **3.21** |

*Green run, 1000 mL cylinder, r = 64 px, PCHIP calibration, method term = median of `u_method_lower_mL` over 955 frames (mean 0.96, p95 1.47 mL).*

```bash
python scripts/uncertainty_report.py --run-dir runs/pour --video pour.mp4 --frame 108
# options: --r-px 64  --alpha-deg 0  --no-back  --levels-csv other.csv  --volumes 100,250,500,750,1000
```

The report prints the geometry and the Markdown table above and writes `uncertainty_curvature.png`, `uncertainty_schematic.png`, `uncertainty_budget.png` and `uncertainty.json` into the run directory. The method term is read from `levels.csv`; without a batch the budget carries a placeholder and says so.

## Project layout

```
liquid-foam-interface-volume-detector/
├── cylvision/
│   ├── io.py                  unicode-safe image IO, VideoSource, key codes
│   ├── magnifier.py           ×8 loupe for pixel-accurate clicks
│   ├── calibration/           models.py (PCHIP / poly2 / Möbius), clicks.py, back_clicks.py,
│   │                          store.py (Calibration, PRESETS, JSON), check.py (verification figure)
│   ├── detection/             gradient.py (Sobel, masks), interfaces.py (DetectionParams, two-pass
│   │                          detector), panels.py (highlight / gradient / mask, zone wash, readout
│   │                          boxes, specimen panel, info strip)
│   ├── ui/                    theme.py (palette, line geometry, wash, fonts), text.py (TrueType text
│   │                          via Pillow, Hershey fallback), controls_panel.py (Tk), frame_picker.py, tuner.py
│   ├── pipeline/              run_store.py (run directory), batch.py (CSV rows), plot.py (V(t))
│   └── uncertainty/           curvature.py (camera geometry, Δ), empirical.py (method), budget.py
├── scripts/                   calibrate.py · tune.py · run_batch.py · uncertainty_report.py ·
│                              make_readme_figures.py
├── docs/                      uncertainty.md (full derivation) · images/ (every README figure, the GIF + manifest)
├── tests/                     pytest, synthetic images only
├── SPEC.md                    internal engineering spec
├── pyproject.toml · requirements.txt · LICENSE
```

## Tests

```bash
pip install -e ".[dev]"
python -m pytest -q          # 78 passed in ~3 s
```

The tests need no video and no display: synthetic three-band crops (air / foam / liquid, both polarities, with noise) check that the detector finds both interfaces within ±2 px and that swapping the polarity breaks it; the calibration models, the legacy JSON layouts, the curvature geometry (Δ = 0 on the horizon, symmetric growth, $u = \Delta/(2\sqrt{3})$), the empirical spread, the batch/CSV round trip and the TrueType text helper (font lookup, anchors, backing box, Hershey fallback, readout alignment) are covered as well.

## Regenerating the figures

Every image in `docs/images/` comes from a real run and is described, with its frame and parameters, in [`docs/images/README_figures.json`](docs/images/README_figures.json). To rebuild them from two run directories (a green-screen run with `lower_brighter` and a black-background run with `lower_darker`):

```bash
python scripts/make_readme_figures.py \
    --green-run runs/green --green-video green.mp4 --frame-scale-green 60 --t0-frame 22 \
    --black-run runs/black --black-video black.mp4 \
    --out docs/images --panel-screenshot
# --only hero,timeline,param_T_lower   regenerate a subset (timeline = strip + detection_timeline.gif)
# --skip-batch                         reuse the cached batch CSVs from --work-dir
# --reuse-batch black                  reuse only the black cache (the 5 GB seek), redo the green batch
# --t0-frame 22                        caption times count from this green frame (start of the pour)
```

The overlay style is set in `cylvision/ui/theme.py`: line geometry (`MEAN_LINE_W` = 3 px flat, `MEAN_LINE_SHADOW_W` = 1 px, `TRIANGLE_W/H`, `LEADER_DOT/GAP` for the dotted leader), wash colours and alphas (`FOAM_WASH_BGR`, `LIQUID_WASH_BGR`, `WASH_ALPHA_*`), and the readout typography (`LABEL_FONT_ZONE` = `DejaVuSans`, `LABEL_FONT_VALUE` = `DejaVuSansMono`, `LABEL_ZONE_SIZE`, `LABEL_VALUE_SIZE`, `LABEL_LETTER_SPACING`, `LABEL_BACKING_*`, `LABEL_BAR_W`). The fonts are the TrueType files shipped with matplotlib (`matplotlib.get_data_path()/fonts/ttf`), rendered with Pillow by `cylvision/ui/text.py`, so the figures come out identical on every machine; any other name from that folder (`DejaVuSerif`, `STIXGeneral`, `cmss10`, `cmtt10`...) or a path to a `.ttf` works, and OpenCV's Hershey font is used if Pillow or the file is missing. The wash and the labels can be switched off per call with `show_wash=False` / `show_labels=False` in `annotate_interfaces` and `make_specimen_panel`.

## License

MIT — see [LICENSE](LICENSE). © 2026 Camille Duparc.
