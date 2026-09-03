# Uncertainty budget of a volume reading

This note derives every term of the uncertainty attached to a volume read by
`cylvision` from a video frame, and explains how they are combined. All
uncertainties are *standard* uncertainties (coverage factor $k = 1$), in mL.
The code lives in `cylvision/uncertainty/` (`curvature.py`, `empirical.py`,
`budget.py`); the numbers quoted below come from a real 1000 mL run
(webcam, 1080 × 1920 portrait frame, band half-width $r = 64$ px).

## 0. Reading chain and notation

A frame goes through four steps, each with its own error source:

| step | what happens | term |
|---|---|---|
| calibration | clicked graduations → row-to-volume model $V(y)$ | $u_{\mathrm{calibration}}$ |
| detection | per-column interface rows, averaged over a band $\lvert x - c_x\rvert \le r$ | $u_{\mathrm{method}}$ |
| quantisation | the row is an integer pixel | $u_{\mathrm{pixel}}(V)$ |
| geometry | the interface is a horizontal *circle*, not a line | $u_{\mathrm{curvature}}(V)$ |

Image axes: $x$ to the right, $y$ (or $v$) **down**. $y_{\mathrm{click}}$ (or
$v_{\mathrm{click}}$) is the row clicked on a graduation during calibration,
$V$ the volume, $s(V) = \lvert \mathrm{d}V / \mathrm{d}y \rvert$ the local
calibration slope in mL/px.

Whenever a quantity is only known to lie inside an interval of width $w$
without preference, we use the uniform-distribution standard deviation

$$
u = \frac{w}{\sqrt{12}} = \frac{w}{2\sqrt{3}} .
$$

## 1. Calibration — $u_{\mathrm{calibration}}$

The calibration clicks $n$ graduations $(y_i, V_i)$ and fits three models
(PCHIP interpolation, quadratic polynomial, Möbius $V = (a y + b)/(c y + 1)$;
see `cylvision/calibration/models.py`). The recommended model has residuals
$\varepsilon_i = V_{\mathrm{model}}(y_i) - V_i$ and

$$
u_{\mathrm{fit}} = \sqrt{\frac{1}{n} \sum_i \varepsilon_i^2} .
$$

For an interpolating model (PCHIP) $u_{\mathrm{fit}} = 0$ by construction,
which does not mean the calibration is perfect: each click is placed on a
graduation *ring* whose thin step is $\delta_s$ mL (10 mL on a 1000 mL
cylinder), and the ring itself has a finite thickness, so the volume
attached to a click is uniformly uncertain over $\pm \delta_s / 2$. The budget
keeps the larger of the two:

$$
u_{\mathrm{calibration}} = \max\!\left(u_{\mathrm{fit}},\; \frac{\delta_s}{\sqrt{12}}\right)
\qquad (\delta_s = 10\ \mathrm{mL} \Rightarrow 2.9\ \mathrm{mL}).
$$

This is the dominant term of the budget on a 1000 mL cylinder; the only way
to reduce it is a finer scale or a volumetric re-calibration of the rings.

## 2. Pixel quantisation — $u_{\mathrm{pixel}}(V)$

A detected row is an integer; the true interface lies anywhere within that
pixel:

$$
u_{\mathrm{pixel}}(V) = \frac{1\ \mathrm{px}}{\sqrt{12}} \; s(V) .
$$

With $s \approx 0.57$ mL/px this is 0.16 mL. The band average of ~130
columns actually reaches sub-pixel precision, so the term is conservative;
it is kept because it is the resolution floor of a *single* column.

## 3. Cylinder curvature — $u_{\mathrm{curvature}}(V)$

![Top view, side view and image-plane ellipse of one graduation](images/uncertainty_schematic.png)

*Figure 1 — `images/uncertainty_schematic.png`: (a) top view with the camera,
the visible arc and the cone of the averaging band, (b) side view with the
pitch and the rays to every graduation, (c) the projected ellipse of one
graduation in the image plane with the band and the span $\Delta$ (zoom
inset).*

### 3.1 Why a horizontal circle has a vertical extent in the image

A graduation, or the liquid/foam interface, is a horizontal circle of radius
$R$ around the cylinder axis. A pinhole camera at distance $D$ from the axis
sees it as an arc of ellipse: the point facing the camera and the points at
the silhouette are at different depths, so they project to different rows
unless the circle is exactly at the camera height (the *camera horizon*).
The detector averages rows over the band $\lvert u - c_x \rvert \le r$; it
therefore samples an arc whose vertical span $\Delta$ is a reading error.

### 3.2 Camera model

World frame: cylinder axis $Z$ (up), radius $R$. Camera at $(0, -D, h_{\mathrm{cam}})$
looking along $+Y$, pitched by $\alpha$ (positive = pointing up). A surface
point at height $Z$ and azimuth $\theta$ (0 = facing the camera) is
$P = (R\sin\theta,\, -R\cos\theta,\, Z)$. In camera coordinates (translation,
then rotation of $-\alpha$ about the camera $X$ axis):

$$
\begin{aligned}
x_c &= R\sin\theta, \\
y_c &= (D - R\cos\theta)\cos\alpha + (Z - h_{\mathrm{cam}})\sin\alpha, \\
z_c &= -(D - R\cos\theta)\sin\alpha + (Z - h_{\mathrm{cam}})\cos\alpha .
\end{aligned}
$$

Pinhole projection with focal length $f$ (px), principal point $(c_x, c_y)$
and $v$ pointing down:

$$
u - c_x = \frac{f\, R\sin\theta}{y_c}, \qquad
v - c_y = \frac{f\left[(D - R\cos\theta)\sin\alpha - (Z - h_{\mathrm{cam}})\cos\alpha\right]}{y_c} .
$$

$c_y$ is the row of the camera horizon: a circle at the camera height
($Z = h_{\mathrm{cam}}$, $\alpha = 0$) projects to a straight line at $v = c_y$.

### 3.3 Link with the Möbius calibration model

At the front ($\theta = 0$), with $p = D - R$ and $B = Z - h_{\mathrm{cam}}$:

$$
q \equiv v_{\mathrm{click}} - c_y = f\,\frac{p\sin\alpha - B\cos\alpha}{p\cos\alpha + B\sin\alpha} .
$$

For a cylinder $Z = V/(\pi R^2) + Z_0$ is affine in $V$, so $B$ is affine in
$V$ and $q$ is a ratio of two affine functions of $V$. Inverting gives
$V = (a v + b)/(c v + 1)$: the Möbius model of the calibration package is the
exact pinhole law, and its $c$ coefficient is the signature of the tilt
($c = 0 \Leftrightarrow \alpha = 0$). Identifying coefficients,

$$
\tan\alpha = \frac{c f}{1 + c\, c_y} .
$$

### 3.4 Offset of the arc

Subtracting the front reading from the projection at azimuth $\theta$
factorises:

$$
v(\theta) - v_{\mathrm{click}} = \frac{f R (1 - \cos\theta)\, B}{D_0\, D_\theta},
\qquad
D_0 = p\cos\alpha + B\sin\alpha, \quad
D_\theta = D_0 + R(1 - \cos\theta)\cos\alpha,
$$

with $B$ and $D_0$ recovered from the clicked row alone:

$$
B = p\,\frac{f\sin\alpha - q\cos\alpha}{f\cos\alpha + q\sin\alpha}, \qquad
D_0 = \frac{p f}{f\cos\alpha + q\sin\alpha} .
$$

The offset is proportional to $B$: it vanishes on the horizon and changes
sign across it (arcs bend *up* below the horizon, *down* above it, as seen
on Figure 2). The horizontal projection $u - c_x = f R\sin\theta / D_\theta$
is maximal at the silhouette, $\cos\theta_t = \rho \equiv R/D$, so the
visible half-width of the cylinder in the image is

$$
a_{\mathrm{px}} = \frac{f\rho}{\sqrt{1 - \rho^2}} .
$$

### 3.5 Span inside the band and the uncertainty

The band keeps $\lvert u - c_x\rvert \le r$, i.e. $\lvert\theta\rvert \le \theta_r$
with $u(\theta_r) - c_x = r$. The full span of the arc inside the band is

$$
\Delta(V) = \max_{\lvert\theta\rvert \le \theta_r} v(\theta) - \min_{\lvert\theta\rvert \le \theta_r} v(\theta)
= \lvert v(\theta_r) - v_{\mathrm{click}} \rvert ,
$$

evaluated numerically by `curvature_delta_px`. For a band much narrower than
the cylinder ($\theta_r \approx r D_0 / (f R)$, no tilt) the small-angle
expansion gives the closed form

$$
\Delta \approx \frac{r^2 \lvert B \rvert}{2 f R}
= \lvert v_{\mathrm{click}} - c_y \rvert \left(\frac{r}{f}\right)^2 \frac{1 - \rho}{2\rho} :
$$

$\Delta$ grows *linearly* with the distance to the horizon and
*quadratically* with the band half-width $r$ — halving $r$ divides the
curvature error by four (at the cost of averaging fewer columns).

The per-column rows are spread over an interval of width $\Delta$ on one side
of the clicked reading; the sign of the offset is known but its magnitude
depends on where the columns fall on the arc, so the reading is treated as
uniformly distributed over $\Delta$:

$$
u_{\mathrm{curvature}}(V) = \frac{\Delta(V)}{2\sqrt{3}} \; s(V)
\qquad (\text{in mL, with the local slope } s).
$$

![Projected circles on the frame and u_curvature(V)](images/uncertainty_curvature.png)

*Figure 2 — `images/uncertainty_curvature.png`: the projected circle of every
graduation over the real frame with the averaging band (left), the resulting
$\Delta$ wedge and $u_{\mathrm{curvature}}(V)$ with the camera horizon
(centre), and the top/side views of the recovered geometry (right).*

On the reference run ($R = 3.20$ cm, $D = 35$ cm, $f \approx 1700$ px,
$\rho = 0.09$, horizon at $\approx 520$ mL) the term is 0.04 mL at 500 mL and
rises to about 0.9–1.0 mL at 100 mL and 1000 mL.

### 3.6 Recovering the geometry from the calibration

The projection needs $\rho$, $c_y$, $f$ and $\alpha$. Two inversions are
implemented in `geometry_from_calibration`.

**Front + back clicks (`focal_free`, preferred).** If the back of the
graduation rings was also clicked (optional step of the calibration), the
rows of the same circle at $\theta = 0$ and $\theta = \pi$ are, without tilt,

$$
v_f - c_y = -\frac{f B}{D - R}, \qquad v_b - c_y = -\frac{f B}{D + R} .
$$

Both are affine in $V$ with slopes $s_f, s_b$ whose ratio is
$(D + R)/(D - R) = (1 + \rho)/(1 - \rho)$, hence

$$
\rho = \frac{s_f - s_b}{s_f + s_b},
$$

and the two lines cross at $B = 0$: the crossing row is the horizon $c_y$ and
the crossing volume $V_h$ is the level at which curvature vanishes. Neither
$R$, $f$ nor $D$ is needed. With a tilt the crossing row becomes
$c_y + f\tan\alpha$, which is exactly the effective horizon the projection
needs, so $\alpha$ is set to 0 in this branch. $D = R/\rho$ and
$f = a_{\mathrm{px}}\sqrt{1 - \rho^2}/\rho$ are then derived for the figures.

**Physical radius only (`no_tilt` / `mobius_tilt`).** With the linear slope
$s_{\mathrm{px}} = \lvert \mathrm{d}v/\mathrm{d}V \rvert$ of the front clicks
and $a_{\mathrm{px}} = (x_{\mathrm{right}} - x_{\mathrm{left}})/2$:

$$
K = \left(\frac{s_{\mathrm{px}}\,\pi R^3}{a_{\mathrm{px}}}\right)^2 = \frac{1 + \rho}{1 - \rho},
\qquad
\rho = \frac{K - 1}{K + 1}, \quad D = \frac{R}{\rho}, \quad f = \frac{a_{\mathrm{px}}\sqrt{1 - \rho^2}}{\rho} .
$$

$c_y$ defaults to the middle of the clicked graduations and $\alpha$ comes
from the Möbius $c$ (§3.3). This inversion is sensitive: with $K \approx 1.2$,
$\mathrm{d}\rho \approx 0.4\,\mathrm{d}K$ and a 2 % error on
$a_{\mathrm{px}}$ (a few pixels on the clicked edges, or the inner/outer
radius ambiguity) moves $\rho$ by about 0.05. Since $\Delta \propto \rho/(1+\rho)$
at fixed $a_{\mathrm{px}}$, the curvature term inherits that relative error —
on the reference run it comes out at 1.4 mL instead of 0.9 mL at 100 mL. Back
clicks are worth the extra minute.

### 3.7 What is *not* corrected

The offset has a known sign; one could shift the detected mean row by the
mean offset of the arc ($\approx \Delta/3$ for a parabolic arc) instead of
counting it as uncertainty. The package does not, because the interface of a
foamy liquid is not a clean circle and the correction would be smaller than
$u_{\mathrm{method}}$; the uniform treatment is the conservative choice.

## 4. Empirical method uncertainty — $u_{\mathrm{method}}$

For every frame and every interface the detector returns the per-column row
$y_j$ of the strongest signed gradient and a validity flag (above threshold,
inside the band). The reported level is $\bar y$, the mean over valid columns.
`method_uncertainty` then:

1. converts every valid column to a volume $V(y_j)$;
2. keeps only the columns with $\lvert V(y_j) - V(\bar y)\rvert \le t$
   ($t = 7$ mL by default) — a column that jumped to a bubble or to the other
   interface is a failure of the *method*, not a property of the interface,
   and would otherwise dominate the range;
3. takes $y_{\mathrm{up}} = \min_j y_j$ and $y_{\mathrm{lo}} = \max_j y_j$
   over the kept columns;
4. reports $\mathrm{range} = \lvert V(y_{\mathrm{up}}) - V(y_{\mathrm{lo}})\rvert$
   and

$$
u_{\mathrm{method,frame}} = \frac{\mathrm{range}}{2\sqrt{3}} .
$$

![Per-column rows, mean line and the two bounds on one frame](images/uncertainty_method.png)

*Figure 3 — `images/uncertainty_method.png`: one frame of the band with the
per-column argmax dots, the mean row and the two bounds $y_{\mathrm{up}}$ /
$y_{\mathrm{lo}}$ kept within $\pm t$ mL of the mean.*

Over a run the budget uses the **median** of the per-frame values (robust to
the few frames where the foam collapses or a bubble crosses the band); the
batch CSV keeps the per-frame value in `u_method_lower_mL` /
`u_method_upper_mL`, and `summarize_method_uncertainty` also reports the
mean, the 95th percentile and the maximum. On the reference run the median is
0.99 mL for the liquid/foam interface (955 frames, mean 0.96, p95 1.47 mL).

## 5. Combination

The four terms come from independent mechanisms and are combined in
quadrature:

$$
u_{\mathrm{total}}(V) = \sqrt{u_{\mathrm{calibration}}^2 + u_{\mathrm{pixel}}(V)^2 + u_{\mathrm{curvature}}(V)^2 + u_{\mathrm{method}}^2} .
$$

![Contributions vs volume and share of the variance](images/uncertainty_budget.png)

*Figure 4 — `images/uncertainty_budget.png`: each contribution and the
quadrature total as a function of the volume (left), and the share of the
variance carried by each term (right).*

Reference run (1000 mL cylinder, $r = 64$ px, PCHIP calibration):

| V (mL) | u_calibration | u_pixel | u_curvature | u_method | u_total |
|---:|---:|---:|---:|---:|---:|
| 100 | 2.89 | 0.16 | 0.87 | 0.99 | 3.18 |
| 250 | 2.89 | 0.17 | 0.56 | 0.99 | 3.11 |
| 500 | 2.89 | 0.17 | 0.04 | 0.99 | 3.06 |
| 750 | 2.89 | 0.16 | 0.48 | 0.99 | 3.09 |
| 1000 | 2.89 | 0.16 | 0.99 | 0.99 | 3.21 |

(Method term: median of `u_method_lower_mL` over 955 frames, mean 0.96,
95th percentile 1.47 mL; the same numbers as `docs/images/README_figures.json`.)

The scale resolution dominates everywhere; curvature only matters at the
ends of the cylinder and the pixel term is negligible. A foam volume
$V_{\mathrm{foam}} = V_{\mathrm{total}} - V_{\mathrm{lower}}$ combines the two
readings in quadrature as well (their calibration terms are correlated, so
this is slightly conservative).

## 6. Error sources considered and not budgeted

| source | why it is not a separate term |
|---|---|
| lens distortion (< 2–3 % at the frame edges for a webcam) | the calibration is fitted on the same image region as the readings; the residual distortion over the crop is absorbed by the model and shows up in $u_{\mathrm{fit}}$ |
| optical blur / PSF (1–2 px) and CMOS sensor noise (< 1 px RMS) | they widen the per-column spread, hence are already inside $u_{\mathrm{method}}$ |
| tolerance on the physical radius $R$ | enters only the curvature term (second order through $\rho$), which is itself ~1 mL; use back clicks to remove the dependence |
| band centre offset from the cylinder axis (a few px) | shifts the arc window sideways; second-order effect on $\Delta$ |
| meniscus / non-planar interface | part of the per-frame spread ($u_{\mathrm{method}}$) |
| time stamp of a frame, $1/(2\sqrt{3}\,\mathrm{fps}) \approx 10$ ms at 30 fps | an uncertainty on $t$, not on $V$; relevant only for rates |
