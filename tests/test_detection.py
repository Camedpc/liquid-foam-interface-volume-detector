"""Detection tests on synthetic crops (no video, no display)."""
from __future__ import annotations

import numpy as np
import pytest

from cylvision.detection import (
    DetectionParams,
    InterfaceResult,
    apply_bottom_mask,
    apply_top_mask,
    band_columns,
    compute_gradient,
    detect_from_crop,
    detect_interfaces,
    make_info_strip,
    make_panels,
    make_specimen_panel,
    annotate_interfaces,
    to_gray,
)

H, W = 400, 120
Y_UPPER, Y_LOWER = 100, 250   # first row of the foam, first row of the liquid


def synthetic_crop(polarity: str, *, seed: int = 0, noise: float = 6.0) -> np.ndarray:
    """Three horizontal bands (air / foam / liquid) + Gaussian noise, BGR uint8.

    ``lower_darker``: dark air, bright foam, dark liquid.
    ``lower_brighter``: bright air, dark foam, bright liquid.
    """
    if polarity == "lower_darker":
        air, foam, liquid = 30.0, 200.0, 60.0
    else:
        air, foam, liquid = 220.0, 60.0, 180.0
    rng = np.random.default_rng(seed)
    img = np.empty((H, W), dtype=np.float64)
    img[:Y_UPPER] = air
    img[Y_UPPER:Y_LOWER] = foam
    img[Y_LOWER:] = liquid
    img += rng.normal(0.0, noise, size=img.shape)
    gray = np.clip(img, 0, 255).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def params_for(polarity: str, **kw) -> DetectionParams:
    # The upper interface is the LOWER FLANK of its gradient blob (bottom-up
    # threshold crossing, by design), so T_upper is set close to the peak to
    # keep that flank within 2 px of the true edge on the synthetic image.
    base = dict(polarity=polarity, T_lower=60.0, T_upper=200.0, r_lower=40, r_upper=40,
                blur_sigma=1.5, blur_h=9, min_h_upper=3, y_top=5, y_bottom=H - 6)
    base.update(kw)
    return DetectionParams(**base)


@pytest.mark.parametrize("polarity", ["lower_darker", "lower_brighter"])
def test_detect_both_interfaces(polarity: str) -> None:
    crop = synthetic_crop(polarity)
    result, Gy = detect_from_crop(crop, params_for(polarity))
    assert Gy.shape == (H, W) and Gy.dtype == np.float32
    assert result.y_lower is not None and result.y_upper is not None
    # The step sits between rows Y-1 and Y; the peak lands on either.
    assert abs(result.y_lower - (Y_LOWER - 0.5)) <= 2.0
    assert abs(result.y_upper - (Y_UPPER - 0.5)) <= 2.0
    assert result.y_upper < result.y_lower
    assert result.n_lower == 81 and result.n_upper == 81
    assert result.std_lower is not None and result.std_lower < 1.5
    assert result.lower_rows.shape == (W,) and result.lower_valid.dtype == bool
    assert result.S.shape == (H, W)


@pytest.mark.parametrize("polarity,wrong", [("lower_darker", "lower_brighter"),
                                            ("lower_brighter", "lower_darker")])
def test_wrong_polarity_misses_lower_interface(polarity: str, wrong: str) -> None:
    crop = synthetic_crop(polarity)
    result, _ = detect_from_crop(crop, params_for(wrong))
    # With the sign flipped the strongest positive peak is the OTHER interface.
    assert result.y_lower is None or abs(result.y_lower - (Y_LOWER - 0.5)) > 2.0


def test_polarity_sign_relation() -> None:
    crop = synthetic_crop("lower_darker")
    p = params_for("lower_darker")
    res, Gy = detect_from_crop(crop, p)
    np.testing.assert_allclose(res.S, -Gy)
    p2 = params_for("lower_brighter")
    res2 = detect_interfaces(Gy, p2)
    np.testing.assert_allclose(res2.S, Gy)


def test_bands_limit_valid_columns() -> None:
    crop = synthetic_crop("lower_darker")
    p = params_for("lower_darker", r_lower=10, r_upper=W, cx=60)
    res, _ = detect_from_crop(crop, p)
    assert res.band_lower == (50, 70)
    assert res.band_upper == (0, W - 1)
    assert res.n_lower == 21
    assert res.n_upper == W
    assert not res.lower_valid[:50].any() and not res.lower_valid[71:].any()
    assert band_columns(W, 5, 10) == (0, 15)
    assert band_columns(W, W - 3, 10) == (W - 13, W - 1)


def test_masks_zero_rows_and_exclude_interfaces() -> None:
    Gy = np.ones((50, 8), dtype=np.float32)
    top = apply_top_mask(Gy, 10)
    assert top[:10].max() == 0.0 and top[10:].min() == 1.0
    assert Gy.min() == 1.0  # input untouched
    bot = apply_bottom_mask(Gy, 39)
    assert bot[40:].max() == 0.0 and bot[:40].min() == 1.0
    assert apply_bottom_mask(Gy, None) is Gy
    assert apply_top_mask(Gy, 0) is Gy
    # Masking below the lower interface hides it from the detector.
    crop = synthetic_crop("lower_darker")
    res, _ = detect_from_crop(crop, params_for("lower_darker", y_bottom=Y_LOWER - 30))
    assert res.y_lower is None or res.y_lower < Y_LOWER - 30
    assert res.y_upper is not None and abs(res.y_upper - (Y_UPPER - 0.5)) <= 2.0
    # Masking above the upper interface hides it but keeps the lower one.
    res, _ = detect_from_crop(crop, params_for("lower_darker", y_top=Y_UPPER + 30))
    assert res.y_upper is None
    assert res.y_lower is not None and abs(res.y_lower - (Y_LOWER - 0.5)) <= 2.0


def test_upper_search_is_bottom_up_from_lower() -> None:
    """A stronger 'upper-sign' edge above the true foam top must be ignored."""
    # air 120 / foam 200 / liquid 20: the foam/air step (+80) is weaker than
    # a decoy stripe of amplitude 120 placed in the air zone (rows 40..59),
    # whose TOP edge (brighter going down) has the upper-interface sign. A
    # raw argmax of -S would pick the decoy; the bottom-up search must not.
    gray = np.full((H, W), 120, dtype=np.uint8)
    gray[Y_UPPER:Y_LOWER] = 200
    gray[Y_LOWER:] = 20
    gray[40:60] = 240
    crop = np.repeat(gray[:, :, None], 3, axis=2)
    p = params_for("lower_darker", T_upper=100.0)
    res, _ = detect_from_crop(crop, p)
    assert res.y_lower is not None and abs(res.y_lower - (Y_LOWER - 0.5)) <= 2.0
    assert res.y_upper is not None and abs(res.y_upper - (Y_UPPER - 0.5)) <= 2.0
    # Sanity: the decoy really is the strongest upper-sign feature.
    assert int(np.argmax(-res.S[:, W // 2])) in (39, 40)


def test_min_h_upper_filters_thin_components() -> None:
    S = np.zeros((60, 20), dtype=np.float32)
    # Lower interface at row 50 (positive S), thin 1-row blip at row 30 and a
    # tall blob at rows 10..14 (negative S).
    S[50] = 100.0
    S[30] = -100.0
    S[10:15] = -100.0
    p = DetectionParams(polarity="lower_brighter", T_lower=10, T_upper=10, r_lower=20, r_upper=20,
                        min_h_upper=3)
    res = detect_interfaces(S, p)
    assert res.y_lower == 50.0
    assert res.y_upper == 14.0          # lowest row of the tall blob, blip rejected
    assert res.upper_lowest == 14
    p.min_h_upper = 0
    res = detect_interfaces(S, p)
    assert res.y_upper == 30.0          # bottom-up: the blip is the first hit


def test_no_interface_when_below_threshold() -> None:
    S = np.zeros((40, 10), dtype=np.float32)
    p = DetectionParams(polarity="lower_brighter", T_lower=5, T_upper=5)
    res = detect_interfaces(S, p)
    assert res.y_lower is None and res.y_upper is None
    assert res.n_lower == 0 and res.n_upper == 0 and res.std_lower is None


def test_compute_gradient_and_channels() -> None:
    gray = np.zeros((30, 10), dtype=np.uint8)
    gray[15:] = 200
    Gy = compute_gradient(gray, blur_sigma=1.0, blur_h=4)
    assert Gy.dtype == np.float32
    assert np.argmax(Gy[:, 5]) in (14, 15)
    assert Gy.max() > 0
    bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    bgr[..., 1] = 77
    assert to_gray(bgr, "G").max() == 77 and to_gray(bgr, "R").max() == 0
    assert to_gray(gray, "R") is gray
    with pytest.raises(ValueError):
        to_gray(bgr, "X")


def test_params_from_legacy_dict_and_roundtrip() -> None:
    legacy = {"T_pos": 14, "T_neg": 47, "y_top": 34, "r_neg": 80, "r_pos": 113,
              "blur_sigma": 5.0, "blur_h": 38, "min_h_pos": 3, "cx": 113,
              "ref_image": "reference.png", "roi_x": [841, 1067]}
    p = DetectionParams.from_dict(legacy)
    assert p.polarity == "lower_darker"
    assert p.T_lower == 47.0 and p.T_upper == 14.0
    assert p.r_lower == 80 and p.r_upper == 113
    assert p.min_h_upper == 3 and p.y_top == 34 and p.cx == 113
    assert p.blur_sigma == 5.0 and p.blur_h == 38
    assert p.y_bottom is None and p.channel == "gray"
    d = p.to_dict()
    assert set(d) == {"polarity", "T_lower", "T_upper", "r_lower", "r_upper", "blur_sigma",
                      "blur_h", "min_h_upper", "y_top", "y_bottom", "cx", "channel"}
    assert DetectionParams.from_dict(d) == p
    # New-style keys win over legacy aliases, and polarity is honoured.
    p2 = DetectionParams.from_dict({**legacy, "T_lower": 99, "polarity": "lower_brighter"})
    assert p2.T_lower == 99.0 and p2.polarity == "lower_brighter"
    with pytest.raises(ValueError):
        DetectionParams(polarity="sideways")  # type: ignore[arg-type]


def test_panels_smoke() -> None:
    crop = synthetic_crop("lower_darker")
    p = params_for("lower_darker")
    res, Gy = detect_from_crop(crop, p)
    panels = make_panels(crop, Gy, res, p)
    assert panels.dtype == np.uint8 and panels.ndim == 3 and panels.shape[2] == 3
    assert panels.shape[1] == 3 * W + 2 * 2
    assert panels.shape[0] > H
    body = make_panels(crop, Gy, res, p, with_labels=False)
    assert body.shape[0] == H
    small = make_panels(crop, Gy, res, p, panel_w=60)
    assert small.shape[1] == 3 * 60 + 2 * 2
    info = make_info_strip(panels.shape[1], res, p, model_fn=lambda y: 1000.0 - 2.0 * np.asarray(y),
                           frame_idx=12, fps=30.0)
    assert info.dtype == np.uint8 and info.shape[1] == panels.shape[1] and info.shape[2] == 3
    assert info.shape[0] >= 20
    info_empty = make_info_strip(200, detect_interfaces(np.zeros((H, W), np.float32), p), p)
    assert info_empty.shape[1] == 200
    canvas = np.concatenate([panels, info], axis=0)
    assert canvas.shape[1] == panels.shape[1]


def test_annotate_and_specimen_smoke() -> None:
    crop = synthetic_crop("lower_brighter")
    p = params_for("lower_brighter")
    res, _ = detect_from_crop(crop, p)
    a1 = annotate_interfaces(crop, res, p)
    assert a1.shape == crop.shape and a1.dtype == np.uint8
    a3 = annotate_interfaces(crop, res, p, zoom=3.0, y_bounds=(res.y_lower - 2, res.y_lower + 2))
    assert a3.shape == (3 * H, 3 * W, 3)
    frame = np.zeros((H, 300, 3), dtype=np.uint8)
    frame[:, 90:90 + W] = crop
    a_frame = annotate_interfaces(frame, res, p, offset_x=90)
    assert a_frame.shape == frame.shape
    spec = make_specimen_panel(frame, (90, 90 + W - 1, 5, H - 6), res)
    assert spec.shape == frame.shape and spec.dtype == np.uint8
    assert isinstance(res, InterfaceResult) and res.to_dict()["y_lower"] == res.y_lower
