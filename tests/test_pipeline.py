"""Pipeline tests on synthetic frames (no video file, no display)."""
from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from cylvision.calibration import Calibration, PRESETS, fit_calibration
from cylvision.detection import DetectionParams
from cylvision.pipeline import (
    CSV_COLUMNS,
    RunDir,
    find_run,
    nan_row,
    plot_levels,
    process_frame,
    process_video,
    read_csv,
    summarize_rows,
    write_csv,
)

H, W_FULL = 400, 300
X_LEFT, X_RIGHT = 90, 209          # crop band -> W = 120
W = X_RIGHT - X_LEFT + 1
Y_UPPER, Y_LOWER = 100, 250        # first row of the foam / of the liquid
Y_TOP, Y_BOTTOM = 5, H - 6

# Linear scale: V = (380 - y) / 0.3 mL (so 3.33 mL per pixel).
Y0, ML_PER_PX = 380.0, 1.0 / 0.3


def linear_v(y: float) -> float:
    return (Y0 - y) * ML_PER_PX


def synthetic_frame(polarity: str, *, seed: int = 0, noise: float = 6.0,
                    y_upper: int = Y_UPPER, y_lower: int = Y_LOWER) -> np.ndarray:
    """Full frame: black border + the three-band crop (air / foam / liquid)."""
    if polarity == "lower_darker":
        air, foam, liquid = 30.0, 200.0, 60.0
    else:
        air, foam, liquid = 220.0, 60.0, 180.0
    rng = np.random.default_rng(seed)
    band = np.empty((H, W), dtype=np.float64)
    band[:y_upper] = air
    band[y_upper:y_lower] = foam
    band[y_lower:] = liquid
    band += rng.normal(0.0, noise, size=band.shape)
    frame = np.zeros((H, W_FULL), dtype=np.uint8)
    frame[:, X_LEFT:X_RIGHT + 1] = np.clip(band, 0, 255).astype(np.uint8)
    return np.repeat(frame[:, :, None], 3, axis=2)


@pytest.fixture
def calib() -> Calibration:
    V = np.array(PRESETS["1000"].graduations_ml, dtype=float)
    y = Y0 - 0.3 * V
    clicks = {"x_left": X_LEFT, "x_right": X_RIGHT, "y_top": Y_TOP, "y_bottom": Y_BOTTOM,
              "V_ml": V, "y_px": y, "x_px": [150] * len(V), "skipped_ml": []}
    return fit_calibration(clicks, PRESETS["1000"], "synthetic.mp4", (W_FULL, H))


def params_for(polarity: str, **kw) -> DetectionParams:
    base = dict(polarity=polarity, T_lower=60.0, T_upper=200.0, r_lower=40, r_upper=40,
                blur_sigma=1.5, blur_h=9, min_h_upper=3, y_top=Y_TOP, y_bottom=Y_BOTTOM)
    base.update(kw)
    return DetectionParams(**base)


class FakeVideo:
    """Minimal stand-in for ``VideoSource`` (n_frames, fps, frames())."""

    def __init__(self, n_frames: int, fps: float, *, bad: set[int] = frozenset(), polarity: str = "lower_darker"):
        self.n_frames, self.fps, self.bad, self.polarity = n_frames, fps, set(bad), polarity
        self.width, self.height = W_FULL, H
        self.read: list[int] = []

    def frames(self, start: int = 0, stop: int | None = None, step: int = 1) -> Iterator[tuple[int, np.ndarray]]:
        stop = self.n_frames if stop is None else min(stop, self.n_frames)
        for idx in range(start, stop, step):
            if idx in self.bad:
                continue
            self.read.append(idx)
            # The liquid level rises by one pixel per frame (volume goes up).
            yield idx, synthetic_frame(self.polarity, seed=idx, y_lower=Y_LOWER - idx)


# --------------------------------------------------------------------------- frame

@pytest.mark.parametrize("polarity", ["lower_darker", "lower_brighter"])
def test_process_frame_matches_linear_model(calib: Calibration, polarity: str) -> None:
    from cylvision.calibration import build_model_fn

    model_fn = build_model_fn(calib)
    assert model_fn(np.array([200.0]))[0] == pytest.approx(linear_v(200.0), abs=1e-6)
    row = process_frame(synthetic_frame(polarity), calib, params_for(polarity), model_fn,
                        frame_idx=7, t_sec=0.25)
    assert set(row) == set(CSV_COLUMNS)
    assert row["frame_idx"] == 7 and row["t_sec"] == 0.25
    assert abs(row["y_lower_px"] - (Y_LOWER - 0.5)) <= 2.0
    assert abs(row["y_upper_px"] - (Y_UPPER - 0.5)) <= 2.0
    tol = 2.0 * ML_PER_PX
    assert row["V_lower_mL"] == pytest.approx(linear_v(Y_LOWER - 0.5), abs=tol)
    assert row["V_total_mL"] == pytest.approx(linear_v(Y_UPPER - 0.5), abs=tol)
    assert row["V_foam_mL"] == pytest.approx(row["V_total_mL"] - row["V_lower_mL"], abs=1e-9)
    assert row["n_lower"] == 81 and row["n_upper"] == 81
    assert row["std_lower_px"] < 1.5
    # Method uncertainty: spread of the per-column rows converted to mL, u = range / (2 sqrt 3).
    assert 0.0 <= row["u_method_lower_mL"] <= 3.0 * ML_PER_PX / (2 * math.sqrt(3)) + 1e-9
    assert math.isfinite(row["u_method_upper_mL"])


def test_process_frame_without_interfaces_gives_nan(calib: Calibration) -> None:
    from cylvision.calibration import build_model_fn

    frame = np.zeros((H, W_FULL, 3), dtype=np.uint8)
    row = process_frame(frame, calib, params_for("lower_darker"), build_model_fn(calib))
    assert math.isnan(row["V_lower_mL"]) and math.isnan(row["V_total_mL"]) and math.isnan(row["V_foam_mL"])
    assert math.isnan(row["y_lower_px"]) and row["n_lower"] == 0
    assert math.isnan(row["u_method_lower_mL"])


# --------------------------------------------------------------------------- video

def test_process_video_nan_rows_on_skipped_and_bad_frames(calib: Calibration) -> None:
    from cylvision.calibration import build_model_fn

    video = FakeVideo(12, 30.0, bad={6})
    seen: list[int] = []
    rows = process_video(video, calib, params_for("lower_darker"), build_model_fn(calib),
                         start=0, end=9, frame_step=3, progress=False, on_row=lambda r: seen.append(r["frame_idx"]))
    assert len(rows) == 10
    assert [r["frame_idx"] for r in rows] == list(range(10))
    assert video.read == [0, 3, 9]                  # 6 failed to decode
    assert seen == [0, 3, 9]
    for r in rows:
        idx = r["frame_idx"]
        assert r["t_sec"] == pytest.approx(idx / 30.0)
        if idx in (0, 3, 9):
            assert math.isfinite(r["V_lower_mL"])
            assert r["V_lower_mL"] == pytest.approx(linear_v(Y_LOWER - idx - 0.5), abs=2.0 * ML_PER_PX)
        else:
            assert math.isnan(r["V_lower_mL"]) and math.isnan(r["y_lower_px"]) and r["n_lower"] == 0
    # Volume rises with the level (one pixel per frame).
    assert rows[9]["V_lower_mL"] > rows[0]["V_lower_mL"]
    # Time scale for a pre-subsampled video, and end=None -> last frame.
    rows2 = process_video(video, calib, params_for("lower_darker"), build_model_fn(calib),
                          start=2, frame_step=5, progress=False, time_scale=60.0)
    assert len(rows2) == 10 and rows2[0]["frame_idx"] == 2
    assert rows2[5]["t_sec"] == pytest.approx(5 * 60.0 / 30.0)
    with pytest.raises(ValueError):
        process_video(video, calib, params_for("lower_darker"), build_model_fn(calib), start=5, end=2)


def test_process_video_progress_prints(calib: Calibration, capsys: pytest.CaptureFixture[str]) -> None:
    from cylvision.calibration import build_model_fn

    process_video(FakeVideo(4, 25.0), calib, params_for("lower_darker"), build_model_fn(calib),
                  start=0, end=3, progress=True)
    out = capsys.readouterr().out
    assert "100.0%" in out and "Batch done" in out


# --------------------------------------------------------------------------- csv / run dir

def test_csv_round_trip(tmp_path: Path) -> None:
    rows = [nan_row(0, 0.0), nan_row(1, 0.5)]
    rows[1].update({"y_lower_px": 249.5, "std_lower_px": 0.4, "n_lower": 81, "y_upper_px": 99.5,
                    "n_upper": 80, "V_lower_mL": 435.0, "V_total_mL": 935.0, "V_foam_mL": 500.0,
                    "u_method_lower_mL": 1.25, "u_method_upper_mL": 0.75})
    path = tmp_path / "sub" / "levels.csv"
    write_csv(rows, path)
    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == CSV_COLUMNS
    back = read_csv(path)
    assert len(back) == 2 and list(back[0]) == CSV_COLUMNS
    assert back[0]["frame_idx"] == 0 and math.isnan(back[0]["V_lower_mL"]) and back[0]["n_lower"] == 0
    for k in CSV_COLUMNS:
        if k in ("frame_idx", "n_lower", "n_upper"):
            assert back[1][k] == rows[1][k] and isinstance(back[1][k], int)
        else:
            assert back[1][k] == pytest.approx(rows[1][k])
    s = summarize_rows(back)
    assert s["n_rows"] == 2 and s["n_lower_found"] == 1
    assert s["V_foam_mL"] == {"min": 500.0, "max": 500.0, "last": 500.0}
    assert s["u_method_lower_mL"]["median"] == pytest.approx(1.25)


def test_run_dir_params_round_trip(tmp_path: Path, calib: Calibration) -> None:
    run = find_run(tmp_path / "runs", Path("some dir") / "pour 10 1000mL.MOV")
    assert run.root == tmp_path / "runs" / "pour 10 1000mL"
    assert not run.root.exists() and not run.has_params()
    p = params_for("lower_brighter", cx=57, T_lower=16.5)
    path = run.save_params(p, frame_step=4)
    assert path == run.params_path and path.exists()
    back, step = run.load_params()
    assert back == p and step == 4
    d = __import__("json").loads(path.read_text(encoding="utf-8"))
    assert d["frame_step"] == 4 and d["polarity"] == "lower_brighter" and d["T_lower"] == 16.5
    run.save_calibration(calib)
    assert run.load_calibration() == calib
    run.save_meta({"a": 1, "arr": np.arange(3), "p": run.root})
    assert run.load_meta()["arr"] == [0, 1, 2]
    assert (run.csv_path.name, run.figure_path.name, run.meta_path.name, run.check_png.name) == (
        "levels.csv", "figure.png", "meta.json", "calibration_check.png")
    # A saved legacy params.json is translated.
    RunDir(tmp_path / "legacy").save_params(p, 1)
    (tmp_path / "legacy" / "params.json").write_text(
        '{"T_pos": 14, "T_neg": 47, "r_neg": 80, "r_pos": 113, "blur_sigma": 5.0, "blur_h": 38, "min_h_pos": 3, "y_top": 34}',
        encoding="utf-8")
    lp, ls = RunDir(tmp_path / "legacy").load_params()
    assert lp.polarity == "lower_darker" and lp.T_lower == 47.0 and ls == 1


# --------------------------------------------------------------------------- figure / canvas

def test_plot_levels_writes_png(tmp_path: Path) -> None:
    rows = []
    for i in range(40):
        r = nan_row(i, i * 0.5)
        if i % 2 == 0:
            V = 200.0 + 5.0 * i
            r.update({"V_lower_mL": V, "V_total_mL": V + 300.0 - 4.0 * i, "V_foam_mL": 300.0 - 4.0 * i})
        rows.append(r)

    class Budget:
        def total(self, V: float) -> float:
            return 2.0 + 0.01 * V

    out = tmp_path / "fig" / "figure.png"
    plot_levels(rows, title="synthetic", out_png=out, budget=Budget())
    assert out.exists() and out.stat().st_size > 5_000
    out2 = tmp_path / "empty.png"
    plot_levels([nan_row(0, 0.0)], title="empty", out_png=out2)
    assert out2.exists()


@pytest.mark.parametrize("show_specimen,inline", [(True, True), (False, True), (True, False), (False, False)])
def test_compose_canvas_fits_screen(monkeypatch: pytest.MonkeyPatch, calib: Calibration,
                                    show_specimen: bool, inline: bool) -> None:
    monkeypatch.setenv("CYLVISION_HEADLESS", "1")
    from cylvision.calibration import build_model_fn
    from cylvision.ui.tuner import compose_canvas

    screen = (960, 540)
    canvas = compose_canvas(synthetic_frame("lower_darker"), calib, params_for("lower_darker"),
                            build_model_fn(calib), show_specimen=show_specimen, inline=inline,
                            screen_size=screen, frame_idx=3, fps=30.0)
    assert canvas.dtype == np.uint8 and canvas.ndim == 3 and canvas.shape[2] == 3
    assert canvas.shape[0] <= screen[1] and canvas.shape[1] <= screen[0]
    assert canvas.shape[:2] == (screen[1], screen[0])
    assert canvas.std() > 10  # something was drawn
    # Upscaled layout on a big screen still fits.
    big = compose_canvas(synthetic_frame("lower_brighter"), calib, params_for("lower_brighter"), None,
                         show_specimen=show_specimen, inline=inline, screen_size=(1920, 1080))
    assert big.shape[:2] == (1080, 1920)
