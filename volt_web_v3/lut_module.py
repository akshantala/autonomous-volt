"""
Autonomous Volt - Torque Lookup Table Module
=============================================

Cleaned from the original Jupyter notebook into an importable module.

Usage:
    from lut_module import build_lut, query_lut

    lut = build_lut()                          # builds from CSVs in data/ if present,
                                               # otherwise falls back to synthetic data
    dT   = lut(speed_kph, grade_pct, steer_deg)   # delta torque correction [lb-ft]
    Tbase = torque_base_lbft(speed_kph)           # flat-road baseline torque [lb-ft]
    Ttot  = Tbase + dT                            # total predicted torque [lb-ft]
"""

from __future__ import annotations

import os
import glob
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter


# ============================================================
# 0) Vehicle parameters & operating limits  (Chevy Bolt EV)
# ============================================================
m     = 1920.0      # vehicle mass [kg]
L     = 2.60        # wheelbase [m]
r_w   = 0.326       # tire radius [m]
g_acc = 9.81        # gravity [m/s^2]
rho   = 1.225       # air density [kg/m^3]
Cd    = 0.308       # drag coefficient
A     = 2.38        # frontal area [m^2]
C_rr  = 0.012       # rolling resistance coefficient
k_B   = 40.0        # linear resistance [N per m/s]
c_turn = 0.04       # turn-resistance coefficient

LBFT_TO_NM = 1.35582

# Operating envelope (from project spec)
V_MAX_KPH = 15.0
GRADE_MIN, GRADE_MAX = -10.0, 10.0
# ±31° (not ±30°) to include real full-lock data from the Volt tests,
# which measured at ±30.1° road-wheel (451° wheel / 15 ratio).
STEER_MIN, STEER_MAX = -31.0, 31.0

# Torque limits
GEAR_RATIO       = 7.05
DRIVETRAIN_EFF   = 0.95
MOTOR_T_MAX_LBFT = 120.0
T_MAX_LBFT = MOTOR_T_MAX_LBFT * GEAR_RATIO * DRIVETRAIN_EFF
T_MIN_LBFT = -T_MAX_LBFT

# Spec cap (from progress reports)
T_SPEC_CAP_LBFT = 185.0

# Simulation constants (used by run_controller_comparison)
SIM_DT = 0.05                # [s] simulation timestep
TORQUE_SLEW_LBFTpS = 800.0   # max torque rate-of-change
STEER_SLEW_DEGpS   = 15.0    # max steering rate (matches design spec)


# ============================================================
# 1) Road-load physics
# ============================================================
def road_load_force(v_ms: float, grade_pct: float) -> float:
    theta = np.arctan(grade_pct / 100.0)
    f_roll  = m * g_acc * C_rr * np.cos(theta)
    f_B     = k_B * v_ms
    f_aero  = 0.5 * rho * Cd * A * v_ms**2
    f_grade = m * g_acc * np.sin(theta)
    return f_roll + f_B + f_aero + f_grade


def turn_resistance(v_ms: float, steer_deg: float) -> float:
    delta = np.deg2rad(steer_deg)
    kappa = np.tan(delta) / L
    a_y   = v_ms**2 * abs(kappa)
    return c_turn * m * a_y


def total_resistive_force(v_ms: float, grade_pct: float, steer_deg: float) -> float:
    return road_load_force(v_ms, grade_pct) + turn_resistance(v_ms, steer_deg)


def required_torque_lbft_for_hold(v_kph: float, grade_pct: float, steer_deg: float) -> float:
    v_ms  = max(0.0, v_kph / 3.6)
    F_tot = total_resistive_force(v_ms, grade_pct, steer_deg)
    T_nm  = F_tot * r_w
    return float(T_nm / LBFT_TO_NM)


def torque_base_lbft(v_kph: float) -> float:
    """Baseline torque for flat, straight-line driving (plus a 10 N·m reserve)."""
    v_ms  = max(0.0, v_kph / 3.6)
    F_flat = road_load_force(v_ms, grade_pct=0.0)
    T_nm   = F_flat * r_w
    reserve_nm = 10.0
    return float((T_nm + reserve_nm) / LBFT_TO_NM)


# ============================================================
# 2) Data loaders - real CSVs (preferred) or synthetic fallback
# ============================================================
def _synthetic_dataset(N: int = 8000, seed: int = 11):
    rng = np.random.default_rng(seed)
    t = np.arange(N) * 0.1

    v_kph = 8.0 + 4.0 * np.sin(0.02 * t + 0.5) + 1.5 * np.sin(0.15 * t)
    v_kph = np.clip(v_kph, 0.5, V_MAX_KPH)

    grade_pct = (5.0 * np.sin(0.015 * t)
                 + 3.0 * np.sin(0.04 * t + 0.7)
                 + rng.normal(0.0, 0.5, size=N))
    grade_pct = np.clip(grade_pct, GRADE_MIN, GRADE_MAX)

    steer_deg = (15.0 * np.sin(0.08 * t)
                 + 8.0 * np.sin(0.20 * t + 0.8)
                 + rng.normal(0.0, 1.0, size=N))
    steer_deg = np.clip(steer_deg, STEER_MIN, STEER_MAX)

    T_req = np.array([
        required_torque_lbft_for_hold(v, gr, st)
        for v, gr, st in zip(v_kph, grade_pct, steer_deg)
    ])
    T_meas = T_req + rng.normal(0.0, 2.0, size=N)
    T_base = np.array([torque_base_lbft(v) for v in v_kph])
    dT     = T_meas - T_base
    return v_kph, grade_pct, steer_deg, dT


# Column-name aliases so the loader is forgiving of how the CSVs are labeled.
# Matching is case-insensitive and does a "contains" check as a fallback —
# real CAN exports use verbose names like
# "Actual Axle Torque (Value [Nm])" / "Steering Wheel Angle (Value [deg])" /
# "Vehicle Speed Average Driven (Value [km / h])"
_SPEED_COLS = [
    "speed_kph", "speed", "v_kph", "vehicle_speed_kph",
    "Vehicle Speed Average Driven (Value [km / h])",
]
_GRADE_COLS = [
    "grade_pct", "grade", "incline", "incline_pct", "road_grade_pct",
]
# NOTE: "Steering Wheel Angle" is at the steering wheel, not the road wheel.
# The loader divides by STEERING_RATIO (15:1 for the Volt) when it detects
# a "wheel angle" column name.
_STEER_COLS = [
    "steering_angle_deg", "steer_deg", "steering_deg", "steer", "steering_angle",
    "road_wheel_deg", "road_wheel_angle",
    "Steering Wheel Angle (Value [deg])",
]
_TORQUE_COLS = [
    "torque_cmd_lbf_ft", "torque_lbft", "torque_cmd", "axle_torque_lbft", "torque",
    "torque_cmd_nm", "torque_nm", "axle_torque_nm",
    "Actual Axle Torque (Value [Nm])",
]

# Vehicle-specific constants
STEERING_RATIO = 15.0  # Chevy Bolt: ~15:1 steering wheel to road wheel


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Match exact (case-insensitive) first, then fallback to substring match."""
    lowmap = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lowmap:
            return lowmap[cand.lower()]
    # Substring fallback
    for cand in candidates:
        low = cand.lower()
        for c in df.columns:
            if low in c.lower():
                return c
    return None


def _load_csvs_from_folder(folder: str, file_diagnostics: list | None = None):
    """
    Load every CSV/XLSX in `folder` and concatenate into one (speed, grade, steer, dT) dataset.

    - Torque in N·m (column name contains "nm") is auto-converted to lb-ft.
    - Steering column labeled as a STEERING WHEEL angle (contains "wheel") is
      divided by STEERING_RATIO to get the road-wheel angle the LUT expects.
    - Grade is optional, defaults to 0.
    - Rows outside the LUT envelope (speed/grade/steer limits) are dropped.

    If `file_diagnostics` is provided, per-file stats are appended for display.
    Returns None if no files or no usable rows.
    """
    if not os.path.isdir(folder):
        return None

    files = sorted(
        glob.glob(os.path.join(folder, "*.csv"))
        + glob.glob(os.path.join(folder, "*.xlsx"))
    )
    if not files:
        return None

    frames = []
    for path in files:
        fname = os.path.basename(path)
        diag = {"file": fname, "raw_rows": 0, "valid_rows": 0, "kept_rows": 0,
                "notes": [], "status": "ok"}
        try:
            if path.lower().endswith(".csv"):
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)
        except Exception as e:
            diag["status"] = "read-error"
            diag["notes"].append(str(e))
            if file_diagnostics is not None:
                file_diagnostics.append(diag)
            print(f"[LUT] skipping {fname}: {e}")
            continue

        diag["raw_rows"] = len(df)

        sp_c = _pick_col(df, _SPEED_COLS)
        st_c = _pick_col(df, _STEER_COLS)
        tq_c = _pick_col(df, _TORQUE_COLS)
        gr_c = _pick_col(df, _GRADE_COLS)

        if not (sp_c and st_c and tq_c):
            missing = [n for n, c in [("speed", sp_c), ("steer", st_c),
                                       ("torque", tq_c)] if not c]
            diag["status"] = "missing-cols"
            diag["notes"].append(f"missing: {', '.join(missing)}")
            if file_diagnostics is not None:
                file_diagnostics.append(diag)
            print(f"[LUT] skipping {fname}: missing columns ({missing})")
            continue

        sub = pd.DataFrame({
            "v":  pd.to_numeric(df[sp_c], errors="coerce"),
            "st": pd.to_numeric(df[st_c], errors="coerce"),
            "T":  pd.to_numeric(df[tq_c], errors="coerce"),
            "gr": pd.to_numeric(df[gr_c], errors="coerce") if gr_c else 0.0,
        }).dropna(subset=["v", "st", "T"])

        diag["valid_rows"] = len(sub)

        # Torque unit handling: N·m -> lb-ft
        if "nm" in tq_c.lower() or "n_m" in tq_c.lower() or "n-m" in tq_c.lower():
            sub["T"] = sub["T"] / LBFT_TO_NM
            diag["notes"].append("T: Nm→lb-ft")

        # Steering angle handling: if column name references the STEERING WHEEL
        # (driver input), divide by steering ratio to get road-wheel angle.
        if "wheel" in st_c.lower():
            sub["st"] = sub["st"] / STEERING_RATIO
            diag["notes"].append(f"steer: wheel/{STEERING_RATIO:.1f}→road")

        sub["source"] = fname
        frames.append(sub)

        # Row-level summary before envelope clipping (so user sees full picture)
        if file_diagnostics is not None:
            diag["speed_range"] = (float(sub["v"].min()), float(sub["v"].max()))
            diag["steer_range"] = (float(sub["st"].min()), float(sub["st"].max()))
            diag["torque_range_lbft"] = (float(sub["T"].min()), float(sub["T"].max()))
            diag["torque_mean_lbft"] = float(sub["T"].mean())
            diag["torque_std_lbft"] = float(sub["T"].std())
            file_diagnostics.append(diag)

    if not frames:
        return None

    data = pd.concat(frames, ignore_index=True)
    n_before = len(data)

    # Clip to envelope so out-of-range rows don't skew the table
    clipped = data[
        (data["v"].between(0, V_MAX_KPH))
        & (data["gr"].between(GRADE_MIN, GRADE_MAX))
        & (data["st"].between(STEER_MIN, STEER_MAX))
    ]
    n_after = len(clipped)
    if file_diagnostics is not None and n_before > 0:
        # Record how many rows were clipped per file (informational)
        for d in file_diagnostics:
            if d["status"] == "ok":
                kept = int((clipped["source"] == d["file"]).sum())
                d["kept_rows"] = kept

    if n_after == 0:
        return None

    data = clipped
    v_kph     = data["v"].to_numpy()
    grade_pct = data["gr"].to_numpy() if np.ndim(data["gr"].to_numpy()) else np.zeros(len(data))
    steer_deg = data["st"].to_numpy()
    T_meas    = data["T"].to_numpy()
    T_base    = np.array([torque_base_lbft(v) for v in v_kph])
    dT        = T_meas - T_base

    print(f"[LUT] loaded {len(data)} samples from {len(frames)} CSV/XLSX file(s)")
    return v_kph, grade_pct, steer_deg, dT


# ============================================================
# 3) The 3-D lookup table
# ============================================================
class DeltaTLookup:
    def __init__(self, v_knots, g_knots, s_knots):
        self.vk = np.array(v_knots, dtype=float)
        self.gk = np.array(g_knots, dtype=float)
        self.sk = np.array(s_knots, dtype=float)
        self.grid = np.full((len(self.vk), len(self.gk), len(self.sk)), np.nan)
        self._interp = None

    @staticmethod
    def _nearest_idx(knots, vals):
        return np.abs(knots.reshape(1, -1) - vals.reshape(-1, 1)).argmin(axis=1)

    def fit_from_samples(self, v_kph, grade_pct, steer_deg, dT_lbft, smooth_sigma: float = 0.7):
        vi = self._nearest_idx(self.vk, np.asarray(v_kph, dtype=float))
        gi = self._nearest_idx(self.gk, np.asarray(grade_pct, dtype=float))
        si = self._nearest_idx(self.sk, np.asarray(steer_deg, dtype=float))

        sum_grid = np.zeros_like(self.grid)
        cnt_grid = np.zeros_like(self.grid)
        for i, j, k, val in zip(vi, gi, si, dT_lbft):
            sum_grid[i, j, k] += val
            cnt_grid[i, j, k] += 1.0

        with np.errstate(invalid="ignore", divide="ignore"):
            self.grid = np.where(cnt_grid > 0, sum_grid / np.maximum(cnt_grid, 1.0), np.nan)

        total = self.grid.size
        filled = int(np.count_nonzero(~np.isnan(self.grid)))
        print(f"[LUT] coverage: {filled}/{total} cells populated ({100*filled/total:.1f}%)")

        self.grid = self._nn_fill(self.grid, passes=5, fallback=0.0)
        if smooth_sigma > 0:
            self.grid = gaussian_filter(self.grid, sigma=smooth_sigma)

        self._build_interpolator()

    @staticmethod
    def _nn_fill(vol, passes=5, fallback=0.0):
        vol = vol.copy()
        for _ in range(passes):
            filled = vol.copy()
            for idx in np.ndindex(vol.shape):
                if np.isnan(vol[idx]):
                    neighbors = []
                    for axis in range(3):
                        for off in (-1, 1):
                            j = list(idx)
                            j[axis] = int(np.clip(j[axis] + off, 0, vol.shape[axis] - 1))
                            val = vol[tuple(j)]
                            if not np.isnan(val):
                                neighbors.append(val)
                    if neighbors:
                        filled[idx] = float(np.mean(neighbors))
            vol = filled
        vol[np.isnan(vol)] = fallback
        return vol

    def _build_interpolator(self):
        self._interp = RegularGridInterpolator(
            (self.vk, self.gk, self.sk),
            self.grid,
            method="linear",
            bounds_error=False,
            fill_value=None,  # nearest-edge extrapolation
        )

    def __call__(self, v, g, s):
        if self._interp is None:
            raise RuntimeError("LUT not fitted yet.")
        # Clip the query to the envelope so we never extrapolate wildly
        v = float(np.clip(v, 0.0, V_MAX_KPH))
        g = float(np.clip(g, GRADE_MIN, GRADE_MAX))
        s = float(np.clip(s, STEER_MIN, STEER_MAX))
        return float(self._interp(np.array([[v, g, s]]))[0])

    def save(self, path: str):
        np.savez(path, vk=self.vk, gk=self.gk, sk=self.sk, grid=self.grid)

    @classmethod
    def load(cls, path: str):
        d = np.load(path)
        obj = cls(d["vk"], d["gk"], d["sk"])
        obj.grid = d["grid"]
        obj._build_interpolator()
        return obj


# ============================================================
# 4) Build function (one call, used by the checklist app)
# ============================================================
def build_lut(data_folder: str | None = None, verbose: bool = True,
              force_synthetic: bool = False) -> DeltaTLookup:
    """
    Build the LUT from real CSV/XLSX files in `data_folder` if any exist,
    otherwise fall back to synthetic data. Drop real files into that folder later —
    no code changes needed.

    The returned LUT has these informational attributes attached:
        .data_source       - "synthetic" or "CSV/XLSX from <folder>"
        .n_samples         - number of training samples used
        .file_diagnostics  - per-file load report (list of dicts), empty for synthetic

    Args:
        force_synthetic: If True, skip loading real CSVs and build the LUT
            from synthetic data. Useful for the controller-comparison demo,
            where the simulation uses the same idealized physics that
            generated the synthetic training data (so the comparison is fair).
    """
    source = "synthetic"
    loaded = None
    file_diagnostics: list = []
    if data_folder and not force_synthetic:
        loaded = _load_csvs_from_folder(data_folder, file_diagnostics=file_diagnostics)
        if loaded is not None:
            source = f"CSV/XLSX from {data_folder}"

    if loaded is None:
        if verbose:
            if force_synthetic:
                print("[LUT] force_synthetic=True — using synthetic dataset")
            else:
                print("[LUT] no real data files found — using synthetic dataset")
        v_kph, grade_pct, steer_deg, dT = _synthetic_dataset()
    else:
        v_kph, grade_pct, steer_deg, dT = loaded

    v_knots = np.arange(0, V_MAX_KPH + 1, 1.0)
    g_knots = np.arange(GRADE_MIN, GRADE_MAX + 1, 2.0)
    s_knots = np.arange(STEER_MIN, STEER_MAX + 1, 5.0)

    if verbose:
        print(f"[LUT] data source: {source}")
        print(f"[LUT] knots: {len(v_knots)} x {len(g_knots)} x {len(s_knots)} = "
              f"{len(v_knots)*len(g_knots)*len(s_knots)} cells")

    lut = DeltaTLookup(v_knots, g_knots, s_knots)
    lut.fit_from_samples(v_kph, grade_pct, steer_deg, dT, smooth_sigma=0.7)
    lut.data_source = source
    lut.n_samples = len(v_kph)
    lut.file_diagnostics = file_diagnostics
    return lut


def predict_torque(lut: DeltaTLookup, v_kph: float, grade_pct: float, steer_deg: float) -> dict:
    """Convenience wrapper: returns baseline, delta, total, and safety flag."""
    base  = torque_base_lbft(v_kph)
    delta = lut(v_kph, grade_pct, steer_deg)
    total = base + delta
    return {
        "baseline_lbft":  base,
        "delta_lbft":     delta,
        "total_lbft":     total,
        "over_spec_cap":  abs(total) > T_SPEC_CAP_LBFT,
        "over_motor_cap": abs(total) > T_MAX_LBFT,
    }


# ============================================================
# 5) Closed-loop controller simulation
# ------------------------------------------------------------
# Runs the same baseline-vs-LUT comparison shown in the
# Jupyter notebook. Used by the web app to demonstrate the
# performance improvement of the LUT-based controller.
# ============================================================

def _clamp_inputs(v_set_kph, grade_pct, steer_deg):
    return (
        min(v_set_kph, V_MAX_KPH),
        float(np.clip(grade_pct, GRADE_MIN, GRADE_MAX)),
        float(np.clip(steer_deg, STEER_MIN, STEER_MAX)),
    )


def _slew(prev, cmd, rate_per_s, dt=SIM_DT):
    """Rate limiter — caps how fast a signal can change per timestep."""
    return float(np.clip(cmd, prev - rate_per_s * dt, prev + rate_per_s * dt))


def _step_bicycle(x, y, psi, v_ms, T_lbft, grade_pct, steer_deg, dt=SIM_DT):
    """Advance vehicle state one timestep using a bicycle model."""
    T_nm  = T_lbft * LBFT_TO_NM
    F_drv = T_nm / r_w
    F_res = total_resistive_force(v_ms, grade_pct, steer_deg)
    a_ms2 = (F_drv - F_res) / m
    v_ms  = max(0.0, v_ms + a_ms2 * dt)

    delta  = np.deg2rad(steer_deg)
    psi   += (v_ms / L) * np.tan(delta) * dt
    x     += v_ms * np.cos(psi) * dt
    y     += v_ms * np.sin(psi) * dt
    return x, y, psi, v_ms, a_ms2


def _ctrl_baseline(v_kph, v_set_kph, grade_pct, steer_deg, state):
    """
    Baseline PI controller (NO lookup table). Represents the 'before' case —
    a controller that only knows the flat-road physics model.
    """
    Kp, Ki = 1.6, 0.25
    e = v_set_kph - v_kph
    state["i"] += e * SIM_DT
    T = torque_base_lbft(v_kph) + (Kp * e + Ki * state["i"])
    return float(np.clip(T, T_MIN_LBFT, T_MAX_LBFT))


def _make_ctrl_adaptive(lut: DeltaTLookup):
    """LUT-based feedforward + small PI on residual error, with anti-windup."""
    Kp, Ki = 1.4, 0.30
    state = {"i": 0.0}

    def ctrl(v_kph, v_set_kph, grade_pct, steer_deg):
        e = v_set_kph - v_kph
        Tff = torque_base_lbft(v_kph) + lut(v_kph, grade_pct, steer_deg)
        T_pi = Kp * e + Ki * state["i"]
        T_raw = Tff + T_pi
        T_sat = float(np.clip(T_raw, T_MIN_LBFT, T_MAX_LBFT))
        # Anti-windup: only integrate when not saturated (or relieving saturation)
        if (T_sat == T_raw
            or (T_sat == T_MAX_LBFT and e < 0)
            or (T_sat == T_MIN_LBFT and e > 0)):
            state["i"] += e * SIM_DT
        return T_sat

    return ctrl


def _compute_metrics(history: np.ndarray) -> dict:
    """Compute speed-tracking, jerk, and slew-rate metrics from a sim run."""
    v, vset = history[:, 1], history[:, 2]
    a, j, T = history[:, 6], history[:, 7], history[:, 5]

    err = v - vset
    inband   = float(np.mean(np.abs(err) <= 0.5) * 100.0)
    rmse_v   = float(np.sqrt(np.mean(err**2)))
    max_err  = float(np.max(np.abs(err)))
    rms_a    = float(np.sqrt(np.mean(a**2)))
    rms_j    = float(np.sqrt(np.mean(j**2)))
    p95_j    = float(np.percentile(np.abs(j), 95))
    dTdt     = np.diff(T) / SIM_DT
    rms_slew = float(np.sqrt(np.mean(dTdt**2)))

    return {
        "inband_pct": inband, "rmse_v_kph": rmse_v, "max_err_kph": max_err,
        "rms_accel_mps2": rms_a, "rms_jerk_mps3": rms_j,
        "p95_jerk_mps3": p95_j, "rms_torque_slew_lbft_per_s": rms_slew,
    }


def run_controller_comparison(lut: DeltaTLookup, duration_s: float = 30.0) -> dict:
    """
    Run the baseline-vs-adaptive controller comparison from the Jupyter notebook.

    Drives both controllers through the same 30-second cycle with varying speed
    setpoints, grade changes, and steering inputs. Returns time-series histories
    and performance metrics for both controllers, plus a percentage-improvement
    summary.

    Returns:
        {
            "baseline":  {"history": ndarray (N, 8), "metrics": {...}},
            "adaptive":  {"history": ndarray (N, 8), "metrics": {...}},
            "improvement": {
                "inband_delta_pct":   float,    # absolute change (percentage points)
                "rmse_v_change_pct":  float,    # % reduction (positive = better)
                "rms_jerk_change_pct": float,
                "torque_slew_change_pct": float,
            },
            "history_columns": [...],
        }

    History columns: [time, v_actual_kph, v_target_kph, grade_pct, steer_deg,
                      torque_lbft, accel_mps2, jerk_mps3]
    """
    N = int(duration_s / SIM_DT)
    history_columns = ["time_s", "v_actual_kph", "v_target_kph", "grade_pct",
                       "steer_deg", "torque_lbft", "accel_mps2", "jerk_mps3"]

    def run_with(controller, ctrl_state=None):
        x = y = psi = 0.0
        v_ms   = 8.0 / 3.6   # start at 8 kph
        T_prev = 0.0
        steer  = 0.0
        a_prev = 0.0
        history = np.zeros((N, 8))

        for k in range(N):
            t = k * SIM_DT
            # Drive cycle: speed setpoint sinusoid 6–10 kph
            v_set = 8.0 + 2.0 * np.sin(0.30 * t)
            # Grade with negative segments (downhill portions)
            grade = 4.0 * np.sin(0.15 * t) + 2.0 * np.sin(0.08 * t)
            # Steering input with two-frequency content
            steer_cmd = 15.0 * np.sin(0.50 * t) + 5.0 * np.sin(0.25 * t)

            v_set, grade, steer_cmd = _clamp_inputs(v_set, grade, steer_cmd)
            steer = _slew(steer, steer_cmd, STEER_SLEW_DEGpS)

            # Two controller-call signatures (baseline takes a state dict)
            if ctrl_state is not None:
                T_cmd = controller(v_ms * 3.6, v_set, grade, steer, ctrl_state)
            else:
                T_cmd = controller(v_ms * 3.6, v_set, grade, steer)
            T = _slew(T_prev, T_cmd, TORQUE_SLEW_LBFTpS)

            x, y, psi, v_ms, a = _step_bicycle(x, y, psi, v_ms, T, grade, steer)
            jerk = (a - a_prev) / SIM_DT
            history[k] = (t, v_ms * 3.6, v_set, grade, steer, T, a, jerk)
            T_prev = T
            a_prev = a

        return history

    # Run baseline
    base_state = {"i": 0.0}
    base_hist = run_with(_ctrl_baseline, ctrl_state=base_state)

    # Run adaptive (LUT-based)
    adap_hist = run_with(_make_ctrl_adaptive(lut))

    base_m = _compute_metrics(base_hist)
    adap_m = _compute_metrics(adap_hist)

    # Improvement summary
    inband_delta = adap_m["inband_pct"] - base_m["inband_pct"]
    def pct_change(old, new):
        return float(100.0 * (1.0 - new / old)) if old != 0 else 0.0

    return {
        "baseline": {"history": base_hist, "metrics": base_m},
        "adaptive": {"history": adap_hist, "metrics": adap_m},
        "improvement": {
            "inband_delta_pct":      inband_delta,
            "rmse_v_change_pct":     pct_change(base_m["rmse_v_kph"], adap_m["rmse_v_kph"]),
            "rms_jerk_change_pct":   pct_change(base_m["rms_jerk_mps3"], adap_m["rms_jerk_mps3"]),
            "torque_slew_change_pct": pct_change(base_m["rms_torque_slew_lbft_per_s"],
                                                  adap_m["rms_torque_slew_lbft_per_s"]),
        },
        "history_columns": history_columns,
        "duration_s": duration_s,
    }


# ============================================================
# 6) Real-data analysis - compute jerk and torque slew rate from
#    recorded test files. Used by the Real Data Analysis tab.
# ============================================================

def _smooth_and_diff(values: np.ndarray, t: np.ndarray, win: int = 21):
    """Light Savitzky-Golay smoothing then derivative. Returns derivative array."""
    from scipy.signal import savgol_filter
    win = max(5, min(win, len(values) // 2 * 2 + 1))
    if win < 5 or len(values) < win:
        return np.gradient(values, t)
    smoothed = savgol_filter(values, win, 3)
    return np.gradient(smoothed, t)


def analyze_real_data_files(data_folder: str) -> dict:
    """
    Walk every CSV/XLSX file in data_folder and compute jerk + torque slew
    rate from the recorded vehicle data. Returns per-file stats and aggregate
    stats across all files.

    Methodology:
        - Speed [km/h] -> m/s -> Savitzky-Golay smooth (window 21, polyorder 3)
          -> first derivative = acceleration -> smooth -> second derivative = jerk
        - Torque [Nm] -> lb-ft -> SG smooth -> first derivative = slew rate
        - Stats trimmed to middle 95% of samples to ignore startup/end transients

    Limitations:
        - Numerical differentiation amplifies noise. The SG filter mitigates
          this but cannot eliminate it. Treat absolute jerk numbers as
          order-of-magnitude estimates, not lab-grade measurements.
        - "Acceleration" derived from speed is not the same as inertial-
          accelerometer data. The Bolt has an IMU but those channels were
          not captured in these CSVs.

    Returns:
        {
            "files": [
                {
                    "name": str,
                    "n_samples": int,
                    "duration_s": float,
                    "mean_speed_kph": float,
                    "rms_jerk_mps3": float,
                    "p95_jerk_mps3": float,
                    "rms_torque_slew_lbft_per_s": float,
                    "p95_torque_slew_lbft_per_s": float,
                    "max_torque_lbft": float,
                    "_traces": {
                        "t": np.ndarray, "v_kph": np.ndarray,
                        "T_lbft": np.ndarray, "jerk": np.ndarray,
                    },
                },
                ...
            ],
            "aggregate": {
                "n_files": int,
                "total_samples": int,
                "mean_rms_jerk": float,
                "median_rms_jerk": float,
                "mean_rms_slew": float,
                "median_rms_slew": float,
            },
        }
    """
    if not os.path.isdir(data_folder):
        return {"files": [], "aggregate": {}}

    files = sorted(
        glob.glob(os.path.join(data_folder, "*.csv"))
        + glob.glob(os.path.join(data_folder, "*.xlsx"))
    )

    results = []
    for path in files:
        try:
            if path.lower().endswith(".csv"):
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)
        except Exception:
            continue

        # Find columns using the same alias system
        sp_c = _pick_col(df, _SPEED_COLS)
        tq_c = _pick_col(df, _TORQUE_COLS)
        # Time column - look for "Time (abs)" or "time"
        time_col = None
        for c in df.columns:
            lc = c.lower()
            if "time" in lc and "abs" in lc:
                time_col = c; break
        if time_col is None:
            for c in df.columns:
                if c.lower().strip() == "time":
                    time_col = c; break

        if not (sp_c and tq_c and time_col):
            continue

        t = pd.to_numeric(df[time_col], errors="coerce").to_numpy()
        v_kph = pd.to_numeric(df[sp_c], errors="coerce").to_numpy()
        T_raw = pd.to_numeric(df[tq_c], errors="coerce").to_numpy()

        valid = ~(np.isnan(t) | np.isnan(v_kph) | np.isnan(T_raw))
        t = t[valid]; v_kph = v_kph[valid]; T_raw = T_raw[valid]
        if len(t) < 50:
            continue

        # Sort and rebase time to zero
        order = np.argsort(t)
        t = t[order]; v_kph = v_kph[order]; T_raw = T_raw[order]
        t = t - t[0]

        # Convert torque to lb-ft if column is N·m
        if "nm" in tq_c.lower() or "n_m" in tq_c.lower() or "n-m" in tq_c.lower():
            T_lbft = T_raw / LBFT_TO_NM
        else:
            T_lbft = T_raw

        # Compute jerk: speed (m/s) -> accel -> jerk, with SG smoothing
        v_ms = v_kph / 3.6
        accel = _smooth_and_diff(v_ms, t)
        jerk = _smooth_and_diff(accel, t)
        # Torque slew
        slew_rate = _smooth_and_diff(T_lbft, t)

        # Trim to middle 95% to ignore startup/shutdown transients
        n = len(t); lo = int(n * 0.025); hi = n - lo
        j_trim = np.abs(jerk[lo:hi])
        s_trim = np.abs(slew_rate[lo:hi])

        rms_j = float(np.sqrt(np.mean(j_trim**2)))
        p95_j = float(np.percentile(j_trim, 95))
        rms_s = float(np.sqrt(np.mean(s_trim**2)))
        p95_s = float(np.percentile(s_trim, 95))

        name = os.path.basename(path)
        clean = name.replace(".csv", "").replace(".xlsx", "")
        clean = clean.replace("__1_", "").replace("__2_", "").replace("Truncated_-_", "")

        results.append({
            "name": clean,
            "n_samples": int(len(t)),
            "duration_s": float(t[-1] - t[0]),
            "mean_speed_kph": float(np.mean(v_kph)),
            "rms_jerk_mps3": rms_j,
            "p95_jerk_mps3": p95_j,
            "rms_torque_slew_lbft_per_s": rms_s,
            "p95_torque_slew_lbft_per_s": p95_s,
            "max_torque_lbft": float(np.max(np.abs(T_lbft))),
            "_traces": {
                "t": t, "v_kph": v_kph, "T_lbft": T_lbft, "jerk": jerk,
            },
        })

    if not results:
        return {"files": [], "aggregate": {}}

    rms_jerks = [r["rms_jerk_mps3"] for r in results]
    rms_slews = [r["rms_torque_slew_lbft_per_s"] for r in results]
    return {
        "files": results,
        "aggregate": {
            "n_files": len(results),
            "total_samples": int(sum(r["n_samples"] for r in results)),
            "mean_rms_jerk": float(np.mean(rms_jerks)),
            "median_rms_jerk": float(np.median(rms_jerks)),
            "mean_rms_slew": float(np.mean(rms_slews)),
            "median_rms_slew": float(np.median(rms_slews)),
        },
    }


# Quick self-test
if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    lut = build_lut(data_folder=os.path.join(here, "data"))
    for v, g, s, desc in [
        (5.0,  0.0,  0.0, "5 kph, flat, straight"),
        (10.0, 5.0,  0.0, "10 kph, 5% uphill, straight"),
        (10.0, -5.0, 0.0, "10 kph, 5% downhill, straight"),
        (8.0,  3.0, 15.0, "8 kph, 3% uphill, 15 deg turn"),
    ]:
        r = predict_torque(lut, v, g, s)
        print(f"  {desc:35s} -> base={r['baseline_lbft']:6.2f}  "
              f"dT={r['delta_lbft']:+6.2f}  total={r['total_lbft']:6.2f} lb-ft")
