"""Monte-Carlo engine: random scenarios, ROC / P_D / P_FA, RMSE, localisation error, threshold calibration."""
from __future__ import annotations

from dataclasses import dataclass, replace, field
import json
import os
import numpy as np

from .config import C, SystemConfig, Scenario, ProcConfig, db2lin
from .array import BeamSet, build_beams
from .channel import simulate_echo, geometry, building_offset_db, simulate_reference
from .processing import process, range_axis, velocity_axis, range_velocity_map, _parabolic


# ------------------------------------------------------------------------------ random scenarios
@dataclass
class Region:
    """Volume in which Interface 1 drops the UAV (in front of both arrays)."""
    x: tuple = (0.0, 200.0)       # between BsA and BsB (default baseline 200 m)
    y: tuple = (150.0, 500.0)     # distance in front of the baseline
    z: tuple = (20.0, 100.0)      # height
    vmax: float = 20.0            # max speed [m/s]


def random_scenario(rng: np.random.Generator, region: Region, base: Scenario | None = None,
                    present: bool = True, **overrides) -> Scenario:
    """Random UAV position, speed and heading inside `region`."""
    base = base or Scenario()
    pos = (rng.uniform(*region.x), rng.uniform(*region.y), rng.uniform(*region.z))
    speed = rng.uniform(0.0, region.vmax)
    az, el = rng.uniform(0, 2 * np.pi), rng.uniform(-np.radians(20), np.radians(20))
    vel = (speed * np.cos(el) * np.cos(az), speed * np.cos(el) * np.sin(az), speed * np.sin(el))
    return replace(base, pos=pos, vel=vel, present=present, **overrides)


@dataclass
class Trials:
    """Per-trial outputs of a Monte-Carlo run (arrays of length n)."""
    p_paf: np.ndarray
    p_cfar: np.ndarray
    present: np.ndarray
    d_true: np.ndarray
    v_true: np.ndarray
    d_hat: np.ndarray
    v_hat: np.ndarray
    pos_true: np.ndarray
    pos_hat: np.ndarray
    beam_hit: np.ndarray      # estimated beam is the one with the highest gain toward the true direction
    snr_db: np.ndarray

    @property
    def d_err(self):
        return self.d_hat - self.d_true

    @property
    def v_err(self):
        return self.v_hat - self.v_true

    @property
    def pos_err(self):
        return np.linalg.norm(self.pos_hat - self.pos_true, axis=1)


def run_trials(n: int, cfg: SystemConfig, pc: ProcConfig, beams: BeamSet, rng: np.random.Generator,
               scn_fn, progress=None, buildings_fn=None, ref_frames: int = 0) -> Trials:
    """Run n trials; scn_fn(rng) -> Scenario for each trial.
    buildings_fn(rng) -> list of Building: a NEW random set of buildings in every trial.
    ref_frames > 0: also record a reference scan of the empty scene (average of that many frames) and subtract it."""
    out = {k: [] for k in ["p_paf", "p_cfar", "present", "d_true", "v_true", "d_hat", "v_hat",
                           "pos_true", "pos_hat", "beam_hit", "snr_db"]}
    for i in range(n):
        scn = scn_fn(rng)
        bld = buildings_fn(rng) if buildings_fn else ()
        off = building_offset_db(cfg, scn) if len(bld) else None
        e = simulate_echo(cfg, scn, beams, rng, buildings=bld, snr_offset_db=off)
        z_ref = (simulate_reference(cfg, scn, beams, rng, bld, m=ref_frames, snr_offset_db=off)
                 if (len(bld) and ref_frames > 0) else None)
        r = process(e.Z, cfg, beams, e.active, pc, z_ref)
        t = e.truth
        out["p_paf"].append(r.p_paf)
        out["p_cfar"].append(r.p_cfar)
        out["present"].append(scn.present)
        out["d_true"].append(t.d)
        out["v_true"].append(t.v_bi)
        out["d_hat"].append(r.d_hat)
        out["v_hat"].append(r.v_hat)
        out["pos_true"].append(t.pos if t.pos is not None else np.full(3, np.nan))
        out["pos_hat"].append(r.pos_hat)
        out["snr_db"].append(t.snr_best_db)
        if scn.present:
            best = int(np.argmax(e.gains))
            out["beam_hit"].append(r.beam_idx == best)
        else:
            out["beam_hit"].append(False)
        if progress is not None:
            progress((i + 1) / n)
    return Trials(**{k: np.array(v) for k, v in out.items()})


# --------------------------------------------------------------------------------- detection metrics
def roc_curve(stat_h0: np.ndarray, stat_h1: np.ndarray, n_points: int = 400):
    """P_FA / P_D for every threshold (exceedance probability)."""
    lo = min(stat_h0.min(), stat_h1.min())
    hi = max(stat_h0.max(), stat_h1.max())
    thr = np.linspace(lo, hi, n_points)
    pfa = (stat_h0[None, :] > thr[:, None]).mean(axis=1)
    pd = (stat_h1[None, :] > thr[:, None]).mean(axis=1)
    return thr, pfa, pd


def threshold_for_pfa(stat_h0: np.ndarray, pfa: float) -> float:
    """Empirical threshold such that P(stat > thr | H0) = pfa."""
    return float(np.quantile(stat_h0, 1.0 - pfa))


def auc(pfa: np.ndarray, pd: np.ndarray) -> float:
    order = np.argsort(pfa)
    return float(np.trapezoid(pd[order], pfa[order]))


def detection_rate(stat: np.ndarray, thr: float) -> float:
    return float((stat > thr).mean())


def rmse(err: np.ndarray, mask: np.ndarray | None = None) -> float:
    e = err if mask is None else err[mask]
    e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan")


# ----------------------------------------------------------------------------------- calibration
CAL_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "calibration.json")


def calibrate(cfg: SystemConfig, pc: ProcConfig, n: int = 1500, pfa: float = 0.03, seed: int = 7,
              region: Region | None = None, base: Scenario | None = None, progress=None) -> dict:
    """Noise-only runs -> thresholds giving the requested system false-alarm probability for both detectors."""
    beams = build_beams(cfg)
    rng = np.random.default_rng(seed)
    base = base or Scenario()
    tr = run_trials(n, cfg, pc, beams, rng, lambda g: replace(base, present=False), progress)
    cal = {"pfa": pfa, "n_trials": n,
           "gamma_paf": threshold_for_pfa(tr.p_paf, pfa),
           "alpha_cfar": threshold_for_pfa(tr.p_cfar, pfa),
           "n_pad": pc.n_pad, "l_pad": pc.l_pad, "gamma_a": pc.gamma_a,
           "paf_h0_mean": float(tr.p_paf.mean()), "cfar_h0_mean": float(tr.p_cfar.mean())}
    return cal


def load_calibration() -> dict | None:
    try:
        with open(CAL_FILE) as f:
            return json.load(f)
    except Exception:
        return None


def save_calibration(cal: dict) -> None:
    os.makedirs(os.path.dirname(CAL_FILE), exist_ok=True)
    with open(CAL_FILE, "w") as f:
        json.dump(cal, f, indent=2)


def apply_calibration(pc: ProcConfig, cal: dict | None) -> ProcConfig:
    if not cal:
        return pc
    return replace(pc, gamma=cal["gamma_paf"], cfar_alpha=cal["alpha_cfar"])


# --------------------------------------------------------------- single-beam CRB validation (paper Fig. 3)
def single_beam_estimates(cfg: SystemConfig, snr_db: float, n_trials: int, n_pad: int, l_pad: int,
                          rng: np.random.Generator, n_sc: int | None = None, n_sym: int | None = None,
                          use_mask: bool = False):
    """One beam, one target, fixed (known) amplitude, random d in [0, d_u) and v in (-v_u, v_u):
    returns (d_err, v_err) of the zero-padded 2D-FFT peak estimator (no interpolation) -- the paper's Fig. 3 set-up."""
    N = n_sc or cfg.n_sc
    L = n_sym or cfg.n_sym
    n_idx = np.arange(1, N + 1)[:, None]
    l_idx = np.arange(1, L + 1)[None, :]
    amp = np.sqrt(db2lin(snr_db))
    d_ax = np.arange(n_pad) * C * cfg.ts / n_pad
    v_ax = (np.arange(l_pad) - l_pad // 2) * cfg.scs * cfg.wavelength / (2 * l_pad)
    mask = np.ones((N, L))
    if use_mask and N == 240 and L == 4:
        from .channel import ssb_mask
        mask = ssb_mask(N, L, True)
    de, ve = [], []
    for _ in range(n_trials):
        d = rng.uniform(0.05, 0.95) * cfg.unamb_range
        # the Doppler FFT grid spans +-lambda*scs/4 (= +-300 m/s): draw v inside it
        v = rng.uniform(-0.9, 0.9) * cfg.unamb_velocity / 2.0
        fd = 2 * v / cfg.wavelength
        Z = amp * np.exp(-2j * np.pi * n_idx * cfg.scs * d / C) * np.exp(2j * np.pi * fd * l_idx * cfg.ts)
        Z = Z + (rng.standard_normal((N, L)) + 1j * rng.standard_normal((N, L))) / np.sqrt(2)
        amp_map = range_velocity_map(Z * mask, n_pad, l_pad)
        i, k = np.unravel_index(int(np.argmax(amp_map)), amp_map.shape)
        de.append(d_ax[i] - d)
        ve.append(v_ax[k] - v)
    return np.array(de), np.array(ve)
