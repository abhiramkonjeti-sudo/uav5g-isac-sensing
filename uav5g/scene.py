"""Scene building blocks: a random manoeuvring flight path, random buildings, multi-frame simulation and scoring.

A *frame* is one SSB observation (about 67 us).  Frames are separated by `dt` seconds (for example every 25th SSB
burst, dt = 0.5 s).  Inside one frame the drone does not move noticeably (1 mm at 20 m/s), so every frame is the usual
snapshot model evaluated at that frame's position and velocity.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .config import SystemConfig, Scenario, ProcConfig, Building, db2lin
from .array import BeamSet
from .channel import (geometry, link_snr_best_db, simulate_echo, simulate_reference, building_offset_db, Echo)
from .processing import process, ProcResult


# ------------------------------------------------------------------------------------------------ flight path
@dataclass
class Path:
    pos: np.ndarray        # (K, 3) drone position in every frame [m]
    vel: np.ndarray        # (K, 3) drone velocity in every frame [m/s]
    dt: float              # time between frames [s]

    @property
    def k(self) -> int:
        return len(self.pos)


def straight_path(pos, vel, n_frames: int = 1, dt: float = 0.5) -> Path:
    t = np.arange(n_frames)[:, None] * dt
    p0, v = np.asarray(pos, float), np.asarray(vel, float)
    return Path(pos=p0 + t * v, vel=np.tile(v, (n_frames, 1)), dt=dt)


def random_path(rng: np.random.Generator, pos, speed: float, heading_deg: float, climb_deg: float,
                n_frames: int = 20, dt: float = 0.5, vmax: float = 20.0, yaw_sigma_deg: float = 25.0,
                acc_sigma: float = 2.0, climb_sigma_deg: float = 3.0, climb_limit_deg: float = 30.0,
                speed_relax_s: float = 5.0,
                bounds=((-100.0, 300.0), (100.0, 600.0), (10.0, 120.0))) -> Path:
    """Random manoeuvring flight: the heading performs a random walk (the drone turns), the speed fluctuates around
    its starting value (speeds up and slows down, sometimes almost hovering) and the climb angle wanders within
    +-climb_limit.  The drone is reflected back when it reaches the edge of `bounds` (x, y, z).  Scaling with
    sqrt(dt) keeps the statistics independent of the frame interval."""
    (x0, x1), (y0, y1), (z0, z1) = bounds
    p = np.array([np.clip(pos[0], x0, x1), np.clip(pos[1], y0, y1), np.clip(pos[2], z0, z1)], float)
    s = float(np.clip(speed, 0.0, vmax))
    s_mean = s
    h, c = np.radians(heading_deg), np.radians(climb_deg)
    sq = np.sqrt(dt)
    P, V = [], []
    for _ in range(n_frames):
        v = np.array([s * np.cos(c) * np.cos(h), s * np.cos(c) * np.sin(h), s * np.sin(c)])
        P.append(p.copy())
        V.append(v)
        h += rng.normal(0.0, np.radians(yaw_sigma_deg) * sq)
        s = float(np.clip(s + (s_mean - s) * dt / speed_relax_s + rng.normal(0.0, acc_sigma * sq), 0.0, vmax))
        c = float(np.clip(c + rng.normal(0.0, np.radians(climb_sigma_deg) * sq),
                          -np.radians(climb_limit_deg), np.radians(climb_limit_deg)))
        p = p + v * dt
        if p[0] < x0 or p[0] > x1:                       # reflect at the area edges
            p[0] = np.clip(2 * (x0 if p[0] < x0 else x1) - p[0], x0, x1)
            h = np.pi - h
        if p[1] < y0 or p[1] > y1:
            p[1] = np.clip(2 * (y0 if p[1] < y0 else y1) - p[1], y0, y1)
            h = -h
        if p[2] < z0 or p[2] > z1:
            p[2] = np.clip(2 * (z0 if p[2] < z0 else z1) - p[2], z0, z1)
            c = -c
    return Path(pos=np.array(P), vel=np.array(V), dt=dt)


# --------------------------------------------------------------------------------------------------- buildings
def random_buildings(rng: np.random.Generator, n: int, x_range, y_range, rcs_range=(15.0, 35.0),
                     z_range=(5.0, 40.0), min_sep: float = 30.0) -> list[Building]:
    """n random buildings: random position, reflectivity and (fixed) echo phase.  Different every call."""
    out: list[Building] = []
    for _ in range(2000):
        if len(out) >= n:
            break
        p = (float(rng.uniform(*x_range)), float(rng.uniform(*y_range)), float(rng.uniform(*z_range)))
        if any(np.hypot(p[0] - b.pos[0], p[1] - b.pos[1]) < min_sep for b in out):
            continue
        out.append(Building(pos=p, rcs_dbsm=float(rng.uniform(*rcs_range)),
                            alpha=complex(np.exp(2j * np.pi * rng.random())),
                            size=(float(rng.uniform(20, 60)), float(rng.uniform(20, 60)), 2.0 * p[2])))
    return out


def building_table(cfg: SystemConfig, scn: Scenario, buildings, pos0) -> list[dict]:
    """Rows for display: where each building is and how strong its echo is compared with the drone's."""
    s0 = replace(scn, pos=tuple(pos0))
    off = building_offset_db(cfg, s0)
    td = geometry(cfg, pos0, (0, 0, 0))
    snr_drone = scn.snr_db if scn.snr_mode == "manual" else link_snr_best_db(cfg, td.d_tx, td.d_rx)
    rows = []
    for i, b in enumerate(buildings):
        tb = geometry(cfg, b.pos, (0, 0, 0))
        snr_b = link_snr_best_db(replace(cfg, rcs_dbsm=b.rcs_dbsm), tb.d_tx, tb.d_rx) + off
        rows.append({"#": i + 1, "x [m]": round(b.pos[0], 1), "y [m]": round(b.pos[1], 1), "z [m]": round(b.pos[2], 1),
                     "RCS [dBsm]": round(b.rcs_dbsm, 1), "bistatic range [m]": round(tb.d, 1),
                     "echo vs drone [dB]": round(snr_b - snr_drone, 1)})
    return rows


# ----------------------------------------------------------------------------------------- multi-frame simulation
def simulate_sequence(cfg: SystemConfig, scn: Scenario, path: Path, buildings, beams: BeamSet,
                      rng: np.random.Generator, ref_frames: int = 8):
    """One Echo per frame (new noise and fading every frame, identical buildings) plus the reference scan."""
    s0 = replace(scn, pos=tuple(path.pos[0]), vel=tuple(path.vel[0]))
    off = building_offset_db(cfg, s0) if len(buildings) else None
    echoes: list[Echo] = []
    for k in range(path.k):
        sk = replace(scn, pos=tuple(path.pos[k]), vel=tuple(path.vel[k]))
        echoes.append(simulate_echo(cfg, sk, beams, rng, buildings=buildings, snr_offset_db=off))
    z_ref = None
    if len(buildings) and ref_frames > 0:
        z_ref = simulate_reference(cfg, s0, beams, rng, buildings, m=ref_frames, snr_offset_db=off)
    return echoes, z_ref


def process_sequence(echoes, cfg: SystemConfig, beams: BeamSet, pc: ProcConfig, z_ref=None) -> list[ProcResult]:
    return [process(e.Z, cfg, beams, e.active, pc, z_ref) for e in echoes]


# ----------------------------------------------------------------------------------------------------- scoring
def classify_frame(d_hat: float, detected: bool, truth, building_ranges, tol: float = 25.0) -> str:
    """'drone' = detected and the range estimate is within tol of the drone's range;  'building' = detected but the
    estimate sits on a building;  'other' = detected somewhere else;  'missed' = drone present but not detected;
    'none' = nothing present and nothing detected."""
    if not detected:
        return "missed" if truth.present else "none"
    if truth.present and abs(d_hat - truth.d) <= tol:
        return "drone"
    if any(abs(d_hat - r) <= tol for r in building_ranges):
        return "building"
    return "other"


def building_ranges(cfg: SystemConfig, buildings) -> list[float]:
    return [geometry(cfg, b.pos, (0, 0, 0)).d for b in buildings]


def score_sequence(results, echoes, cfg: SystemConfig, buildings, pc: ProcConfig, detector: str = "paf",
                   tol: float = 25.0) -> dict:
    br = building_ranges(cfg, buildings)
    labels, errs = [], []
    for r, e in zip(results, echoes):
        det = r.detected_paf if detector == "paf" else r.detected_cfar
        lab = classify_frame(r.d_hat, det, e.truth, br, tol)
        labels.append(lab)
        if lab == "drone" and np.all(np.isfinite(r.pos_hat)):
            errs.append(float(np.linalg.norm(r.pos_hat - e.truth.pos)))
    n = len(labels)
    cnt = {k: labels.count(k) for k in ("drone", "building", "other", "missed", "none")}
    return dict(labels=labels, n=n, counts=cnt,
                detected=(n - cnt["missed"] - cnt["none"]) / max(n, 1),
                correct=cnt["drone"] / max(n, 1),
                on_building=cnt["building"] / max(n, 1),
                pos_med=float(np.median(errs)) if errs else float("nan"))
