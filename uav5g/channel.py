"""Bistatic geometry, link budget and SSB echo generation (paper Eqs. 3-6).

After direct-link cancellation the received block of surveillance beam r is

    Z_r[n, l] = sqrt(SNR_r) * alpha * exp(-j 2 pi n df d / c) * exp(j 2 pi f_d l Ts)  +  w[n, l]

with n = 1..N (subcarrier), l = 1..L (OFDM symbol), d = d_tx + d_rx the bistatic range, f_d = 2 v / lambda,
alpha ~ CN(0,1) (Swerling-1, constant over the block) and w ~ CN(0, 1) (AWGN, unit variance).
SNR_r = SNR_best * (|g_r| / M)^2 : beams that point away from the drone see a weaker echo.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

from .config import C, SystemConfig, Scenario, Building, db2lin, lin2db
from .array import BeamSet, beam_gains


# --------------------------------------------------------------------------------------------- geometry
@dataclass
class Truth:
    """Ground truth of one scenario (stored by Interface 1, compared against estimates in Interface 2)."""

    present: bool
    pos: np.ndarray | None = None
    vel: np.ndarray | None = None
    d_tx: float = np.nan
    d_rx: float = np.nan
    d: float = np.nan            # bistatic range  d_tx + d_rx
    tau: float = np.nan
    v_bi: float = np.nan         # bistatic radial velocity (positive = approaching), f_d = 2 v_bi / lambda
    f_d: float = np.nan
    u_a: np.ndarray | None = None   # unit vector BsA -> drone
    snr_best_db: float = np.nan
    speed: float = np.nan


def geometry(cfg: SystemConfig, pos, vel) -> Truth:
    p = np.asarray(pos, float)
    v = np.asarray(vel, float)
    pa, pb = np.asarray(cfg.bs_a, float), np.asarray(cfg.bs_b, float)
    ra, rb = p - pa, p - pb
    d_tx, d_rx = np.linalg.norm(ra), np.linalg.norm(rb)
    ua, ub = ra / d_tx, rb / d_rx
    d = d_tx + d_rx
    # d(d_tx + d_rx)/dt = v.(ua + ub);  closing speed => positive Doppler
    v_bi = -0.5 * float(v @ (ua + ub))
    return Truth(present=True, pos=p, vel=v, d_tx=float(d_tx), d_rx=float(d_rx), d=float(d), tau=float(d / C),
                 v_bi=v_bi, f_d=2.0 * v_bi / cfg.wavelength, u_a=ua, speed=float(np.linalg.norm(v)))


def link_snr_best_db(cfg: SystemConfig, d_tx: float, d_rx: float) -> float:
    """SNR per sample for a perfectly aligned transmit beam (|g| = M):  rho * beta * M / sigma^2  (Eq. 5)."""
    lam = cfg.wavelength
    beta = lam ** 2 * db2lin(cfg.rcs_dbsm) / ((4 * np.pi) ** 3 * d_tx ** 2 * d_rx ** 2)
    rho_dbm = cfg.tx_power_dbm - 10 * np.log10(cfg.carrier_subcarriers)       # power per subcarrier
    noise_dbm = -174.0 + 10 * np.log10(cfg.scs) + cfg.noise_figure_db          # per-subcarrier noise
    return float(rho_dbm + lin2db(beta) + 10 * np.log10(cfg.m) + cfg.rx_gain_db - noise_dbm)


# ---------------------------------------------------------------------------------------------- SSB layout
def ssb_mask(n_sc: int, n_sym: int, enabled: bool = True) -> np.ndarray:
    """(N, L) 0/1 mask of the resource elements an NR SSB really occupies (TS 38.211 sec. 7.4.3.1).

    sym0: PSS (127 centre sc)   sym1: PBCH (all 240)   sym2: PBCH (48 + 48 edge) + SSS (127 centre)
    sym3: PBCH (all 240).  Falls back to a full grid for non-standard sizes."""
    m = np.ones((n_sc, n_sym))
    if enabled and n_sc == 240 and n_sym == 4:
        m[:] = 0.0
        m[56:183, 0] = 1.0
        m[:, 1] = 1.0
        m[0:48, 2] = 1.0
        m[192:240, 2] = 1.0
        m[56:183, 2] = 1.0
        m[:, 3] = 1.0
    return m


# --------------------------------------------------------------------------------------------- simulation
def _cn(rng: np.random.Generator, shape) -> np.ndarray:
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


@dataclass
class Echo:
    Z: np.ndarray            # (R, N, L) complex received SSB grids (zero for deactivated beams)
    active: np.ndarray       # (R,) bool
    truth: Truth
    gains: np.ndarray        # (R,) |g_r|/M  (0 if no target)
    alpha: complex


def building_offset_db(cfg: SystemConfig, scn: Scenario) -> float:
    """dB added to every building's physical link SNR.  In manual-SNR mode the drone keeps the SNR the user typed, and
    the buildings are shifted by the same amount, so the drone-to-building power ratio stays physical (RCS, distances,
    beam gain).  0 in link-budget mode, where every echo already comes from the same link budget."""
    if scn.snr_mode != "manual":
        return 0.0
    t = geometry(cfg, scn.pos, scn.vel)
    return float(scn.snr_db - link_snr_best_db(cfg, t.d_tx, t.d_rx))


def simulate_echo(cfg: SystemConfig, scn: Scenario, beams: BeamSet, rng: np.random.Generator,
                  buildings=(), snr_offset_db: float | None = None, noise_scale: float = 1.0) -> Echo:
    """buildings: static scatterers added to the received block.  snr_offset_db: see building_offset_db (None = compute
    it from scn.pos; a frame sequence passes one fixed value so the buildings stay identical).  noise_scale scales the
    noise standard deviation (used for the averaged reference scan)."""
    R, N, L = beams.r, cfg.n_sc, cfg.n_sym
    # which beams are switched on (uniform random deactivation)
    active = np.ones(R, dtype=bool)
    n_off = int(round(scn.deact_pct / 100.0 * R))
    if n_off > 0:
        active[rng.choice(R, size=min(n_off, R), replace=False)] = False

    Z = _cn(rng, (R, N, L)) * noise_scale         # AWGN, sigma_n^2 = 1 (before noise_scale)
    n_idx = np.arange(1, N + 1)[:, None]
    l_idx = np.arange(1, L + 1)[None, :]

    truth = Truth(present=False)
    gains = np.zeros(R)
    alpha = 0j
    if scn.present:
        truth = geometry(cfg, scn.pos, scn.vel)
        snr_db = scn.snr_db if scn.snr_mode == "manual" else link_snr_best_db(cfg, truth.d_tx, truth.d_rx)
        truth.snr_best_db = snr_db
        gains = beam_gains(cfg, beams, truth.u_a) / cfg.m
        alpha = _cn(rng, ()) if scn.swerling else 1.0 + 0j
        amp = np.sqrt(db2lin(snr_db)) * alpha * gains
        sig = np.exp(-2j * np.pi * n_idx * cfg.scs * truth.d / C) * np.exp(2j * np.pi * truth.f_d * l_idx * cfg.ts)
        Z += amp[:, None, None] * sig[None]

    if np.isfinite(scn.direct_residual_db):       # uncancelled direct path: zero Doppler, range = baseline
        a_dir = np.sqrt(db2lin(scn.direct_residual_db))
        ph = np.exp(2j * np.pi * rng.random(R))
        Z += (a_dir * ph)[:, None, None] * np.exp(-2j * np.pi * n_idx * cfg.scs * cfg.baseline / C)[None]

    if np.isfinite(scn.clutter_cnr_db):           # range-localised interference band (non-stationary noise floor)
        bin_m = C / (N * cfg.scs)
        r_axis = np.arange(N) * bin_m
        env = np.exp(-0.5 * ((r_axis - scn.clutter_center_m) / scn.clutter_width_m) ** 2)
        w_delay = _cn(rng, (R, N, L)) * (np.sqrt(db2lin(scn.clutter_cnr_db)) * env)[None, :, None]
        Z += np.fft.fft(w_delay, axis=1, norm="ortho")

    if len(buildings):                            # static scene: zero Doppler, same complex echo in every frame
        off = building_offset_db(cfg, scn) if snr_offset_db is None else snr_offset_db
        stat = np.zeros((R, N, 1), dtype=complex)
        for b in buildings:
            tb = geometry(cfg, b.pos, (0.0, 0.0, 0.0))
            snr_b = link_snr_best_db(replace(cfg, rcs_dbsm=b.rcs_dbsm), tb.d_tx, tb.d_rx) + off
            g_b = beam_gains(cfg, beams, tb.u_a) / cfg.m
            amp_b = np.sqrt(db2lin(snr_b)) * b.alpha * g_b
            stat = stat + amp_b[:, None, None] * np.exp(-2j * np.pi * n_idx * cfg.scs * tb.d / C)[None]
        if np.isfinite(scn.stability_db):         # receiver drift between frames -> imperfect cancellation later
            stat = stat * (1.0 + _cn(rng, ()) * np.sqrt(db2lin(-scn.stability_db)))
        Z += stat

    Z *= ssb_mask(N, L, cfg.use_ssb_mask)[None]   # receiver nulls resource elements that carry no SSB
    Z[~active] = 0.0
    return Echo(Z=Z, active=active, truth=truth, gains=gains, alpha=complex(alpha))


def simulate_reference(cfg: SystemConfig, scn: Scenario, beams: BeamSet, rng: np.random.Generator, buildings,
                       m: int = 8, snr_offset_db: float | None = None) -> np.ndarray:
    """Reference scan of the empty scene: the average of m drone-free frames (all beams on).  The static buildings are
    kept, the noise variance drops to 1/m.  Subtracting it from a later frame cancels the buildings."""
    off = building_offset_db(cfg, scn) if snr_offset_db is None else snr_offset_db
    s0 = replace(scn, present=False, deact_pct=0.0, stability_db=np.inf, direct_residual_db=-np.inf,
                 clutter_cnr_db=-np.inf)
    return simulate_echo(cfg, s0, beams, rng, buildings=buildings, snr_offset_db=off,
                         noise_scale=1.0 / np.sqrt(max(m, 1))).Z
