"""Cramer-Rao bounds for bistatic range and velocity (paper Eqs. 11-17, single target, known amplitude)."""
from __future__ import annotations

import numpy as np

from .config import C, SystemConfig, db2lin


def crb_range_var(snr_db, n: int, l: int, cfg: SystemConfig) -> np.ndarray:
    """var(d_hat) >= 3 Ts^2 c^2 (2L+1) / (pi^2 SNR N L (7NL - N - L - 5)(N+1))     [m^2]"""
    snr = db2lin(snr_db)
    return 3 * cfg.ts ** 2 * C ** 2 * (2 * l + 1) / (np.pi ** 2 * snr * n * l * (7 * n * l - n - l - 5) * (n + 1))


def crb_vel_var(snr_db, n: int, l: int, cfg: SystemConfig) -> np.ndarray:
    """var(v_hat) >= 3 lambda^2 df^2 (2N+1) / (4 pi^2 SNR N L (7NL - N - L - 5)(L+1))   [(m/s)^2]"""
    snr = db2lin(snr_db)
    lam, df = cfg.wavelength, cfg.scs
    return 3 * lam ** 2 * df ** 2 * (2 * n + 1) / (4 * np.pi ** 2 * snr * n * l * (7 * n * l - n - l - 5) * (l + 1))


def fim(snr_db: float, n: int, l: int, cfg: SystemConfig) -> np.ndarray:
    """2x2 Fisher information matrix for Theta = [d, v]  (Eq. 16)."""
    snr = float(db2lin(snr_db))
    lam, df, ts = cfg.wavelength, cfg.scs, cfg.ts
    k = 4 * np.pi ** 2 * snr * n * l
    return k * np.array([
        [(df ** 2 / C ** 2) * (n + 1) * (2 * n + 1) / 3.0, -(l + 1) * (n + 1) / (C * lam)],
        [-(l + 1) * (n + 1) / (C * lam), (4.0 / 3.0) * (ts ** 2 / lam ** 2) * (l + 1) * (2 * l + 1)],
    ])
