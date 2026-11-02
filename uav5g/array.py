"""Uniform planar array (UPA), SSB beam codebook and beamforming gain (paper Eqs. 1-2)."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import SystemConfig


def steering_vector(theta: float, phi: float, mv: int, mh: int) -> np.ndarray:
    """Half-wavelength UPA response a(theta, phi) = a_MV(phi, 0) (x) a_MH(theta, phi)."""
    a_v = np.exp(-1j * np.pi * np.arange(mv) * np.sin(phi))
    a_h = np.exp(-1j * np.pi * np.arange(mh) * np.sin(theta) * np.cos(phi))
    return np.kron(a_v, a_h)


def direction_to_angles(u: np.ndarray) -> tuple[float, float]:
    """Unit vector -> (azimuth theta, elevation phi) [rad] under the project's convention."""
    phi = np.arcsin(np.clip(u[2], -1.0, 1.0))
    cphi = np.cos(phi)
    theta = np.arcsin(np.clip(u[0] / max(cphi, 1e-12), -1.0, 1.0))
    return float(theta), float(phi)


def angles_to_direction(theta: float, phi: float) -> np.ndarray:
    ux = np.sin(theta) * np.cos(phi)
    uz = np.sin(phi)
    uy = np.sqrt(max(1.0 - ux * ux - uz * uz, 0.0))
    return np.array([ux, uy, uz])


@dataclass
class BeamSet:
    theta: np.ndarray      # (R,) azimuth  [rad]
    phi: np.ndarray        # (R,) elevation [rad]
    weights: np.ndarray    # (R, M) conj-steering vectors  a*(theta_r, phi_r)  (precoder columns * sqrt(M))
    dirs: np.ndarray       # (R, 3) pointing unit vectors
    steer: np.ndarray      # (R, M) steering vectors a(theta_r, phi_r)

    @property
    def r(self) -> int:
        return len(self.theta)


def build_beams(cfg: SystemConfig, surveillance_only: bool = True) -> BeamSet:
    """Beam grid of Eq. (1).  M=100 -> angles {0, +-11.5, +-23.6, +-36.9, +-53.1} deg, 81 beams;
    the R=45 beams with elevation >= 0 form the surveillance set."""
    m = cfg.m
    qmax = int(np.floor(np.sqrt(m) / 2 - 1))
    q = np.arange(-qmax, qmax + 1)
    ang = np.arcsin(2 * q / np.sqrt(m))
    th, ph = [], []
    for t in ang:
        for p in ang:
            if surveillance_only and p < -1e-12:
                continue
            th.append(t)
            ph.append(p)
    th, ph = np.array(th), np.array(ph)
    steer = np.stack([steering_vector(t, p, cfg.mv, cfg.mh) for t, p in zip(th, ph)])
    dirs = np.stack([angles_to_direction(t, p) for t, p in zip(th, ph)])
    return BeamSet(theta=th, phi=ph, weights=np.conj(steer), dirs=dirs, steer=steer)


def beam_gains(cfg: SystemConfig, beams: BeamSet, u_target: np.ndarray) -> np.ndarray:
    """|g_r| = |a^T(theta_d, phi_d) a^*(theta_r, phi_r)| for every beam; 0 behind the array (uy <= 0)."""
    if u_target[1] <= 0:
        return np.zeros(beams.r)
    th, ph = direction_to_angles(u_target)
    a_t = steering_vector(th, ph, cfg.mv, cfg.mh)
    return np.abs(beams.weights @ a_t)
