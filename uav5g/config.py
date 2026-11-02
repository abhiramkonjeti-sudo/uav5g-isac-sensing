"""Configuration objects and constants for the 5G NR SSB passive-bistatic UAV sensing simulator.

Reference: P. Jopanya and D. P. M. Osorio, "Utilizing 5G NR SSB Blocks for Passive Detection and
Localization of Low-Altitude Drones", IEEE SPAWC 2025.

Geometry convention used throughout the project
-----------------------------------------------
* Both base stations (BsA = sensing transmitter, BsB = sensing receiver) have their planar array
  facing the +y direction.  x is the horizontal axis along the baseline, z is height.
* A direction is described by a unit vector u = (ux, uy, uz) with uy > 0.
  elevation phi = arcsin(uz),  azimuth theta = arcsin(ux / cos(phi)).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

C = 299_792_458.0  # speed of light [m/s]


def db2lin(x):
    return 10.0 ** (np.asarray(x, dtype=float) / 10.0)


def lin2db(x):
    return 10.0 * np.log10(np.maximum(np.asarray(x, dtype=float), 1e-300))


@dataclass
class SystemConfig:
    """Physical-layer / array / geometry parameters (defaults = Table I of the paper)."""

    mv: int = 10                 # antennas per column  (M_V)
    mh: int = 10                 # antennas per row     (M_H)
    scs: float = 60e3            # subcarrier spacing   [Hz]
    fc: float = 15e9             # carrier frequency    [Hz]
    n_sc: int = 240              # SSB subcarriers      (N)
    n_sym: int = 4               # SSB OFDM symbols     (L)
    rcs_dbsm: float = -10.0      # drone radar cross-section [dBsm]
    bs_a: tuple = (0.0, 0.0, 0.0)      # sensing transmitter position [m]
    bs_b: tuple = (200.0, 0.0, 0.0)    # sensing receiver position    [m]
    # --- link budget (only used when snr_mode == "link") ---
    tx_power_dbm: float = 46.0
    carrier_subcarriers: int = 1656    # 100 MHz carrier @ 60 kHz SCS (138 RB x 12)
    noise_figure_db: float = 5.0
    rx_gain_db: float = 0.0            # BsB receive array gain (paper: no receive beamforming -> 0 dB)
    # --- waveform detail ---
    use_ssb_mask: bool = True    # honour the real SSB layout (PSS/SSS/PBCH occupy only part of the grid)

    @property
    def m(self) -> int:
        return self.mv * self.mh

    @property
    def wavelength(self) -> float:
        return C / self.fc

    @property
    def ts(self) -> float:
        """OFDM useful symbol duration Ts = 1/scs."""
        return 1.0 / self.scs

    @property
    def bandwidth(self) -> float:
        return self.n_sc * self.scs

    @property
    def unamb_range(self) -> float:
        """d_u = c * Ts   [m]  (unambiguous bistatic range)."""
        return C * self.ts

    @property
    def unamb_velocity(self) -> float:
        """v_u = lambda * scs / 2   [m/s]."""
        return self.wavelength * self.scs / 2.0

    @property
    def baseline(self) -> float:
        return float(np.linalg.norm(np.asarray(self.bs_b) - np.asarray(self.bs_a)))


@dataclass
class Scenario:
    """One simulated situation (what Interface 1 builds)."""

    pos: tuple = (100.0, 250.0, 60.0)     # drone position [m]
    vel: tuple = (10.0, -5.0, 0.0)        # drone velocity vector [m/s]
    present: bool = True                  # False -> noise-only (H0) trial
    snr_mode: str = "manual"              # "manual": snr_db given; "link": computed from link budget
    snr_db: float = -5.0                  # per-sample SNR at BsB for a perfectly aligned beam (|g| = M)
    swerling: bool = True                 # Swerling-1 RCS fluctuation, alpha ~ CN(0,1)
    deact_pct: float = 0.0                # % of surveillance beams switched off (power saving)
    # --- non-idealities (extensions beyond the paper) ---
    direct_residual_db: float = -np.inf   # residual (uncancelled) direct-path power vs noise [dB]; -inf = perfect
    clutter_cnr_db: float = -np.inf       # range-localised interference band, CNR at its centre [dB]; -inf = none
    clutter_center_m: float = 500.0
    clutter_width_m: float = 150.0
    # receiver stability between frames: the whole static scene (buildings) is scaled by (1 + eps) in every frame,
    # eps ~ CN(0, 10^(-stability/10)).  inf = perfectly stable, so a stored reference scan cancels the buildings exactly
    stability_db: float = np.inf


@dataclass
class Building:
    """A static point-like scatterer (building corner / roof).  Its complex echo is identical in every frame
    (zero Doppler, fixed amplitude and phase), which is what makes background subtraction possible."""

    pos: tuple                          # (x, y, z) of the dominant scattering point [m]
    rcs_dbsm: float = 25.0
    alpha: complex = 1.0 + 0.0j         # fixed complex reflection coefficient, |alpha| = 1
    size: tuple = (40.0, 30.0, 30.0)    # footprint x, y and height [m] - used only for drawing


@dataclass
class ProcConfig:
    """Receiver-side processing parameters (what Interface 2 uses)."""

    n_pad: int = 512             # N'  zero-padded IFFT length (range)
    l_pad: int = 32              # L'  zero-padded FFT length  (Doppler)
    gamma_a: float = 6.0         # beam-aggregation threshold on PAF
    gamma: float = 4.0           # PAF detection threshold (overridden by calibration)
    cfar_alpha: float = 20.0     # CFAR threshold factor      (overridden by calibration)
    cfar_guard: int = 4          # CFAR guard cells (each side, range)
    cfar_train: int = 16         # CFAR training cells (each side, range)
    refine: bool = True          # parabolic peak interpolation
    angle_method: str = "strongest"   # "strongest" (paper/slides) | "weighted" (our extension)
    gate_direct: bool = False    # exclude range bins around the direct path (needed with residual direct path)
    gate_guard_m: float = 60.0
    window: str | None = None    # taper across subcarriers: None | "hann" | "blackmanharris" (sidelobe suppression)
    d_min: float = 0.0           # search window for the bistatic range [m]
    d_max: float | None = None   # None -> unambiguous range
