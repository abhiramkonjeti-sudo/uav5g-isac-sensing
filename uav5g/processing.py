"""Receiver processing chain: 2D-FFT range-velocity map -> PAF / CFAR detection -> beam aggregation ->
range / velocity / angle estimation -> localisation (paper Sec. III + our CFAR / localisation extensions)."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import convolve1d
from scipy.signal import windows as _win

from .config import C, SystemConfig, ProcConfig
from .array import BeamSet, direction_to_angles


# ------------------------------------------------------------------------------------------------ axes
def range_axis(cfg: SystemConfig, n_pad: int) -> np.ndarray:
    """Bistatic range of IFFT bin n':  d = n' c Ts / N'."""
    return np.arange(n_pad) * C * cfg.ts / n_pad


def velocity_axis(cfg: SystemConfig, l_pad: int) -> np.ndarray:
    """Bistatic radial velocity of (fft-shifted) Doppler bin:  v = l' scs lambda / (2 L')."""
    return (np.arange(l_pad) - l_pad // 2) * cfg.scs * cfg.wavelength / (2.0 * l_pad)


# --------------------------------------------------------------------------------------- 2D FFT (Eq. 8)
def range_velocity_map(Z: np.ndarray, n_pad: int, l_pad: int) -> np.ndarray:
    """|IFFT_range( FFT_Doppler(Z) )| with zero padding.  Z: (..., N, L) -> (..., N', L') amplitude,
    Doppler axis fft-shifted so that zero velocity is in the middle."""
    X = np.fft.ifft(Z, n=n_pad, axis=-2) * n_pad          # sum_n Z[n,l] e^{+j2pi n n'/N'}
    X = np.fft.fft(X, n=l_pad, axis=-1)                    # sum_l ...    e^{-j2pi l l'/L'}
    X = np.fft.fftshift(X, axes=-1)
    return np.abs(X)


def _parabolic(y_m: float, y_0: float, y_p: float) -> float:
    den = y_m - 2.0 * y_0 + y_p
    if abs(den) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (y_m - y_p) / den, -0.5, 0.5))


def taper_matrix(n_sc: int, n_sym: int, name: str | None, mask: np.ndarray | None = None) -> np.ndarray:
    """(N, L) amplitude taper applied across the occupied subcarriers of every OFDM symbol (sidelobe control)."""
    w = np.ones((n_sc, n_sym))
    if not name:
        return w
    for l in range(n_sym):
        occ = np.where(mask[:, l] > 0)[0] if mask is not None else np.arange(n_sc)
        if occ.size < 3:
            continue
        span = occ[-1] - occ[0] + 1
        win = getattr(_win, name)(span, sym=True)
        w[:, l] = 0.0
        w[occ[0]: occ[-1] + 1, l] = win
    return w if mask is None else w * mask


# ------------------------------------------------------------------------------------------------ CFAR
def cfar_statistic(P: np.ndarray, guard: int, train: int) -> np.ndarray:
    """Range-adaptive cell-averaging CFAR statistic  T = P / sigma_hat^2.

    P: (R, N', L') power maps.  The local noise level of range bin n' is the mean power of the training cells
    n'-g-T..n'-g-1 and n'+g+1..n'+g+T (all Doppler bins, circular in range), so a slowly varying noise floor
    (clutter residue, interference) is tracked while a target occupying a few cells is excluded by the guard."""
    row = P.mean(axis=2)                                         # (R, N')
    k = np.zeros(2 * (guard + train) + 1)
    k[: train] = 1.0
    k[-train:] = 1.0
    k /= k.sum()
    sigma2 = convolve1d(row, k, axis=1, mode="wrap")             # (R, N')
    return P / np.maximum(sigma2, 1e-30)[:, :, None]


# ------------------------------------------------------------------------------------------- main chain
@dataclass
class ProcResult:
    paf: np.ndarray                  # (R,) per-beam peak-to-average factor (-inf for inactive beams)
    p_paf: float                     # detection statistic  max_r PAF_r
    cfar_beam: np.ndarray            # (R,) per-beam max CFAR statistic
    p_cfar: float                    # detection statistic  max_r max_cells T
    eta: np.ndarray                  # indices of the beams aggregated
    detected_paf: bool
    detected_cfar: bool
    d_hat: float
    v_hat: float
    beam_idx: int
    u_hat: np.ndarray
    theta_hat: float
    phi_hat: float
    d_tx_hat: float
    pos_hat: np.ndarray
    agg_map: np.ndarray              # (N', L') aggregated power map
    d_axis: np.ndarray
    v_axis: np.ndarray
    peak: tuple


def localise(d: float, u: np.ndarray, bs_a, bs_b) -> tuple[float, np.ndarray]:
    """Ray (beam direction u from BsA)  intersected with the bistatic-range ellipse  d_tx + d_rx = d:

        d_tx = (d^2 - L_b^2) / (2 (d - u.b)),    p = p_A + d_tx u
    """
    pa, pb = np.asarray(bs_a, float), np.asarray(bs_b, float)
    b = pb - pa
    lb2 = float(b @ b)
    den = 2.0 * (d - float(u @ b))
    if den <= 1e-9 or d * d <= lb2:
        return float("nan"), np.full(3, np.nan)
    d_tx = (d * d - lb2) / den
    return float(d_tx), pa + d_tx * u


def process(Z: np.ndarray, cfg: SystemConfig, beams: BeamSet, active: np.ndarray, pc: ProcConfig,
            z_ref: np.ndarray | None = None) -> ProcResult:
    """z_ref (optional): stored reference scan of the empty scene.  Subtracting it removes static clutter such as
    buildings; the beams that are switched off stay at zero."""
    R = beams.r
    d_axis = range_axis(cfg, pc.n_pad)
    v_axis = velocity_axis(cfg, pc.l_pad)
    if z_ref is not None:
        Z = (Z - z_ref) * active[:, None, None]

    if pc.window:
        from .channel import ssb_mask
        Z = Z * taper_matrix(cfg.n_sc, cfg.n_sym, pc.window, ssb_mask(cfg.n_sc, cfg.n_sym, cfg.use_ssb_mask))[None]
    amp = range_velocity_map(Z, pc.n_pad, pc.l_pad)              # (R, N', L')
    P = amp ** 2

    # --- search window in bistatic range (+ optional direct-path blanking)
    d_max = cfg.unamb_range if pc.d_max is None else pc.d_max
    valid = (d_axis >= pc.d_min) & (d_axis <= d_max)
    if pc.gate_direct:
        valid &= np.abs(d_axis - cfg.baseline) > pc.gate_guard_m
    if not valid.any():
        valid[:] = True

    # --- PAF per beam (peak / mean of the amplitude map)
    A = amp[:, valid, :]
    paf = A.max(axis=(1, 2)) / np.maximum(A.mean(axis=(1, 2)), 1e-30)
    paf = np.where(active, paf, -np.inf)
    p_paf = float(paf.max()) if active.any() else 0.0

    # --- CFAR per beam
    T = cfar_statistic(P, pc.cfar_guard, pc.cfar_train)
    cfar_beam = T[:, valid, :].max(axis=(1, 2))
    cfar_beam = np.where(active, cfar_beam, -np.inf)
    p_cfar = float(cfar_beam.max()) if active.any() else 0.0

    # --- beam aggregation (non-coherent power integration of the beams with PAF > gamma_a)
    eta = np.where(paf > pc.gamma_a)[0]
    if eta.size == 0:
        eta = np.array([int(np.argmax(paf))])
    agg = P[eta].mean(axis=0)
    masked = np.where(valid[:, None], agg, -np.inf)
    i0, k0 = np.unravel_index(int(np.argmax(masked)), agg.shape)

    di = dk = 0.0
    if pc.refine:
        lg = lambda x: np.log(max(x, 1e-30))
        N_, L_ = agg.shape
        di = _parabolic(lg(agg[(i0 - 1) % N_, k0]), lg(agg[i0, k0]), lg(agg[(i0 + 1) % N_, k0]))
        dk = _parabolic(lg(agg[i0, (k0 - 1) % L_]), lg(agg[i0, k0]), lg(agg[i0, (k0 + 1) % L_]))
    d_hat = (i0 + di) * C * cfg.ts / pc.n_pad
    v_step = cfg.scs * cfg.wavelength / (2.0 * pc.l_pad)
    v_hat = (k0 + dk - pc.l_pad // 2) * v_step

    # --- angle from the strongest beam at the detected cell (or amplitude-weighted neighbours)
    peak_amp = np.where(active, amp[:, i0, k0], 0.0)
    b_idx = int(np.argmax(peak_amp))
    if pc.angle_method == "weighted":
        top = np.argsort(peak_amp)[-4:]
        w = peak_amp[top]
        u = (w[:, None] * beams.dirs[top]).sum(0)
        u_hat = u / np.linalg.norm(u)
    else:
        u_hat = beams.dirs[b_idx]
    th_hat, ph_hat = direction_to_angles(u_hat)

    d_tx_hat, pos_hat = localise(d_hat, u_hat, cfg.bs_a, cfg.bs_b)

    return ProcResult(paf=paf, p_paf=p_paf, cfar_beam=cfar_beam, p_cfar=p_cfar, eta=eta,
                      detected_paf=p_paf > pc.gamma, detected_cfar=p_cfar > pc.cfar_alpha,
                      d_hat=float(d_hat), v_hat=float(v_hat), beam_idx=b_idx, u_hat=u_hat,
                      theta_hat=th_hat, phi_hat=ph_hat, d_tx_hat=d_tx_hat, pos_hat=pos_hat,
                      agg_map=agg, d_axis=d_axis, v_axis=v_axis, peak=(int(i0), int(k0)))
