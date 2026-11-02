"""5G NR SSB passive-bistatic UAV sensing - interactive simulator.

Run:   streamlit run app.py

Interface 1  Scenario Builder      -> places the drone / base stations, sets SNR, fading, beam deactivation,
                                       and generates the SSB echo received at BsB (ground truth is stored).
Interface 2  Detection Dashboard   -> range-velocity map, PAF / CFAR detection, range / velocity / angle / location
                                       estimates compared with the truth, and a Monte-Carlo P_D / P_FA / RMSE runner.
Validation   pre-computed figures from experiments/run_validation.py
"""
from __future__ import annotations

import json
import os
from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from uav5g.config import C, SystemConfig, Scenario, ProcConfig, db2lin
from uav5g.array import build_beams, angles_to_direction
from uav5g.channel import simulate_echo, link_snr_best_db, geometry
from uav5g.processing import process
from uav5g.scene import (straight_path, random_path, random_buildings, building_table, simulate_sequence,
                         process_sequence, score_sequence, building_ranges)
from uav5g.montecarlo import (Region, random_scenario, run_trials, roc_curve, rmse, calibrate, save_calibration,
                              load_calibration, apply_calibration, auc)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
BLUE, ORANGE, GREEN, RED, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#6b7280"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.titleweight": "bold", "legend.frameon": False})

st.set_page_config(page_title="5G ISAC UAV Sensing", layout="wide")
st.title("UAV sensing with 5G NR SSB signals (passive bistatic ISAC)")
st.caption("BsA transmits SSB beams -> the drone reflects a weak echo -> BsB receives -> Python estimates "
           "detection, range, velocity, angle and position.  Simulator based on Jopanya & Osorio, IEEE SPAWC 2025.")

# ------------------------------------------------------------------------------------------ session defaults
DEFAULTS = dict(ax=0.0, ay=0.0, az=0.0, bx=200.0, by=0.0, bz=0.0, dx=100.0, dy=250.0, dz=60.0, speed=12.0,
                heading=-25.0, climb=0.0, snr_mode="Manual SNR", snr_db=-8.0, tx_dbm=46.0, rcs=-10.0,
                swerling=True, deact=0, seed=1, resid_on=False, resid_db=20.0, clut_on=False, clut_db=12.0,
                clut_c=600.0, clut_w=250.0, rx_gain=0.0, window="none", guard=60.0,
                motion="Straight line", n_frames=20, dt=0.5, vmax_path=20.0, yaw=25.0,
                bld_on=False, n_bld=5, rcs_lo=15.0, rcs_hi=35.0, stab_db=60.0, ref_frames=8,
                reseed=True, suppress=True, mc_bld=False)
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def randomize():
    rng = np.random.default_rng()
    bx = st.session_state["bx"] - st.session_state["ax"]
    st.session_state["dx"] = float(round(st.session_state["ax"] + rng.uniform(0.0, max(bx, 1.0)), 1))
    st.session_state["dy"] = float(round(rng.uniform(150, 500), 1))
    st.session_state["dz"] = float(round(rng.uniform(20, 100), 1))
    st.session_state["speed"] = float(round(rng.uniform(0, 20), 1))
    st.session_state["heading"] = float(round(rng.uniform(-180, 180), 1))
    st.session_state["climb"] = float(round(rng.uniform(-20, 20), 1))
    st.session_state["seed"] = int(rng.integers(1, 100_000))


def new_try():
    """Everything random again: drone start, speed, heading, the flight path, the buildings and the noise."""
    randomize()
    st.session_state["auto_gen"] = True


def reseed_on_generate():
    if st.session_state["reseed"]:
        st.session_state["seed"] = int(np.random.default_rng().integers(1, 100_000))


def bld_area(cfg):
    xa, xb = sorted([cfg.bs_a[0], cfg.bs_b[0]])
    return (xa - 150.0, xb + 150.0), (80.0, 650.0)


def build_path(cfg, scn, rng):
    s = st.session_state
    if s["motion"] == "Straight line":
        return straight_path(scn.pos, scn.vel, 1)
    xa, xb = sorted([cfg.bs_a[0], cfg.bs_b[0]])
    return random_path(rng, scn.pos, s["speed"], s["heading"], s["climb"], int(s["n_frames"]), float(s["dt"]),
                       vmax=float(s["vmax_path"]), yaw_sigma_deg=float(s["yaw"]),
                       bounds=((xa - 100.0, xb + 100.0), (100.0, 600.0), (10.0, 120.0)))


def build_buildings(cfg, rng):
    s = st.session_state
    if not s["bld_on"]:
        return []
    xr, yr = bld_area(cfg)
    return random_buildings(rng, int(s["n_bld"]), xr, yr, (float(s["rcs_lo"]), float(s["rcs_hi"])))


def build_buildings_with(cfg, rng):
    """Random buildings with the current building settings, ignoring the on/off checkbox (used by Monte-Carlo)."""
    s = st.session_state
    xr, yr = bld_area(cfg)
    return random_buildings(rng, int(s["n_bld"]), xr, yr, (float(s["rcs_lo"]), float(s["rcs_hi"])))


def draw_buildings(ax, blds, view="top"):
    from matplotlib.patches import Rectangle
    for i, b in enumerate(blds):
        if view == "top":
            ax.add_patch(Rectangle((b.pos[0] - b.size[0] / 2, b.pos[1] - b.size[1] / 2), b.size[0], b.size[1],
                                   color=GREY, alpha=0.55, lw=0))
            ax.annotate(str(i + 1), (b.pos[0], b.pos[1]), color="white", ha="center", va="center", fontsize=7)
        else:
            ax.add_patch(Rectangle((b.pos[1] - b.size[1] / 2, 0), b.size[1], b.size[2], color=GREY, alpha=0.45, lw=0))
            ax.plot(b.pos[1], b.pos[2], ".", color="white", ms=3)


def build_cfg() -> SystemConfig:
    s = st.session_state
    return SystemConfig(bs_a=(s["ax"], s["ay"], s["az"]), bs_b=(s["bx"], s["by"], s["bz"]),
                        tx_power_dbm=s["tx_dbm"], rcs_dbsm=s["rcs"], rx_gain_db=s["rx_gain"])


def build_scn(present: bool = True) -> Scenario:
    s = st.session_state
    h, c = np.radians(s["heading"]), np.radians(s["climb"])
    vel = (s["speed"] * np.cos(c) * np.cos(h), s["speed"] * np.cos(c) * np.sin(h), s["speed"] * np.sin(c))
    return Scenario(pos=(s["dx"], s["dy"], s["dz"]), vel=vel, present=present,
                    snr_mode="link" if s["snr_mode"] == "Physical link budget" else "manual",
                    snr_db=s["snr_db"], swerling=s["swerling"], deact_pct=float(s["deact"]),
                    direct_residual_db=s["resid_db"] if s["resid_on"] else -np.inf,
                    clutter_cnr_db=s["clut_db"] if s["clut_on"] else -np.inf,
                    clutter_center_m=s["clut_c"], clutter_width_m=s["clut_w"],
                    stability_db=float(s["stab_db"]) if s["bld_on"] else np.inf)


def thresholds() -> tuple[float, float, bool]:
    cal = load_calibration()
    if cal:
        return cal["gamma_paf"], cal["alpha_cfar"], True
    return 4.5, 18.0, False


tab1, tab2, tab3 = st.tabs(["Interface 1 - Scenario Builder", "Interface 2 - Detection Dashboard",
                            "Validation results"])

# ============================================================================================ INTERFACE 1
with tab1:
    left, right = st.columns([1, 1.15])
    with left:
        st.subheader("Scenario Builder")
        st.markdown("**Base stations** (arrays face +y; x along the baseline, z = height)")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("BsA - sensing transmitter")
            st.number_input("x [m]", key="ax"); st.number_input("y [m]", key="ay"); st.number_input("z [m]", key="az")
        with c2:
            st.caption("BsB - sensing receiver")
            st.number_input("x [m]", key="bx"); st.number_input("y [m]", key="by"); st.number_input("z [m]", key="bz")
        st.markdown("**Drone**")
        c1, c2, c3 = st.columns(3)
        c1.number_input("x [m]", key="dx"); c2.number_input("y [m] (distance in front)", key="dy")
        c3.number_input("altitude z [m]", key="dz", min_value=0.0)
        c1, c2, c3 = st.columns(3)
        c1.number_input("speed [m/s]", key="speed", min_value=0.0, max_value=150.0)
        c2.number_input("heading in x-y plane [deg]", key="heading", min_value=-180.0, max_value=180.0)
        c3.number_input("climb angle [deg]", key="climb", min_value=-60.0, max_value=60.0)
        st.radio("Motion model", ["Straight line", "Random maneuvering path"], key="motion", horizontal=True,
                 help="Straight line = one snapshot with a constant velocity.  Random path = a sequence of frames; "
                      "the drone turns, speeds up, slows down, climbs and descends (a different path every try). "
                      "The values above are its starting position, speed, heading and climb angle.")
        if st.session_state["motion"] != "Straight line":
            c1, c2, c3, c4 = st.columns(4)
            c1.number_input("frames", 2, 100, key="n_frames")
            c2.number_input("time between frames [s]", 0.1, 5.0, key="dt", step=0.1)
            c3.number_input("max speed [m/s]", 1.0, 100.0, key="vmax_path")
            c4.number_input("turn randomness [deg/sqrt(s)]", 0.0, 90.0, key="yaw")
        st.button("Randomize drone start (between BsA and BsB)", on_click=randomize, key="btn_random")

        st.markdown("**Buildings** (static reflectors, drawn at random)")
        c1, c2, c3 = st.columns(3)
        c1.checkbox("Add random buildings", key="bld_on")
        c2.number_input("number of buildings", 1, 15, key="n_bld")
        c3.number_input("receiver stability [dB]", 20.0, 120.0, key="stab_db", step=5.0,
                        help="How steady the receiver is from one frame to the next. Background subtraction can only "
                             "cancel a building down to this level below its own echo; a building that is 35 dB "
                             "stronger than the drone needs a stability well above 35 dB.")
        c1, c2, c3 = st.columns(3)
        c1.number_input("building RCS min [dBsm]", key="rcs_lo")
        c2.number_input("building RCS max [dBsm]", key="rcs_hi")
        c3.number_input("empty-scene reference frames", 1, 64, key="ref_frames",
                        help="Drone-free frames averaged into the stored reference scan.")

        st.markdown("**Channel**")
        st.radio("SNR source", ["Manual SNR", "Physical link budget"], key="snr_mode", horizontal=True)
        if st.session_state["snr_mode"] == "Manual SNR":
            st.slider("SNR at BsB, best-aligned beam, per sample [dB]", -30.0, 20.0, key="snr_db", step=0.5)
        else:
            c1, c2, c3 = st.columns(3)
            c1.number_input("BsA transmit power [dBm]", key="tx_dbm")
            c2.number_input("Drone RCS [dBsm]", key="rcs")
            c3.number_input("BsB receive array gain [dB]", key="rx_gain",
                            help="Paper: 0 dB (no receive beamforming). >0 assumes BsB steers a receive beam at the drone - an optimistic extension.")
        c1, c2, c3 = st.columns(3)
        c1.checkbox("Swerling-1 fading", key="swerling")
        c2.slider("Beams deactivated [%]", 0, 100, key="deact", step=5)
        c3.number_input("Random seed", key="seed", min_value=0, step=1)
        st.checkbox("New random seed on every Generate (new noise, path and buildings each try)", key="reseed")
        with st.expander("Non-idealities (extensions beyond the paper)"):
            st.checkbox("Imperfect direct-path cancellation", key="resid_on")
            st.number_input("Residual direct-path power above noise [dB]", key="resid_db")
            st.checkbox("Range-localised interference band (non-stationary noise floor)", key="clut_on")
            c1, c2, c3 = st.columns(3)
            c1.number_input("CNR at centre [dB]", key="clut_db")
            c2.number_input("centre [m]", key="clut_c"); c3.number_input("width [m]", key="clut_w")

        g1, g2, g3 = st.columns(3)
        gen = g1.button("Generate echo", type="primary", key="btn_generate", on_click=reseed_on_generate)
        gen0 = g2.button("Generate noise-only echo (no drone)", key="btn_generate_h0", on_click=reseed_on_generate)
        g3.button("New random try (drone, path, buildings)", key="btn_newtry", on_click=new_try,
                  help="Randomises the drone start, speed and heading, then draws a new flight path, new buildings "
                       "and new noise, and generates the echo.")

    auto = st.session_state.pop("auto_gen", False)
    if gen or gen0 or auto:
        cfg = build_cfg()
        beams = build_beams(cfg)
        seed = int(st.session_state["seed"])
        rng = np.random.default_rng(seed)               # noise and fading
        rng_scene = np.random.default_rng(seed + 12345)    # flight path and buildings
        scn = build_scn(present=not gen0)
        path = build_path(cfg, scn, rng_scene)
        blds = build_buildings(cfg, rng_scene)
        echoes, z_ref = simulate_sequence(cfg, scn, path, blds, beams, rng, int(st.session_state["ref_frames"]))
        st.session_state.update(echo=echoes[0], seq=echoes, path=path, blds=blds, z_ref=z_ref, seed_used=seed,
                                cfg=cfg, scn=scn, beams=beams)
        for k_ in ("result", "track"):
            st.session_state.pop(k_, None)

    with right:
        st.subheader("True scenario (ground truth)")
        if "echo" not in st.session_state:
            st.info("Set the scenario and press **Generate echo**.")
        else:
            cfg, scn, echo = st.session_state["cfg"], st.session_state["scn"], st.session_state["echo"]
            path, blds = st.session_state["path"], st.session_state["blds"]
            t = echo.truth
            st.caption(f"Seed used: {st.session_state['seed_used']} (type it in the seed box and untick the reseed "
                       f"option to repeat exactly this try)  |  frames: {path.k}"
                       + (f", {path.dt:g} s apart ({(path.k - 1) * path.dt:g} s of flight)" if path.k > 1 else "")
                       + f"  |  buildings: {len(blds)}")
            if not t.present:
                st.warning("Noise-only echo generated (hypothesis H0: no drone).")
            else:
                st.table({
                    "quantity": ["bistatic range d = d_tx + d_rx", "BsA -> drone", "drone -> BsB", "delay",
                                 "bistatic velocity", "Doppler shift", "speed", "SNR (best beam)",
                                 "beams on / total", "unambiguous range / velocity span"],
                    "value": [f"{t.d:.1f} m", f"{t.d_tx:.1f} m", f"{t.d_rx:.1f} m", f"{t.tau * 1e6:.3f} us",
                              f"{t.v_bi:+.2f} m/s", f"{t.f_d:+.1f} Hz", f"{t.speed:.1f} m/s",
                              f"{t.snr_best_db:.1f} dB", f"{int(echo.active.sum())} / {echo.active.size}",
                              f"{cfg.unamb_range:.0f} m / +-{cfg.unamb_velocity / 2:.0f} m/s"]})
                fig, ax = plt.subplots(1, 2, figsize=(8, 3.4))
                beams = st.session_state["beams"]
                for u, on in zip(beams.dirs, echo.active):
                    ax[0].plot([cfg.bs_a[0], cfg.bs_a[0] + 700 * u[0]], [cfg.bs_a[1], cfg.bs_a[1] + 700 * u[1]],
                               color=BLUE if on else GREY, alpha=0.12 if on else 0.05, lw=1)
                ax[0].plot(*cfg.bs_a[:2], "^", color=BLUE, ms=10, label="BsA (Tx)")
                ax[0].plot(*cfg.bs_b[:2], "v", color=GREEN, ms=10, label="BsB (Rx)")
                draw_buildings(ax[0], blds, "top")
                draw_buildings(ax[1], blds, "side")
                if path.k > 1:
                    ax[0].plot(path.pos[:, 0], path.pos[:, 1], "-", color=RED, lw=1.6, label="drone path")
                    ax[0].plot(path.pos[0, 0], path.pos[0, 1], "o", color=RED, ms=8, label="start")
                    ax[0].plot(path.pos[-1, 0], path.pos[-1, 1], "s", color=RED, ms=7, label="end")
                    ax[1].plot(path.pos[:, 1], path.pos[:, 2], "-", color=RED, lw=1.6)
                    ax[1].plot(path.pos[0, 1], path.pos[0, 2], "o", color=RED, ms=8)
                    ax[1].plot(path.pos[-1, 1], path.pos[-1, 2], "s", color=RED, ms=7)
                else:
                    ax[0].plot(t.pos[0], t.pos[1], "o", color=RED, ms=9, label="drone")
                    ax[0].annotate("", xy=(t.pos[0] + 3 * t.vel[0], t.pos[1] + 3 * t.vel[1]), xytext=t.pos[:2],
                                   arrowprops=dict(arrowstyle="->", color=RED))
                    ax[1].plot(t.pos[1], t.pos[2], "o", color=RED, ms=9)
                if blds:
                    ax[0].plot([], [], "s", color=GREY, label="buildings")
                ax[0].autoscale_view()
                ax[0].set(title="Top view", xlabel="x [m]", ylabel="y [m]", aspect="equal"); ax[0].legend(fontsize=7)
                ax[1].plot(cfg.bs_a[1], cfg.bs_a[2], "^", color=BLUE, ms=10)
                ax[1].plot(cfg.bs_b[1], cfg.bs_b[2], "v", color=GREEN, ms=10)
                ax[1].autoscale_view()
                ax[1].set(title="Side view", xlabel="y [m]", ylabel="height z [m]")
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)
            if blds:
                with st.expander(f"Buildings in this try ({len(blds)})", expanded=False):
                    st.dataframe(building_table(cfg, scn, blds, path.pos[0]), hide_index=True)
                    st.caption("'echo vs drone' = how much stronger the building's echo is than the drone's, in the "
                               "best-aligned beam.  Buildings are static (zero Doppler) and give the same echo in "
                               "every frame.")

# ============================================================================================ INTERFACE 2
with tab2:
    st.subheader("Detection Dashboard")
    if "echo" not in st.session_state:
        st.info("Generate an echo in Interface 1 first.")
    else:
        g_paf, a_cfar, calibrated = thresholds()
        with st.expander("Processing parameters", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            n_pad = c1.select_slider("IFFT size N' (range)", [256, 512, 1024, 2048], value=512)
            l_pad = c2.select_slider("FFT size L' (Doppler)", [8, 16, 32, 64, 128], value=32)
            gamma_a = c3.number_input("Aggregation threshold gamma_a", value=6.0, step=0.5)
            angle_m = c4.selectbox("Angle estimator", ["strongest", "weighted"],
                                   help="strongest = centre of the strongest SSB beam (paper); "
                                        "weighted = amplitude-weighted average of the 4 strongest beams (our extension)")
            c1, c2 = st.columns(2)
            window = c1.selectbox("Taper across subcarriers (sidelobe control)", ["none", "hann", "blackmanharris"],
                                  help="Needed together with direct-path blanking when the direct path is not cancelled perfectly.")
            guard = c2.number_input("Direct-path blanking half-width [m]", value=60.0, step=10.0)
            c1, c2, c3, c4 = st.columns(4)
            gamma = c1.number_input("PAF threshold gamma", value=float(round(g_paf, 2)), step=0.1)
            alpha = c2.number_input("CFAR threshold", value=float(round(a_cfar, 1)), step=0.5)
            refine = c3.checkbox("Parabolic peak interpolation", value=True)
            gate = c4.checkbox("Blank range bins around the direct path", value=False)
            if not calibrated:
                st.warning("No calibration file found - using default thresholds.  Run the validation script or press "
                           "Recalibrate.")
            elif (n_pad, l_pad, gamma_a, window) != (512, 32, 6.0, "none"):
                st.warning("Thresholds were calibrated for N'=512, L'=32, gamma_a=6, no taper (P_FA = 0.03). "
                           "Press Recalibrate after changing these.")
            if st.button("Recalibrate thresholds for P_FA = 0.03 (~1 min)", key="btn_recal"):
                prog = st.progress(0.0)
                cfg0 = st.session_state["cfg"]
                cal = calibrate(cfg0, ProcConfig(n_pad=n_pad, l_pad=l_pad, gamma_a=gamma_a, gate_direct=gate,
                                                  window=None if window == "none" else window, gate_guard_m=guard),
                                n=800, progress=prog.progress)
                save_calibration(cal)
                st.success(f"New thresholds: PAF {cal['gamma_paf']:.2f}, CFAR {cal['alpha_cfar']:.1f} - reload the page.")
        pc = ProcConfig(n_pad=n_pad, l_pad=l_pad, gamma_a=gamma_a, gamma=gamma, cfar_alpha=alpha,
                        angle_method=angle_m, refine=refine, gate_direct=gate,
                        window=None if window == "none" else window, gate_guard_m=guard)

        seq, blds = st.session_state["seq"], st.session_state["blds"]
        z_ref_all = st.session_state.get("z_ref")
        K = len(seq)
        suppress = False
        if blds or st.session_state.get("mc_bld"):
            suppress = st.checkbox("Suppress static clutter: subtract the stored reference scan of the empty scene",
                                   key="suppress",
                                   help="Buildings give the same echo in every frame, a drone does not. Subtracting a "
                                        "drone-free reference removes the buildings and keeps the drone, even a "
                                        "hovering one.")
        z_ref = z_ref_all if suppress else None
        fr = 0
        if K > 1:
            fr = st.slider("Frame to inspect", 0, K - 1, 0, key="frame",
                           help="The detailed view below shows one frame; 'Whole path' further down processes all frames.")
        echo = seq[fr]

        if st.button("Process received signal", type="primary", key="btn_process"):
            cfg, beams = st.session_state["cfg"], st.session_state["beams"]
            st.session_state["result"] = (process(echo.Z, cfg, beams, echo.active, pc, z_ref), pc, fr, suppress)

        if "result" in st.session_state and st.session_state["result"][2:] == (fr, suppress):
            res, pc_used = st.session_state["result"][:2]
            cfg, beams = st.session_state["cfg"], st.session_state["beams"]
            t = echo.truth
            if K > 1:
                st.caption(f"Frame {fr + 1} of {K}")
            if blds and not suppress:
                st.warning("Buildings are present and not suppressed: their strong echoes dominate the map, so a "
                           "detection here is usually a building, not the drone.")

            v1, v2, v3 = st.columns(3)
            v1.metric("PAF detector", "UAV present" if res.detected_paf else "no UAV",
                      f"PAF = {res.p_paf:.2f}  (threshold {pc_used.gamma:.2f})", delta_color="off")
            v2.metric("CFAR detector", "UAV present" if res.detected_cfar else "no UAV",
                      f"T = {res.p_cfar:.1f}  (threshold {pc_used.cfar_alpha:.1f})", delta_color="off")
            v3.metric("Beams aggregated", f"{res.eta.size}", f"strongest beam #{res.beam_idx}", delta_color="off")

            col_a, col_b = st.columns([1.1, 1])
            with col_a:
                fig, ax = plt.subplots(figsize=(6.2, 4.2))
                d_top = st.slider("Range axis max [m]", 200, int(cfg.unamb_range), int(min(cfg.unamb_range, max(1000, 1.6 * (t.d if t.present else 600)))), 50)
                db = 10 * np.log10(res.agg_map / res.agg_map.mean())
                im = ax.pcolormesh(res.v_axis, res.d_axis, db, shading="auto", cmap="viridis")
                ax.set_ylim(0, d_top)
                if t.present:
                    ax.plot(t.v_bi, t.d, "o", mfc="none", mec="white", ms=13, mew=1.5, label="truth")
                ax.plot(res.v_hat, res.d_hat, "x", color=RED, ms=10, mew=2.2, label="estimate")
                if blds:
                    br_ = building_ranges(cfg, blds)
                    ax.plot([res.v_axis.max()] * len(br_), br_, "<", color=ORANGE, ms=6, label="building range")
                ax.set(title="Range-velocity map (aggregated beams)", xlabel="bistatic velocity [m/s]",
                       ylabel="bistatic range [m]"); ax.grid(False); ax.legend(loc="upper right", fontsize=8)
                fig.colorbar(im, label="dB over mean power"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
            with col_b:
                fig, ax = plt.subplots(figsize=(5.2, 4.2))
                paf = np.where(np.isfinite(res.paf), res.paf, 0)
                cols = [ORANGE if i in res.eta else BLUE for i in range(len(paf))]
                ax.bar(range(len(paf)), paf, color=cols, width=0.8)
                ax.axhline(pc_used.gamma, color=RED, ls="--", lw=1.2, label=f"detection threshold {pc_used.gamma:.2f}")
                ax.axhline(pc_used.gamma_a, color=GREY, ls=":", lw=1.2, label=f"aggregation threshold {pc_used.gamma_a:.1f}")
                ax.set(title="Peak-to-average factor per beam", xlabel="surveillance beam", ylabel="PAF"); ax.legend(fontsize=8)
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)

            st.markdown("#### Estimates vs ground truth")
            if t.present:
                err_pos = float(np.linalg.norm(res.pos_hat - t.pos)) if np.all(np.isfinite(res.pos_hat)) else float("nan")
                th_t = np.degrees(np.arctan2(t.u_a[0], t.u_a[1]))
                st.table({
                    "quantity": ["bistatic range [m]", "bistatic velocity [m/s]", "Doppler [Hz]", "azimuth of BsA beam [deg]",
                                 "elevation [deg]", "BsA -> drone distance [m]", "x [m]", "y [m]", "z [m]"],
                    "truth": [f"{t.d:.2f}", f"{t.v_bi:+.2f}", f"{t.f_d:+.1f}", f"{np.degrees(np.arcsin(t.u_a[0] / max(np.cos(np.arcsin(t.u_a[2])), 1e-9))):+.1f}",
                              f"{np.degrees(np.arcsin(t.u_a[2])):+.1f}", f"{t.d_tx:.1f}", f"{t.pos[0]:.1f}", f"{t.pos[1]:.1f}", f"{t.pos[2]:.1f}"],
                    "estimate": [f"{res.d_hat:.2f}", f"{res.v_hat:+.2f}", f"{2 * res.v_hat / cfg.wavelength:+.1f}",
                                 f"{np.degrees(res.theta_hat):+.1f}", f"{np.degrees(res.phi_hat):+.1f}", f"{res.d_tx_hat:.1f}",
                                 f"{res.pos_hat[0]:.1f}", f"{res.pos_hat[1]:.1f}", f"{res.pos_hat[2]:.1f}"],
                    "error": [f"{res.d_hat - t.d:+.2f}", f"{res.v_hat - t.v_bi:+.2f}", f"{2 * (res.v_hat - t.v_bi) / cfg.wavelength:+.1f}",
                              f"{np.degrees(res.theta_hat) - np.degrees(np.arcsin(t.u_a[0] / max(np.cos(np.arcsin(t.u_a[2])), 1e-9))):+.1f}",
                              f"{np.degrees(res.phi_hat - np.arcsin(t.u_a[2])):+.1f}", f"{res.d_tx_hat - t.d_tx:+.1f}",
                              f"{res.pos_hat[0] - t.pos[0]:+.1f}", f"{res.pos_hat[1] - t.pos[1]:+.1f}", f"{res.pos_hat[2] - t.pos[2]:+.1f}"]})
                st.metric("Localisation error (3D)", f"{err_pos:.1f} m")

                # localisation geometry: ray + bistatic-range ellipse (cut at the estimated height)
                fig, ax = plt.subplots(figsize=(6.2, 4.4))
                pa, pb = np.asarray(cfg.bs_a), np.asarray(cfg.bs_b)
                zc = float(res.pos_hat[2]) if np.isfinite(res.pos_hat[2]) else float(t.pos[2])
                xs = np.linspace(min(pa[0], pb[0]) - 300, max(pa[0], pb[0]) + 300, 300)
                ys = np.linspace(-50, max(t.pos[1], res.pos_hat[1] if np.isfinite(res.pos_hat[1]) else 0) * 1.6 + 100, 300)
                X, Y = np.meshgrid(xs, ys)
                D = np.sqrt((X - pa[0]) ** 2 + (Y - pa[1]) ** 2 + (zc - pa[2]) ** 2) + \
                    np.sqrt((X - pb[0]) ** 2 + (Y - pb[1]) ** 2 + (zc - pb[2]) ** 2)
                ax.contour(X, Y, D, levels=[res.d_hat], colors=[ORANGE], linewidths=1.6)
                ax.plot([], [], color=ORANGE, label="bistatic-range ellipse")
                ray_end = pa + 1.6 * max(res.d_tx_hat if np.isfinite(res.d_tx_hat) else 500, 300) * res.u_hat
                ax.plot([pa[0], ray_end[0]], [pa[1], ray_end[1]], color=BLUE, lw=1.4, label="beam ray")
                ax.plot(*pa[:2], "^", color=BLUE, ms=10); ax.plot(*pb[:2], "v", color=GREEN, ms=10)
                ax.plot(t.pos[0], t.pos[1], "o", color=RED, ms=9, label="true position")
                ax.plot(res.pos_hat[0], res.pos_hat[1], "x", color="black", ms=10, mew=2, label="estimated position")
                ax.set(title="Localisation: ray x ellipse", xlabel="x [m]", ylabel="y [m]", aspect="equal"); ax.legend(fontsize=8)
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)
            else:
                st.info("Noise-only scenario: nothing to compare against. Any detection here is a false alarm.")

        # ------------------------------------------------------------------------------ whole path
        if K > 1 or blds:
            st.markdown("---")
            st.markdown("#### Whole path: process every frame")
            c1, c2 = st.columns([1, 3])
            det_name = c1.radio("Detector", ["PAF", "CFAR"], horizontal=True, key="track_det")
            if c1.button("Process all frames", key="btn_track"):
                cfg, beams = st.session_state["cfg"], st.session_state["beams"]
                with st.spinner("Processing every frame ..."):
                    off_ = process_sequence(seq, cfg, beams, pc, None)
                    on_ = process_sequence(seq, cfg, beams, pc, z_ref_all) if blds else None
                st.session_state["track"] = (off_, on_, pc)
            if "track" in st.session_state:
                off_, on_, pcT = st.session_state["track"]
                cfg = st.session_state["cfg"]
                dn = det_name.lower()
                scores = {"without suppression": score_sequence(off_, seq, cfg, blds, pcT, dn)}
                if on_ is not None:
                    scores["with suppression"] = score_sequence(on_, seq, cfg, blds, pcT, dn)
                present = seq[0].truth.present
                names = list(scores)
                st.table({
                    "": ["frames", "detected (anything)", "drone found (range within 25 m)", "locked on a building",
                         "detected elsewhere", "drone missed", "median 3D position error (drone frames)"],
                    **{nm: [str(scores[nm]["n"]), f"{scores[nm]['detected']:.0%}", f"{scores[nm]['correct']:.0%}",
                            str(scores[nm]["counts"]["building"]), str(scores[nm]["counts"]["other"]),
                            str(scores[nm]["counts"]["missed"]),
                            "-" if not np.isfinite(scores[nm]["pos_med"]) else f"{scores[nm]['pos_med']:.1f} m"]
                       for nm in names}})
                if not present:
                    st.info("Noise-only run: every detection is a false alarm.")
                show_on = suppress and on_ is not None
                show = on_ if show_on else off_
                lab = scores["with suppression" if show_on else "without suppression"]["labels"]
                st.caption("Plots below show the " + ("suppressed" if show_on else "unsuppressed") +
                           " processing (change the checkbox above to switch).")
                sty = {"drone": (GREEN, "o", "drone found"), "building": (ORANGE, "s", "locked on building"),
                       "other": (GREY, "x", "detected elsewhere")}
                fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1.2, 1]))
                a = ax[0]
                draw_buildings(a, blds, "top")
                a.plot(*cfg.bs_a[:2], "^", color=BLUE, ms=9); a.plot(*cfg.bs_b[:2], "v", color=GREEN, ms=9)
                if present:
                    Ptrue = np.array([e.truth.pos for e in seq])
                    a.plot(Ptrue[:, 0], Ptrue[:, 1], "-", color=RED, lw=1.2, label="true path")
                    a.plot(Ptrue[0, 0], Ptrue[0, 1], "o", color=RED, ms=6)
                used = set()
                for k_, (r, lb) in enumerate(zip(show, lab)):
                    ph = r.pos_hat
                    if lb in sty and np.all(np.isfinite(ph)) and abs(ph[0]) < 1500 and -200 < ph[1] < 1500:
                        col, mk, nm = sty[lb]
                        a.plot(ph[0], ph[1], mk, color=col, ms=6, label=None if lb in used else nm)
                        used.add(lb)
                    elif lb == "missed" and present:
                        a.plot(Ptrue[k_, 0], Ptrue[k_, 1], "o", mfc="none", mec=RED, ms=9,
                               label=None if "missed" in used else "drone missed")
                        used.add("missed")
                a.set(title="Top view: estimated positions per frame", xlabel="x [m]", ylabel="y [m]", aspect="equal")
                a.autoscale_view()
                if a.get_legend_handles_labels()[0]:
                    a.legend(fontsize=7, loc="upper right")
                b = ax[1]
                stat = np.array([r.p_paf if dn == "paf" else r.p_cfar for r in show])
                thr = pcT.gamma if dn == "paf" else pcT.cfar_alpha
                cols = [sty[lb][0] if lb in sty else (RED if lb == "missed" else BLUE) for lb in lab]
                b.bar(range(len(stat)), stat, color=cols, width=0.8)
                b.axhline(thr, color=RED, ls="--", lw=1.2, label=f"threshold {thr:.2f}")
                if stat.max() / max(stat.min(), 1e-9) > 20:
                    b.set_yscale("log")
                b.set(title=f"{det_name} statistic per frame", xlabel="frame", ylabel=det_name); b.legend(fontsize=8)
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        # ------------------------------------------------------------------------------ Monte-Carlo runner
        st.markdown("---")
        st.markdown("#### Monte-Carlo runner (random drone positions, fading and noise)")
        st.caption("Uses the channel settings from Interface 1 (SNR, fading, deactivation, non-idealities) and the "
                   "processing parameters above; the drone is dropped randomly between BsA and BsB with a random "
                   "speed and heading.")
        c1, c2 = st.columns([1, 3])
        n_mc = c1.number_input("trials per hypothesis", 50, 3000, 200, 50)
        c2.checkbox("Draw a new random set of buildings in every trial (building settings from Interface 1; "
                    "suppression follows the checkbox above)", key="mc_bld")
        if c1.button("Run Monte-Carlo", key="btn_mc"):
            cfg = st.session_state["cfg"]
            beams = st.session_state["beams"]
            rng = np.random.default_rng(int(st.session_state["seed"]) + 1)
            mc_b = st.session_state["mc_bld"]
            base = build_scn(True)
            if mc_b:
                base = replace(base, stability_db=float(st.session_state["stab_db"]))
            reg = Region(x=(cfg.bs_a[0], cfg.bs_b[0]))
            bfn = (lambda g: build_buildings_with(cfg, g)) if mc_b else None
            refn = int(st.session_state["ref_frames"]) if (mc_b and suppress) else 0
            bar = st.progress(0.0, text="H1: drone present")
            h1 = run_trials(int(n_mc), cfg, pc, beams, rng, lambda g: random_scenario(g, reg, base, True),
                            lambda f: bar.progress(f * 0.5, text="H1: drone present"), bfn, refn)
            h0 = run_trials(int(n_mc), cfg, pc, beams, rng, lambda g: replace(base, present=False),
                            lambda f: bar.progress(0.5 + f * 0.5, text="H0: noise only"), bfn, refn)
            bar.empty()
            st.session_state["mc"] = (h0, h1, pc)
        if "mc" in st.session_state:
            h0, h1, pcm = st.session_state["mc"]
            det = h1.p_paf > pcm.gamma
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("P_D (PAF)", f"{(h1.p_paf > pcm.gamma).mean():.3f}")
            m2.metric("P_FA (PAF)", f"{(h0.p_paf > pcm.gamma).mean():.3f}")
            m3.metric("P_D (CFAR)", f"{(h1.p_cfar > pcm.cfar_alpha).mean():.3f}")
            m4.metric("P_FA (CFAR)", f"{(h0.p_cfar > pcm.cfar_alpha).mean():.3f}")
            inl = det & np.isfinite(h1.d_err) & (np.abs(h1.d_err) <= 25.0)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("outlier rate (detected, |range err| > 25 m)", f"{(1 - inl.sum() / max(det.sum(), 1)):.3f}",
                      help="Detected trials where the peak was a noise peak / deep fade rather than the drone.")
            m2.metric("range RMSE (non-outlier)", f"{rmse(h1.d_err, inl):.2f} m")
            m3.metric("velocity RMSE (non-outlier)", f"{rmse(h1.v_err, inl):.2f} m/s")
            m4.metric("median localisation error", f"{np.nanmedian(h1.pos_err[det]) if det.any() else float('nan'):.1f} m")
            fig, ax = plt.subplots(figsize=(5.2, 3.8))
            for name, a, b, col, ls in [("PAF", h0.p_paf, h1.p_paf, BLUE, "-"), ("CFAR", h0.p_cfar, h1.p_cfar, ORANGE, "--")]:
                _, pfa, pdv = roc_curve(a, b)
                ax.plot(pfa, pdv, color=col, ls=ls, label=f"{name} (AUC {auc(pfa, pdv):.3f})")
            ax.plot([0, 1], [0, 1], color=GREY, ls=":", label="random")
            ax.set(title="ROC from this Monte-Carlo run", xlabel="P_FA", ylabel="P_D"); ax.legend(fontsize=8)
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

# ============================================================================================ VALIDATION
with tab3:
    st.subheader("Validation against the reference paper and extension results")
    res_path = os.path.join(OUT, "results.json")
    if not os.path.exists(res_path):
        st.info("Run `python experiments/run_validation.py` to generate the validation figures.")
    else:
        figs = [("fig_crb_vs_N_L.png", "CRB for different N and L (paper Fig. 2)"),
                ("fig_rmse_vs_crb.png", "Range / velocity RMSE of the 2D-FFT vs the CRB (paper Fig. 3)"),
                ("fig_rmse_vs_padding.png", "RMSE vs zero-padding size"),
                ("fig_roc.png", "ROC curves - PAF vs CFAR"),
                ("fig_pd_rmse_vs_snr.png", "P_D and RMSE vs SNR"),
                ("fig_pd_vs_deactivation.png", "P_D vs beam deactivation (paper Fig. 4)"),
                ("fig_vs_range.png", "Drone range sweep (physical link budget)"),
                ("fig_vs_speed.png", "Drone speed sweep"),
                ("fig_localisation_error.png", "Localisation error - strongest beam vs amplitude-weighted beams"),
                ("fig_cfar_vs_paf_clutter.png", "CFAR vs PAF with a non-stationary noise floor"),
                ("fig_direct_path_residual.png", "Imperfect direct-path cancellation"),
                ("fig_buildings.png", "Random buildings: background subtraction vs receiver stability")]
        for f, cap in figs:
            p = os.path.join(OUT, f)
            if os.path.exists(p):
                st.image(p, caption=cap)
        with st.expander("Raw numbers (results.json)"):
            st.json(json.load(open(res_path)))
