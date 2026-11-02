<div align="center">

# 📡 UAV Sensing in 5G-Advanced Networks using ISAC

**Passive Bistatic Drone Detection and 3D Localization via 5G NR Synchronization Signal Blocks (SSB)**

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Signal_Model-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-CFAR_&_DSP-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge)](https://matplotlib.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive_UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

*B.Tech Major Project · Group 37 · Dept. of Electronics and Communication Engineering, MANIT Bhopal*  
*Supervised by **Dr. Rahul Pal***

[Overview](#-overview) •
[Novel Extensions](#-novel-extensions-beyond-the-paper) •
[Mathematical Model](#-mathematical-signal-model) •
[Results](#-validation--results) •
[Architecture](#-system-architecture) •
[Quick Start](#-getting-started)

</div>

---

## 📖 Overview

**Integrated Sensing and Communication (ISAC)** enables next-generation cellular networks to act as perceptive radar sensors without dedicating extra spectrum or hardware. This repository implements an end-to-end Python simulation and interactive dashboard for **passive bistatic radar** using standard **5G NR Synchronization Signal Blocks (SSB)**:

1. **Transmitter ($\text{BsA}$):** Broadcasts standard periodic 5G NR SSB beam sweeps across an 81-beam codebook ($R = 45$ upward-looking surveillance beams).
2. **Receiver ($\text{BsB}$):** Operates completely passively, capturing echoes reflected off low-altitude UAVs to extract time-of-arrival (bistatic range) and Doppler frequency shifts (bistatic velocity).
3. **3D Localization:** Intersects the highest-energy transmit beam ray from $\text{BsA}$ with the bistatic range ellipsoid defined by focal points $\text{BsA}$ and $\text{BsB}$ to pinpoint the UAV in 3D space.

This work reproduces and significantly extends the signal processing chain proposed in:
> **N. Jopanya and J. Osorio**, *"Utilizing 5G NR SSB Blocks for Passive Detection and Localization of Low-Altitude Drones,"* IEEE International Workshop on Signal Processing Advances in Wireless Communications (SPAWC), 2025.

```text
┌──────────────┐    ┌─────────────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌───────────────────────┐
│  3D Scenario │───►│   5G NR SSB Echo    │───►│  2D Range-Doppler│───►│   PAF / CA-CFAR  │───►│    3D Localization    │
│  Generation  │    │   Channel Model     │    │  Periodogram     │    │   Target         │    │  (Beam Ray ∩ Bistatic │
│ (UAV + Urban │    │ (Delay, Doppler,    │    │ (IFFT Subcarrier,│    │   Detection      │    │   Range Ellipsoid)    │
│  Buildings)  │    │  Swerling-1, AWGN)  │    │  FFT OFDM Symbol)│    │ (P_FA = 0.03)    │    │   & CRB Evaluation    │
└──────────────┘    └─────────────────────┘    └──────────────────┘    └──────────────────┘    └───────────────────────┘

#,Extension,Technical Impact
1,2D Cell-Averaging CFAR (CA-CFAR),Benchmarks adaptive local noise-floor thresholding against the paper's global Peak-to-Average-Floor (PAF) detector.
2,Closed-Form 3D Localization,Solves the analytical intersection between the BsA directional beam ray and the BsA–BsB bistatic range ellipsoid.
3,Static Building Clutter & Clutter Cancellation,Models multi-scatterer zero-Doppler urban buildings and implements reference-frame background subtraction (enabling detection of hovering/stationary UAVs) bounded by hardware receiver stability limits.
4,3GPP-Compliant SSB Resource Mask,"Models the exact 5G NR SSB time-frequency grid (PSS, SSS, PBCH, and DMRS layout occupying 830 of 960 resource elements) rather than an idealized full grid."
5,Kinematic Manoeuvring Flight Paths,"Supports multi-frame continuous trajectories with heading turns, altitude climbs, acceleration, and boundary reflections alongside linear flight paths."
6,Empirical PFA​ Threshold Calibration,"Calibrates both PAF (γPAF​≈4.54) and CA-CFAR (α≈18.5) via 1,500 Monte Carlo noise-only trials to strictly enforce PFA​=0.03."
7,Two-Stage Interactive Streamlit Suite,"Decouples 3D scenario design (Scenario Builder) from signal processing, detection heatmaps, and Monte Carlo benchmarking (Detection Dashboard)."

