<div align="center">
📡 UAV Sensing in 5G-Advanced Networks using ISAC

Detect and localise drones using the 5G signals that are already in the air.

An interactive Python simulator of passive bistatic drone sensing with 5G NR SSB blocks, with a two-interface Streamlit app, a CFAR detector compared against the paper's PAF detector, and ray × ellipse localisation.

Python NumPy SciPy Matplotlib Streamlit

B.Tech Major Project · Group 37 · Dept. of Electronics and Communication Engineering, MANIT Bhopal Supervisor: Dr. Rahul Pal

</div>
What it does

Integrated Sensing and Communication (ISAC) lets a cellular network see as well as talk. This project simulates a passive bistatic radar built from two 5G base stations that never transmit anything extra:

BsA transmits the normal 5G SSB (Synchronization Signal Block) beams, which it sends anyway.
BsB only listens. It picks up the echoes that bounce off a drone and measures how long they took and how much they shifted in frequency.

From that echo the simulator estimates the drone's bistatic range, bistatic velocity, beam direction and finally its 3-D position, and compares every estimate against the ground truth.

The work is a Python reproduction of the signal chain of Jopanya & Osorio, "Utilizing 5G NR SSB Blocks for Passive Detection and Localization of Low-Altitude Drones", IEEE SPAWC 2025, plus our own extensions (listed below).

Scenario ─► SSB echo ─► 2D FFT ─► PAF / CFAR ─► range, velocity, angle ─► localisation ─► compare with truth
            (delay, Doppler,        (range-Doppler   (detection)           (beam ray ∩ bistatic
             path loss, fading,      map per beam)                          range ellipse)
             AWGN)
Key features
Interface 1 · Scenario Builder
Place the drone manually or press New random try to drop it at a random position, speed and heading
Straight-line flight or a random manoeuvring path (turns, speed changes, climbs, reflected at the area edge)
Random buildings, redrawn on every new try: static reflectors with zero Doppler and different strength each time
Choose the SNR from a physical link budget (Tx power, RCS, path loss) or type it in manually
Switch on non-idealities: Swerling-1 fading, beam deactivation, uncancelled direct path, range-localised clutter
Top view, side view and the range-velocity ground truth, all plotted from the stored scenario
Interface 2 · Detection Dashboard
Generates the SSB echo for the saved scenario and produces the range-velocity heatmap (per beam and aggregated)
Runs the PAF detector (the paper) and CA-CFAR (our addition) side by side
Shows the estimated range, velocity, beam, angle and localised position against the truth with the error
Background subtraction with a drone-free reference scan removes buildings, so a stationary drone can still be found
Monte Carlo mode: ROC curve, P_D, P_FA, RMSE and localisation error over thousands of random scenarios, with the CRB for comparison
Validation viewer
Browse every figure produced by the validation experiments (ROC, RMSE vs CRB, P_D vs SNR, range and speed sweeps, clutter, buildings, and more)
Core ideas
Concept	What it means here
SSB	240 subcarriers × 4 OFDM symbols at 60 kHz spacing (14.4 MHz), 15 GHz carrier, real PSS / SSS / PBCH layout (830 of 960 resource elements carry signal)
Beams	10 × 10 uniform planar array, 81-beam codebook, R = 45 upward-looking surveillance beams
Bistatic range	d = d_tx + d_rx = c·τ: the sum of the two path lengths, which defines an ellipse with BsA and BsB as foci
Bistatic velocity	v = −½ · v_drone · (u_A + u_B), giving the Doppler f_d = 2v/λ
Channel	`Z[n,l] = √SNR · (
Processing	IFFT over subcarriers gives range, FFT over symbols gives Doppler (zero-padded to 512 × 32)
Detection	PAF = max / mean of the amplitude map; CFAR = range-adaptive cell-averaging threshold. Both calibrated for P_FA = 0.03
Localisation	beam ray ∩ bistatic ellipse: d_tx = (d² − L_b²) / (2(d − u·b)), position = p_A + d_tx · u
Bound	closed-form CRB for range and velocity (inverse Fisher information)

True resolutions: range c/B ≈ 20.8 m, velocity λ·Δf/(2L) = 150 m/s. With only 4 OFDM symbols the Doppler resolution is coarse, which is why the heatmap shows a streak along the velocity axis. Zero-padding refines the estimate, not the resolution.

Our extensions beyond the paper
CA-CFAR detector compared against the paper's PAF threshold
Localisation of the drone from the beam ray and the bistatic-range ellipse
Interactive two-interface application (Scenario Builder → Detection Dashboard)
Random manoeuvring flight path instead of a straight line
Random static buildings that change on every try, plus background subtraction so that stationary drones are still detected
Calibration of both thresholds on noise-only runs for a fixed false-alarm probability
Real SSB resource-element mask (PSS / SSS / PBCH), receiver-stability limit on clutter cancellation
Results

Figures are produced by experiments/run_validation.py and saved in outputs/.

Range-velocity map	ROC (PAF vs CFAR)
Range velocity map	ROC
RMSE vs CRB	Localisation error
RMSE vs CRB	Localisation error
P_D vs beam deactivation	Buildings and clutter suppression
P_D vs deactivation	Buildings

Calibrated thresholds for P_FA = 0.03 (1500 noise-only trials): γ_PAF ≈ 4.54 (paper quotes γ = 4) and CFAR α ≈ 18.5. Full numbers are in outputs/results.json and outputs/validation_log.txt.

Tech stack
Layer	Tech
Language	Python 3.12+
Signal model and processing	NumPy: complex arrays, fft / ifft, linear algebra for the CRB
Filtering	SciPy: convolve1d for the CFAR sliding window, signal.windows
Plots	Matplotlib (headless Agg backend)
Frontend	Streamlit with session_state to pass the scenario from Interface 1 to Interface 2
Data structures	dataclasses for SystemConfig, Scenario, ProcConfig, Building, Path
Tests	plain assert tests plus Streamlit AppTest for headless UI checks
Hosting	Streamlit Community Cloud (from this GitHub repository)
Project structure
uav5g-isac-sensing/
├── app.py                      # Streamlit frontend: Interface 1, Interface 2, validation viewer
├── requirements.txt            # numpy, scipy, matplotlib, streamlit
├── uav5g/                      # backend package
│   ├── __init__.py
│   ├── config.py               # SystemConfig (Table I), Scenario, ProcConfig, Building
│   ├── array.py                # 10x10 UPA steering vectors, 81-beam codebook, 45 surveillance beams
│   ├── channel.py              # bistatic geometry, link budget, SSB mask, echo simulation
│   ├── processing.py           # 2D FFT, PAF, CFAR, aggregation, estimators, localisation
│   ├── crb.py                  # closed-form range/velocity Cramér-Rao bounds
│   ├── montecarlo.py           # random scenarios, ROC, RMSE, threshold calibration
│   └── scene.py                # random flight path, random buildings, multi-frame simulation, scoring
├── experiments/
│   └── run_validation.py       # all validation experiments -> outputs/
├── tests/
│   └── test_core.py            # 15 unit tests
└── outputs/                    # figures, results.json, calibration.json, validation_log.txt

Build order / data flow: config → array → channel → processing → crb → montecarlo → scene → app. Each file only imports from the ones before it.

Getting started
Prerequisites
Python 3.12 or newer
Nothing else: all four dependencies install from PyPI, and no API keys are needed
1. Clone and install
bash
git clone https://github.com/abhiramkonjeti-sudo/uav5g-isac-sensing.git
cd uav5g-isac-sensing
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
2. Run the app
bash
streamlit run app.py

Open http://localhost:8501. Start in Interface 1, build or randomise a scenario, then go to Interface 2 to run detection.

3. Optional: regenerate the figures and run the tests
bash
python experiments/run_validation.py          # all figures in outputs/ (~15 min on 2 cores)
python experiments/run_validation.py --quick  # ~2 min smoke test
python tests/test_core.py                     # 15 unit tests
Conventions worth knowing
Geometry: both arrays face +y, x runs along the baseline, z is height. Default BsA = (0, 0, 0), BsB = (200, 0, 0); the drone is dropped 150–500 m in front of the baseline at 20–100 m height. Elevation is fully modelled.
Bistatic velocity is not the drone's speed. It is the rate of change of d_tx + d_rx. Interface 2 compares estimates against this value.
SNR is the per-sample SNR at BsB for a perfectly aligned beam; beams pointing away see SNR × (|g_r|/M)².
PAF uses the amplitude peak-to-mean ratio. The paper's printed expression cannot produce values such as γ = 4, so thresholds are calibrated on noise-only runs instead.
Angle comes from the transmit beam of BsA. BsB does no receive beamforming, as in the paper.
Buildings are static, so they sit at zero Doppler with identical echoes in every frame. A drone-free reference scan is subtracted to remove them; how well it cancels is limited by receiver stability.
Tests

tests/test_core.py checks, among others: the beam codebook against the paper, aligned-beam gain equals M, the SSB mask layout, bistatic range and Doppler geometry, exactness of the localisation formula, noise-free estimates hitting the truth, CRB closed form = FIM⁻¹, the random path staying bounded, random buildings differing between tries yet staying static within one, and background subtraction removing buildings while keeping a stationary drone.

Limitations and future work
 Real measured SSB captures instead of simulated ones
 Tracking across frames (Kalman filter) rather than independent per-frame estimates
 Extended 3-D building models with shadowing and multipath
 More than one drone at a time
 Moving clutter (vehicles, trees) in addition to static buildings
Reference

N. Jopanya and J. Osorio, "Utilizing 5G NR SSB Blocks for Passive Detection and Localization of Low-Altitude Drones," IEEE International Workshop on Signal Processing Advances in Wireless Communications (SPAWC), 2025.

