# AutoEQ — 64-band Parametric EQ for Poweramp

A psychoacoustic-aware IEM correction engine for **Poweramp 64-band Parametric EQ**.  
Takes stereo L/R measurements and a target curve, outputs a 64-band PEQ preset ready to enter into Poweramp.

> **Platform:** Python on PC (web interface). Output is for Poweramp on Android.

---

## Features

- **64-band parametric EQ** output with exact biquad filter modeling
- **Web-based GUI** — runs in your browser, no terminal needed
- **Auto file detection** — drop your measurement files and run
- **Auto normalization** — finds the flattest region between 500–800 Hz automatically
- **Auto freq-end detection** — detects usable upper limit from measurement rolloff
- **Auto output naming** — output file named from your IEM + target automatically
- **EQ Bands selector** — choose 16 / 32 / 64 bands
- **Psychoacoustic optimization** — perceptual weighting, Huber robust loss, multi-resolution IRLS
- **Exact biquad pipeline** — no filter shape approximations
- **Post-EQ simulation** — always generated; upload `*_PostEQ.txt` to Squiglink to verify visually
- **Built-in target** — Euvony Reference (vocal-centric, detail-hunter, airy)

---

## Requirements

```
Python 3.8+
numpy
scipy
tqdm  (optional)
flask
```

> `run.bat` installs Flask automatically on first launch. For numpy/scipy/tqdm, run once:

```bash
pip install numpy scipy tqdm
```

---

## Quick Start

**1. Clone or download this repo**

**2. Place your measurement files in the `Measurement/` folder**

Files must be named with `[L]` / `[R]` or `_L` / `_R` in the filename.  
Example: `KZ Libra [L].txt` and `KZ Libra [R].txt`

**3. Double-click `run.bat`**

Browser opens automatically at `localhost:5000`. That's it.

---

## Using the Interface

![Main interface](docs/screenshot_main.png)

**Left panel — Input:**
- Upload your **L** and **R** measurement files (`.txt` or `.csv`)
- Pick a **target curve** from the dropdown, or upload a custom one
- Set optional parameters: Freq End, Norm Freq, Norm Mode, Sample Rate, EQ Bands

**Click Run**

![Loading](docs/screenshot_loading.png)

A loading overlay appears with a live progress indicator and elapsed timer.

**Right panel — Results:**

![Result](docs/screenshot_result.png)

- **Preamp value** — displayed prominently in amber. Set this in Poweramp first.
- **RMSE** — how closely the EQ matches the target
- **Download Preset** — the PEQ file to import into Poweramp
- **Download PostEQ Simulation** — upload to [squig.link](https://squig.link) to visually verify
- **Frequency Response chart** — Raw L+R avg (grey), Target (blue dashed), Post-EQ (green)

![Accuracy table](docs/screenshot_accuracy.png)

- **Accuracy vs Target table** — per-octave error, color-coded ✓ / ⚠ / ✗

![Filter bands](docs/screenshot_bands.png)

- **Filter Bands** — all 64 band values ready to enter into Poweramp

---

## How to Import into Poweramp

1. Open Poweramp → three-dot menu → **Equalizer**
2. Set **Preamp** to the value shown (e.g. `-8.8 dB`) — **mandatory, do this first**
3. Import the preset file directly — no manual band entry needed

> ⚠️ Always set Preamp first. Without it, boost filters will cause digital clipping.

---

## Included Targets

All targets are in the `targets/` folder. Tuned for female vocal-centric listening.

| File | Character |
|---|---|
| `Euvony Personal Flat.txt` | Analytical reference, no bass emphasis |
| `Euvony Personal Target Musical.txt` | Musical, long listening, balanced |
| `Euvony Personal Target Moderate.txt` | Moderate bass, vocal forward |
| `Euvony Personal Target.txt` | Personal reference |
| `Euvony VocalGod.txt` | Vocal-forward, earlier version |
| `Euvony_VocalGod_v1i.txt` | **Vocal supremacy** — female vocal absolute dominant |
| `Diffuse Field Target.txt` | Diffuse field reference |
| `Harman 2019v2 Target.txt` | Harman 2019 v2 consumer target |

> Targets were tuned using spectral analysis of real tracks (aespa, YOASOBI, GFRIEND, Lilas Ikuta) and iterated through listening tests.

---

## Built-in Target: Euvony Reference

The default target when no custom target is selected.

**Philosophy:** Vocal-centric · Detail-hunter · Airy

- **150–350 Hz** — female chest voice elevated
- **500 Hz** — dipped to remove IEM boxiness
- **2.5–3.8 kHz** — presence plateau, singer's formant dominant
- **4–9.5 kHz** — maintained for clarity and transient precision
- **9.5–20 kHz** — linear descent, natural and extended

Compared to references:
- vs **Harman**: less bass, +5 dB @ 3 kHz, much more treble extension
- vs **Diffuse Field**: similar sub-bass, +5 dB vocal body at 200–300 Hz

---

## Input File Format

```
# Comments start with #
20.0,96.80
20.3,96.78
...
20000.0,65.12
```

Supported separators: comma or tab.  
AutoEQ CSV format (`frequency,raw,smoothed`) is also supported — `raw` column is used.

---

## Normalization

All input files (L, R, target) must be on the same absolute dB scale.

**Recommended — Squiglink:**  
Upload to [squig.link](https://squig.link) → `Normalize → 60 dB → 500 Hz → Download`

---

## Algorithm Overview

**Optimization stages:**
1. Gauss-Newton warm start with Jacobian column normalization
2. 2-pass IRLS with Huber robust loss and multi-resolution residual decomposition
3. Adaptive band reallocation (iterative greedy, up to 4 passes)
4. Joint fc+Q+gain optimizer (L-BFGS-B with residual-aware importance sampling)
5. Post-joint reallocation check

**Regularization:** Ridge · Smoothness (1st + 3rd derivative) · Band energy · Q penalty · Perceptual pole radius · HF proximity · Phase slope

**Psychoacoustic features:** Perceptual weighting · A-weighting masking · ERB-based min-phase blend · Adaptive σ frequency jitter · Multi-resolution IRLS

---

## Measurement Sources

- [squig.link](https://squig.link) — community database aggregator
- [crinacle.com](https://crinacle.com) — EARS + 711 measurements
- [graph.hangout.audio](https://graph.hangout.audio)

---

## License

MIT — see [LICENSE](LICENSE)

---

## Author

**DAPAAADF** (Euvony)  
github.com/DAPAAADF
