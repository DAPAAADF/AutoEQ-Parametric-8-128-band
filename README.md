# AutoEQ — 64-band Parametric EQ for Poweramp

A psychoacoustic-aware IEM correction engine for **Poweramp 64-band Parametric EQ**.  
Takes stereo L/R measurements and a target curve, outputs a 64-band PEQ preset ready to enter into Poweramp.

> **Platform:** Python on PC. Output is for Poweramp on Android.

---

## Features

- **64-band parametric EQ** output with exact biquad filter modeling
- **Auto file detection** — drop your measurement files in the folder and run
- **Auto normalization** — finds the flattest region between 500–800 Hz automatically
- **Auto freq-end detection** — detects usable upper limit from measurement rolloff (HF peak drop method)
- **Psychoacoustic optimization** — perceptual weighting, Huber robust loss, multi-resolution IRLS
- **Exact biquad pipeline** — no filter shape approximations, all optimization on real digital filters
- **Joint fc+Q+gain optimizer** — fine-tunes band center frequencies and Q values after gain solve
- **Adaptive band reallocation** — iteratively moves underused bands to residual hotspots
- **HF boost cap** — prevents optimizer from forcing boost into rolled-off treble regions
- **Post-EQ simulation** — write corrected FR for visual verification in Squiglink
- **Built-in target** — Euvony Reference (vocal-centric, detail-hunter, airy)

---

## Requirements

```
Python 3.8+
numpy
scipy
tqdm  (optional, for progress bars)
```

Install dependencies:

```bash
pip install numpy scipy tqdm
```

---

## Quick Start

**1. Clone or download this repo**

**2. Place your measurement files in the repo folder**

Files must be named with `_L` / `_R` or `[L]` / `[R]` in the filename.  
Example: `KZ Libra [L].txt` and `KZ Libra [R].txt`

**3. Run**

```bash
python AutoEQ.py
```

The script auto-detects your measurement files, normalizes them, and outputs `autoeq_result.txt` ready for Poweramp.

---

## Usage

### Simplest — auto everything
```bash
python AutoEQ.py
```

### With a custom target
```bash
python AutoEQ.py --target targets/Euvony_VocalGod_v1i.txt
```

### With manual anchor at 800 Hz
```bash
python AutoEQ.py --target targets/Euvony_VocalGod_v1i.txt --norm-freq 800 --norm-mode freq
```

### Limit to 16 kHz and generate simulation file
```bash
python AutoEQ.py --freq-end 16000 --simulate
```

### Full example
```bash
python AutoEQ.py \
    --target targets/Euvony_VocalGod_v1i.txt \
    --norm-freq 800 --norm-mode freq \
    --freq-end 16000 \
    --simulate \
    -o my_result.txt
```

---

## Input File Format

All files use the same two-column format:

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

**Recommended method — Squiglink:**
Upload your file to [squig.link](https://squig.link) → `Normalize → 60 dB → 500 Hz → Download`

**Python method:**
```python
import numpy as np

def normalize(path, out_path, ref_hz=500, ref_db=60):
    data = np.loadtxt(path, delimiter=',')
    val = np.interp(ref_hz, data[:, 0], data[:, 1])
    data[:, 1] = data[:, 1] - val + ref_db
    np.savetxt(out_path, data, fmt='%.10f,%.10f')

normalize('IEM_L.txt', 'IEM_L_norm.txt')
normalize('IEM_R.txt', 'IEM_R_norm.txt')
```

> The built-in target is pre-normalized. When using `--norm-mode freq` with `--norm-freq 800`, target and measurements are internally aligned automatically.

---

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `left` | auto | Left channel measurement file |
| `right` | auto | Right channel measurement file |
| `--target`, `-t` | built-in | Custom target file |
| `--output`, `-o` | `autoeq_result.txt` | Output file path |
| `--freq-start` | `20` | Lower bound of optimization range (Hz) |
| `--freq-end` | auto | Upper bound (Hz). Auto-detected from rolloff. |
| `--norm-freq` | auto | Normalization anchor (Hz). Auto-selects flattest point 500–800 Hz. |
| `--norm-mode` | `perceptual` | `freq` / `energy` / `perceptual` |
| `--bands` | `64` | Number of PEQ bands |
| `--simulate` | off | Write post-EQ simulation file (`*_PostEQ.txt`) |
| `--spawn` | off | Iterative band spawning for complex IEMs |
| `--fs` | `48000` | Target device sample rate (Hz) |

---

## Output

The result file contains:

**1. Header** — run parameters, RMSE, preamp value

**2. Accuracy table** — per-octave accuracy vs target at standard checkpoints

**3. Parametric EQ (Poweramp format)**
```
Preamp: -9.5 dB

Filter  1: ON PK Fc     42.18 Hz  Gain +3.21 dB  Q 0.50
Filter  2: ON PK Fc    128.54 Hz  Gain -2.18 dB  Q 0.70
...
Filter 64: ON PK Fc  14823.00 Hz  Gain +1.45 dB  Q 12.00
```

### How to import into Poweramp

1. Open Poweramp → three-dot menu → **Equalizer**
2. Set **Preamp** to the value shown (e.g. `-9.5 dB`) — **this is mandatory**
3. Enable **Parametric EQ** → set to **64 bands**
4. For each Filter line: enter Frequency, Gain, and Q
5. Filter type must be **Peak (PK)** for all bands

> ⚠️ Always set the Preamp value first. Without it the boost filters will cause digital clipping.

---

## Included Targets

All targets are in the `targets/` folder. These are personal targets tuned for female vocal-centric listening.

| File | Character |
|---|---|
| `Euvony_Personal_Flat.txt` | Analytical reference, no bass emphasis |
| `Euvony_Personal_Target_Musical.txt` | Musical, long listening, balanced |
| `Euvony_Personal_Target_Moderate.txt` | Moderate bass, vocal forward |
| `Euvony_AnalyticalVocal_v1r3.txt` | Analytical but vocal-aware |
| `Euvony_VocalGod_v1i.txt` | **Vocal supremacy** — female vocal absolute dominant |
| `Euvony_Custom_v1r5.txt` | Detail extraction, presence + air focused |

> These targets were tuned using spectral analysis of real tracks (aespa, YOASOBI, GFRIEND, Lilas Ikuta) and iterated through listening tests.

---

## Built-in Target: Euvony Reference

The default target when no `--target` is specified.

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

## Workflow Tips

- Use `--simulate` to generate a `_PostEQ.txt` file, then upload to [squig.link](https://squig.link) alongside your target to visually verify the result before listening
- For A/B comparison between targets, use `--norm-mode energy` so all presets are loudness-neutral
- For vocal-focused targets, use `--norm-freq 800 --norm-mode freq`

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
