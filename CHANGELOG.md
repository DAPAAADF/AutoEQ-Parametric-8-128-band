# Changelog

All notable changes to this project are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [v17.3] — Poweramp Frequency Floor Fix

### Fixed

#### EQ bands drifting below 20 Hz (Poweramp minimum)
Poweramp's parametric EQ has a minimum center frequency of **20 Hz**. Previously, the joint fc+Q+gain optimizer's center-frequency perturbation (±`fc_range`, default ±15%) could drift initial bands below this limit. Example: a band initialized at 22 Hz with `fc_range=0.15` could settle at ~18.7 Hz — a frequency Poweramp cannot represent, wasting a filter slot.

**Two-stage fix:**

1. **Fit range floor** — `f_start` for EQ optimization is now `max(freq_start, 20.0, l_pts[0,0])`, ensuring band placement never starts below 20 Hz even if `--freq-start` is set lower or measurements begin at sub-20 Hz frequencies.

2. **Post-optimization clamp** — after all optimization stages complete (IRLS → realloc → joint → post-joint realloc), all `fc` values are hard-clamped to 20 Hz minimum:
   ```python
   fcs = np.maximum(fcs, 20.0)
   ```
   Console prints how many bands were clamped, if any:
   ```
   FC clamp   : 2 band(s) below 20 Hz clamped to 20 Hz
   ```

**Measurement data below 20 Hz is unaffected** — it is still loaded and used for normalization accuracy. Only EQ band placement is restricted.

### Changed
- `--freq-start` help text updated to document the 20 Hz Poweramp minimum
- `autoeq_config.txt` note updated

---

## [v17.2] — Speed + Auto Naming + Bands Selector

### Changed

#### Optimization speed — default iterations reduced
- `--iters` default: 3000 → **1500**
- `--joint-iters` default: 1200 → **800**

Convergence typically completes well before the old limits — `Converged: True` was already printed early in most runs. Reducing defaults cuts typical run time from 4–5 minutes to **1.5–2.5 minutes** with no measurable difference in output quality. Both parameters remain user-overridable for complex IEMs that need more iterations.

#### Auto output filename
Output files are now named automatically from the IEM measurement filename and target, instead of always defaulting to `autoeq_result.txt`:

```
KZ X ANGELEARS LIBRA X - EUVONY VOCALG0D V1I.txt
KZ PR3 - HARMAN 2019V2 TARGET.txt
IEM NAME - EUVONY REF.txt   ← when using built-in target
```

Logic: strip file extension + L/R channel tag from measurement filename, uppercase both sides, join with ` - `. Sanitized for filesystem compatibility.

CLI: still overrideable with `-o my_output.txt`. GUI: preview shown live in sidebar before run, final filename shown in result bar after run.

#### `autoeq_config.txt` updated
Added documented entries for `fs` and `bands` with recommended values and notes.

### Added (GUI)

#### EQ Bands selector
New dropdown in Parameters section: **16 / 32 / 64 bands**. Passes `--bands` to AutoEQ.py. Useful for:
- Players with lower band limits
- Lighter filter chains for less complex IEMs
- Equalizer APO / Peace users who don't need full 64 bands

#### Live output filename preview
Sidebar shows the auto-generated output filename as you select files and target, before running. Updates on every file/target change.

#### Filename in result bar
After run completes, the result bar shows the actual output filename above the download buttons.

#### Download route — dynamic filename lookup
Download route now finds the latest result by modification time instead of hardcoded `result.txt`. Supports dynamic filenames correctly.

#### Subprocess timeout
Increased from 180s to 300s to accommodate slower machines or high band counts.

---

## [Unreleased] — GUI Release

### Added — Web GUI (`gui.py`)

A full web-based interface for AutoEQ, built with Flask + vanilla HTML/CSS/JS.  
Run with `python gui.py` — browser opens automatically at `localhost:5000`.  
Windows users can double-click `run.bat` for a one-click launch (auto-installs Flask).

**Input panel (sidebar):**
- Drag & drop or click-to-upload for L and R channel measurement files (`.txt`, `.csv`)
- Target curve selector — dropdown loads all files from `targets/` folder automatically
- Custom target upload — upload any `.txt` file outside the built-in collection; mutually exclusive with dropdown (selecting one clears the other)
- Parameters: Freq End (Hz), Norm Freq (Hz), Norm Mode (`perceptual` / `freq` / `energy`), Sample Rate (`44100` / `48000` / `96000` / `192000`)

**Loading experience:**
- Full-screen animated overlay during optimization (blocks accidental interaction)
- 9-bar waveform animation
- 6-step progress indicator with glowing dot advancing through: Loading → Normalizing → IRLS → Band reallocation → Joint optimizer → Output
- Real-time elapsed timer (updates every 100ms)
- All steps flash green on completion, overlay fades out

**Results panel (main area):**
- Result bar appears after run with: ✓ status, **Preamp value** (large amber display), RMSE, Download Preset button, Download PostEQ Simulation button
- **Frequency Response chart** — Chart.js log-scale, three curves: Raw L+R avg (grey), Target (blue dashed), Post-EQ (green). All normalized at 1 kHz for easy visual comparison. Interactive tooltip on hover.
- **Accuracy vs Target table** — all standard checkpoints from 20 Hz to 20 kHz, color-coded: ✓ green (< 0.5 dB), ⚠ amber (< 1.5 dB), ✗ red (≥ 1.5 dB). Positive/negative errors color-coded separately.
- **Filter Bands grid** — all 64 bands in 4-column grid. Each cell shows band number, center frequency, gain (amber = boost, blue = cut), and Q value.
- **Console output** — collapsible panel showing full AutoEQ.py stdout for debugging

**Downloads:**
- `autoeq_result.txt` — ready-to-enter PEQ preset for Poweramp
- `autoeq_result_PostEQ.txt` — simulated post-EQ FR curve, uploadable to squig.link

**Branding:**
- Euvony butterfly logo in header
- "Euvony AutoEQ" branding with serif typography

---

## [v17.1] — Bug Fixes & Documentation

### Fixed

#### Unicode crash on Windows (critical)
AutoEQ.py outputs `→` characters in console (e.g. `weighted mean offset = -34.597 dB → centered to 0`). On Windows with default cp1252 encoding, calling this via `subprocess.run()` caused `UnicodeDecodeError` and silent failure.

**Fix:** `PYTHONIOENCODING=utf-8` injected into subprocess environment + `encoding="utf-8", errors="replace"` in `subprocess.run()`. CLI usage was unaffected.

#### No-op interpolation in `optimize_gains_biquad()`
```python
# Before — interpolating a grid onto itself (always identity)
raw_fit = np.interp(f_fit, f_fit, raw_avg_db)

# After — direct assignment
raw_fit = raw_avg_db
```
Functionally harmless (rolloff slope estimate was correct either way), but wasteful.

#### Duplicate frequency entries in `_BUILTIN_TARGET`
Last 3 frequency points duplicated at array tail (indices 479–481):
`19330.5459`, `19611.7151`, `19896.9741` Hz each appeared twice.
`builtin_target_pts()` already deduplicated via `np.unique()` so no effect on results. Raw data cleaned.

#### `--simulate` / `--no-simulate` flag inversion in README
README documented `--simulate` as opt-in. Actual behavior: simulation is **on by default**.
The real flag is `--no-simulate` to suppress it. `--simulate` was not a registered argparse argument — passing it had no effect, no error. All README examples corrected.

#### Target filename inconsistency — spaces vs underscores
README and all CLI examples used underscore filenames (`Euvony_Personal_Flat.txt`) but actual files in `targets/` use spaces (`Euvony Personal Flat.txt`). Copying any example from README would produce `FileNotFoundError`. Only `Euvony_VocalGod_v1i.txt` correctly uses underscores. README corrected with actual filenames + note to always quote paths.

#### Missing target files in README table
Two targets listed in README table did not exist in repo:
- `Euvony_AnalyticalVocal_v1r3.txt` — not present
- `Euvony_Custom_v1r5.txt` — not present

Table updated to list all 8 files actually present in `targets/`.

#### `OVERWRITE_EXISTING` config constant unused in `sync_lyrics.py`
`OVERWRITE_EXISTING = True` declared but never checked in `process()`. Tracks with existing `LYRICS` tags are always overwritten. Documented as known issue pending fix.

### Changed

#### README — complete revision
- All usage examples audited and corrected
- `--fs` parameter expanded: explicit warning that biquad coefficients are sample-rate dependent; must match Poweramp's actual output sample rate. Filters at 8–15 kHz shift ±0.4–0.6 dB between `fs=48000` and `fs=192000`
- `--freq-end 12000` now recommended for most budget DD/hybrid IEMs (previously `16000`). Setting `freq-end` beyond the driver's usable rolloff causes optimizer to boost aggressively into the rolloff zone
- Target table corrected with actual filenames and CLI quoting note
- Output section: two files always generated per run (`.txt` + `_PostEQ.txt`)
- Quick Start updated

### Known Issues

#### IR `min_flatness_db` false positive
The "Min frame flatness" metric in IR diagnostic consistently reports −70 to −75 dB ("ringing region — fatigue risk") even for trivially clean single-filter cascades. Cause: sliding 4ms Hanning window STFT on sparse biquad IR always produces near-zero power bins between energy peaks, driving geometric mean → 0 and flatness → −∞. This is a property of the metric on IIR filter impulse responses, not evidence of actual audible ringing. `spectral_flatness_db` (global) is reliable. `min_flatness_db` should be disregarded until metric is revised.

#### HF optimizer artifact near `freq-end` boundary
When `--freq-end` is set beyond the IEM's physical rolloff, optimizer bands near the boundary boost aggressively. On Libra X with `--freq-end 16000`: bands 62–64 received +5.7 to +7.7 dB at 14.5–15.7 kHz, `max adj delta = 7.58 dB`. With `--freq-end 12000`: max delta dropped to 2.69 dB, RMSE improved 1.40 → 0.15 dB. Always set `freq-end` at or below the physical rolloff point.

---

## [v17.0] — Initial Public Release

### Added
- Full IEM correction pipeline: raw L/R measurements → 64-band PEQ preset for Poweramp
- **Exact biquad modeling** — optimization runs on real digital biquad filters, not approximations. Analytical gradients for gain (`dR/dg`), center frequency (`dR/d log fc`), and Q (`dR/d log Q`)
- **Multi-stage optimizer:**
  1. Gauss-Newton warm start with Jacobian column RMS normalization
  2. 2-pass IRLS with Huber robust loss (δ=1.0)
  3. Multi-resolution residual reweighting (macro/meso/micro layers, frequency-adaptive σ)
  4. Adaptive band reallocation (iterative greedy, up to 4 passes)
  5. Joint fc+Q+gain optimizer (L-BFGS-B, residual-aware importance sampling)
  6. Post-joint reallocation check
- **Psychoacoustic regularization:**
  - Perceptual weighting (presence 2k–8k boosted ×2.5)
  - A-weighting masking (ISO 226 approximation)
  - ERB-based min-phase blend (Glasberg & Moore 1990, curvature-aware via cubic spline d²/d(logf)²)
  - Pole radius penalty with HF proximity term (fc/fnyq)^γ
  - Frequency-dependent Q log-penalty (presence region strictest)
  - Phase slope regularizer (analytical ∂φ/∂g)
  - First + third derivative smoothness (jerk penalty)
  - Ridge + band energy regularization
- **HF boost cap** — per-frequency max gain: < 15 kHz: 6 dB, 15–17 kHz: 2.5 dB, 17–19 kHz: 1.5 dB, > 19 kHz: 0.8 dB, tightened by measured rolloff slope
- **Auto-detection:** L/R file, normalization anchor (flattest 500–800 Hz), freq-end (2-stage: absolute floor + HF peak drop)
- **3 normalization modes:** `perceptual` (default), `freq` (anchor), `energy` (loudness-neutral)
- **Post-EQ simulation** — exact biquad cascade applied to raw measurement for squig.link upload
- **IR diagnostic** — temporal centroid, early energy ratio, Schroeder −20 dB decay, spectral flatness
- **Iterative band spawning** (`--spawn`) — grow from 32 to 64 bands at residual hotspots
- **Built-in Euvony Reference target** + 7 targets in `targets/` — tuned for female vocal-centric listening
- `autoeq_config.txt` — persistent run config via key=value file
- `.vscode/tasks.json` — VS Code task runner integration
- Python 3.8+ · numpy · scipy · tqdm (optional)

---

*Developed by DAPAAADF (Euvony) · MIT License*
