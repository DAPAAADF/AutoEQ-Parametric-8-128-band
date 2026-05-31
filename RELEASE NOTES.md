# v17.3 — Poweramp Frequency Floor Fix

## What Changed

### EQ bands now start from 20 Hz minimum

Poweramp's parametric EQ supports frequencies from **20 Hz** minimum. Previously, the joint optimizer's center-frequency perturbation could drift bands below 20 Hz (e.g. Band 1 at 18.7 Hz), wasting a filter slot on a frequency Poweramp cannot reproduce.

**Fix:**
- Optimization fit range is now clamped to 20 Hz minimum, regardless of how low the measurement file starts
- After all optimization stages, any band that drifted below 20 Hz is clamped back to 20 Hz
- Measurements below 20 Hz are still fully used for normalization accuracy — only EQ band placement is affected

**Console output** now shows if any bands were clamped:
```
FC clamp   : 2 band(s) below 20 Hz clamped to 20 Hz
```

No changes to output format, preset structure, or GUI.

---

## Installation

No new dependencies. Replace `AutoEQ.py` and restart.

---

# v17.2 — Speed + Auto Naming + Bands Selector

## Highlights

### Faster optimization
Run time reduced from ~4–5 minutes to **~1.5–2.5 minutes** by tuning default iteration counts — without sacrificing accuracy. Convergence typically happens well before the old limits, so results are identical.

### Auto output naming
Output files are now named automatically from your IEM and target:
```
KZ X ANGELEARS LIBRA X - EUVONY VOCALG0D V1I.txt
KZ PR3 - HARMAN 2019V2 TARGET.txt
```
Preview updates live in the GUI sidebar before you run. Still overrideable via `-o` on CLI.

### EQ Bands selector (GUI)
Choose how many bands to use: **16 / 32 / 64**. Useful if you're using the preset in a player other than Poweramp, or want a lighter filter chain.

---

## Changes

**AutoEQ.py:**
- Default `--iters` reduced from 3000 → 1500
- Default `--joint-iters` reduced from 1200 → 800
- Auto output filename generation from measurement + target names
- `autoeq_config.txt` updated with `fs` and `bands` documentation

**GUI (`gui/`):**
- Bands dropdown: 16 / 32 / 64
- Live output filename preview in sidebar
- Output filename shown in result bar after run
- Download route now dynamically finds latest result file by name
- Subprocess timeout increased to 300s

---

## Notes

- Preset format is unchanged — compatible with Poweramp, Equalizer APO, Peace, and any player reading standard AutoEQ `.txt` format
- For Poweramp: max 64 bands. For Equalizer APO / Peace: any band count works

---

## Installation

```bash
pip install numpy scipy flask tqdm
```

**GUI:**
```
Double-click gui/run.bat   (Windows)
python gui/gui.py          (all platforms)
```

**CLI:**
```bash
python AutoEQ.py \
  --target "targets/Euvony VocalGod.txt" \
  --norm-freq 800 --norm-mode freq \
  --freq-end 12000 \
  --fs 192000
```

---

*See [CHANGELOG.md](CHANGELOG.md) for full version history.*
