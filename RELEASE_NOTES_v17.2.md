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
