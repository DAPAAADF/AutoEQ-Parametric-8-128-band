"""
AutoEQ GUI — Flask backend
Run: python gui.py
Browser opens automatically.
"""

import os
import re
import sys
import json
import threading
import webbrowser
import subprocess
import tempfile
import uuid
import zipfile
import io
import numpy as np
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_file

app    = Flask(__name__, static_folder="static")
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
SCRIPT = Path(__file__).parent.parent / "AutoEQ.py"
TARGET = Path(__file__).parent.parent / "targets"
SESS   = Path(tempfile.gettempdir()) / "autoeq_gui_session"
SESS.mkdir(exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_curve(path):
    pts = []
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                for sep in (",", "\t"):
                    parts = s.split(sep)
                    if len(parts) >= 2:
                        try:
                            pts.append([float(parts[0]), float(parts[1])])
                            break
                        except ValueError:
                            pass
    except Exception:
        pass
    return pts


def avg_lr(left_pts, right_pts):
    if not left_pts or not right_pts:
        return left_pts or right_pts
    l_arr = np.array(left_pts)
    r_arr = np.array(right_pts)
    # Use np.interp — O(N log N), not O(N²)
    r_interp = np.interp(l_arr[:, 0], r_arr[:, 0], r_arr[:, 1])
    avg = (l_arr[:, 1] + r_interp) / 2
    return [[float(f), float(d)] for f, d in zip(l_arr[:, 0], avg)]


def parse_result(path):
    filters, preamp = [], 0.0
    if not Path(path).exists():
        return preamp, filters
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("Preamp:"):
                try:
                    preamp = float(line.split()[1])
                except Exception:
                    pass
            elif line.startswith("Filter"):
                m = re.match(
                    r"Filter\s+(\d+): ON PK Fc\s+([\d.]+) Hz\s+Gain\s+([+-]?[\d.]+) dB\s+Q\s+([\d.]+)",
                    line,
                )
                if m:
                    filters.append({
                        "n":    int(m.group(1)),
                        "fc":   float(m.group(2)),
                        "gain": float(m.group(3)),
                        "q":    float(m.group(4)),
                    })
    return preamp, filters


def parse_accuracy(stdout):
    rows = []
    in_table = False
    for line in stdout.splitlines():
        if "Freq" in line and "Target" in line and "Error" in line:
            in_table = True
            continue
        if in_table:
            if line.strip().startswith("---"):
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    rows.append({
                        "freq":   int(parts[0]),
                        "target": float(parts[1]),
                        "fit":    float(parts[2]),
                        "error":  float(parts[3]),
                        "status": parts[4],
                    })
                except (ValueError, IndexError):
                    if rows:
                        break
    return rows


# ── routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/targets")
def get_targets():
    out = []
    if TARGET.exists():
        for f in sorted(TARGET.glob("*.txt")):
            out.append({"name": f.stem, "path": str(f)})
    return jsonify(out)


@app.route("/api/run", methods=["POST"])
def run():
    lf = request.files.get("left")
    rf = request.files.get("right")
    if not lf or not rf:
        return jsonify({"error": "Upload both L and R files"}), 400

    # UUID prefix prevents filename collision / overwrite between runs
    _uid = uuid.uuid4().hex[:8]
    left_path  = SESS / f"{_uid}_L_{lf.filename}"
    right_path = SESS / f"{_uid}_R_{rf.filename}"
    lf.save(left_path)
    rf.save(right_path)

    target    = request.form.get("target", "").strip()
    freq_end  = request.form.get("freq_end", "").strip()
    norm_freq = request.form.get("norm_freq", "").strip()
    norm_mode = request.form.get("norm_mode", "perceptual").strip()
    fs        = request.form.get("fs", "48000").strip()
    bands     = request.form.get("bands", "64").strip()

    # Advanced params
    use_minphase   = request.form.get("use_minphase",   "1") == "1"
    use_phase      = request.form.get("use_phase",      "0") == "1"
    gain_floor     = request.form.get("gain_floor",     "0.10").strip()
    presence_boost = request.form.get("presence_boost", "2.2").strip()
    treble_boost   = request.form.get("treble_boost",   "1.8").strip()
    fc_range       = request.form.get("fc_range",       "").strip()

    # Direct iters (override preset)
    iters_direct = request.form.get("iters",       "").strip()
    joint_direct = request.form.get("joint_iters", "").strip()
    iters = iters_direct if iters_direct else "2500"
    joint = joint_direct if joint_direct else "2500"

    # Lambda params
    lam        = request.form.get("lam",        "").strip()
    lam_smooth = request.form.get("lam_smooth", "").strip()
    lam_energy = request.form.get("lam_energy", "").strip()
    lam_q      = request.form.get("lam_q",      "").strip()
    lam_gd     = request.form.get("lam_gd",     "").strip()
    lam_driver = request.form.get("lam_driver", "").strip()

    # custom target file upload takes priority
    custom_target_f = request.files.get("target_file")
    if custom_target_f and custom_target_f.filename:
        custom_target_path = SESS / custom_target_f.filename
        custom_target_f.save(custom_target_path)
        target = str(custom_target_path)

    # auto output name: IEM - TARGET.txt
    import re as _re
    def _clean_name(s):
        s = Path(s).stem
        s = _re.sub(r'\s*[\[\(]?[LR][\]\)]?\s*$', '', s, flags=_re.IGNORECASE).strip()
        s = _re.sub(r'[_\-]\s*[LR]\s*$', '', s, flags=_re.IGNORECASE).strip()
        return s.upper()

    iem_name = _clean_name(lf.filename)
    if target:
        tgt_name = Path(target).stem.upper()
    else:
        tgt_name = "EUVONY REF"
    safe_name = _re.sub(r'[^\w\s\-]', '', f"{iem_name} - {tgt_name}").strip()
    out_filename = safe_name + ".txt"
    result_path = SESS / out_filename

    cmd = [sys.executable, str(SCRIPT),
           str(left_path), str(right_path),
           "-o", str(result_path),
           "--norm-mode", norm_mode,
           "--fs", fs or "48000",
           "--bands", bands or "64",
           "--iters", iters,
           "--joint-iters", joint]

    if not use_minphase:
        cmd += ["--no-minphase"]
    if use_phase:
        cmd += ["--lam-ph", "0.0005"]
    if gain_floor:
        cmd += ["--gain-floor", gain_floor]
    if presence_boost:
        cmd += ["--presence-boost", presence_boost]
    if treble_boost:
        cmd += ["--treble-boost", treble_boost]
    if fc_range:
        cmd += ["--fc-range", fc_range]
    if lam:
        cmd += ["--lambda", lam]
    if lam_smooth:
        cmd += ["--lam-smooth", lam_smooth]
    if lam_energy:
        cmd += ["--lam-energy", lam_energy]
    if lam_q:
        cmd += ["--lam-q", lam_q]
    if lam_gd:
        cmd += ["--lam-gd", lam_gd]
    if lam_driver:
        cmd += ["--lam-driver", lam_driver]
    if target:
        cmd += ["--target", target]
    if freq_end:
        cmd += ["--freq-end", freq_end]
    if norm_freq:
        cmd += ["--norm-freq", norm_freq]

    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            cwd=str(Path(__file__).parent.parent),
            encoding="utf-8", errors="replace", env=env
        )
        stdout = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")

        if proc.returncode != 0:
            return jsonify({"error": stdout}), 500

        preamp, filters = parse_result(result_path)
        accuracy        = parse_accuracy(stdout)

        raw_l   = parse_curve(left_path)
        raw_r   = parse_curve(right_path)
        raw_avg = avg_lr(raw_l, raw_r)

        posteq_path = str(result_path).replace(".txt", "_PostEQ.txt")
        post_eq     = parse_curve(posteq_path)

        target_pts = parse_curve(target) if target and Path(target).exists() else []

        return jsonify({
            "success":     True,
            "out_filename": out_filename,
            "output":   stdout,
            "preamp":   preamp,
            "filters":  filters,
            "accuracy": accuracy,
            "curves": {
                "raw":    raw_avg,
                "post":   post_eq,
                "target": target_pts,
            },
        })

    except subprocess.TimeoutExpired:
        return jsonify({"error": "Timeout — optimization took too long (>5 min)"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/download/<kind>")
def download(kind):
    txt_files    = sorted(SESS.glob("*.txt"), key=lambda f: f.stat().st_mtime, reverse=True)
    posteq_files = sorted(SESS.glob("*_PostEQ.txt"), key=lambda f: f.stat().st_mtime, reverse=True)

    if kind == "result":
        result_files = [f for f in txt_files if "_PostEQ" not in f.name and "_report" not in f.name]
        p = result_files[0] if result_files else None
        if p and p.exists():
            return send_file(p, as_attachment=True, download_name=p.name)
        return "Not found", 404

    elif kind == "posteq":
        p = posteq_files[0] if posteq_files else None
        if p and p.exists():
            return send_file(p, as_attachment=True, download_name=p.name)
        return "Not found", 404

    elif kind == "zip":
        result_files = [f for f in txt_files if "_PostEQ" not in f.name and "_report" not in f.name]
        report_files = [f for f in txt_files if "_report" in f.name]
        result  = result_files[0] if result_files else None
        report  = report_files[0] if report_files else None
        posteq  = posteq_files[0] if posteq_files else None

        if not result:
            return "No result found", 404

        zip_name = result.stem + "_AutoEQ.zip"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            if result and result.exists():
                zf.write(result,  result.name)
            if report and report.exists():
                zf.write(report,  report.name)
            if posteq and posteq.exists():
                zf.write(posteq,  posteq.name)
        buf.seek(0)
        return send_file(buf, as_attachment=True, download_name=zip_name,
                         mimetype="application/zip")

    return "Not found", 404


# ── launch ───────────────────────────────────────────────────────────────────

def _open_browser():
    import time
    time.sleep(1.1)
    webbrowser.open("http://localhost:5000")


if __name__ == "__main__":
    threading.Thread(target=_open_browser, daemon=True).start()
    print("\n  AutoEQ GUI running → http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
