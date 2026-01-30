#!/usr/bin/env python3
# qbh_engine.py — FULL engine for:
#   - index_full  (full track => windows 7s sliding)
#   - query_extract (recorder => 7s extract)
#
# Reads JSON on stdin, writes JSON on stdout.
# IMPORTANT: matchmaking stays in Wix JSW.

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys, json, subprocess, tempfile, traceback, time
from typing import Optional, Dict, Any, List

ENGINE_VERSION = "qbh_engine_full_v2_2026-01-30"

HERE = os.path.dirname(os.path.abspath(__file__))
FINGERPRINT_PY = os.path.join(HERE, "fingerprint.py")

DEFAULT_SR = 22050
DEFAULT_QUERY_SECONDS = 7.0

DEFAULT_WINDOW_SECONDS = 7.0
DEFAULT_HOP_SECONDS = 2.0
DEFAULT_MAX_WINDOWS = 140

def log(msg: str):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()

def fail(msg: str, code: int = 1, extra: Optional[dict] = None):
    payload = {"status": "error", "message": msg, "version": ENGINE_VERSION}
    if extra is not None:
        payload["extra"] = extra
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))
    sys.stdout.flush()
    sys.exit(code)

def _safe_float(x, default: float) -> float:
    try: return float(x)
    except Exception: return float(default)

def _safe_int(x, default: int) -> int:
    try: return int(x)
    except Exception: return int(default)

def normalize_mode(mode_raw: str) -> str:
    m = (mode_raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "index_full": "index_full",
        "full_index": "index_full",
        "index": "index_full",
        "query_extract": "query_extract",
        "extract_query": "query_extract",
        "query": "query_extract",
    }
    return aliases.get(m, m)

def ffprobe_duration_seconds(input_path: str) -> Optional[float]:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_path
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8", errors="ignore").strip()
        if not out:
            return None
        d = float(out)
        if d <= 0:
            return None
        return d
    except Exception:
        return None

def convert_to_wav_segment(input_path: str, sr: int, start_sec: float, max_seconds: float) -> str:
    out_path = tempfile.mktemp(suffix=".wav")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-threads", "1",
        "-ss", str(float(start_sec)),
        "-i", input_path,
        "-t", str(float(max_seconds)),
        "-vn", "-sn", "-dn",
        "-ac", "1",
        "-ar", str(int(sr)),
        out_path,
    ]
    subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError("ffmpeg produced empty wav")
    return out_path

def run_fingerprint_on_wav(wav_path: str) -> Dict[str, Any]:
    if not os.path.exists(FINGERPRINT_PY):
        fail("fingerprint.py not found next to qbh_engine.py", extra={"fingerprint_path": FINGERPRINT_PY})

    cmd = [sys.executable, FINGERPRINT_PY, wav_path]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = (p.stdout or "").strip()

    if p.returncode != 0:
        fail("fingerprint.py failed", extra={
            "returncode": p.returncode,
            "stderr_tail": (p.stderr or "")[-2000:],
            "stdout_tail": out[-2000:],
        })

    try:
        j = json.loads(out or "{}")
    except Exception:
        fail("fingerprint.py returned non-json", extra={"stdout_tail": out[-2000:], "stderr_tail": (p.stderr or "")[-2000:]})

    if not isinstance(j, dict) or j.get("ok") is not True:
        fail("fingerprint.py returned ok!=true", extra={"payload": j})

    return j

def window_offsets(duration: Optional[float], window_sec: float, hop_sec: float, max_windows: int) -> List[float]:
    if duration is None or duration <= window_sec:
        return [0.0]
    n = int((duration - window_sec) / hop_sec) + 1
    if n <= max_windows:
        return [i * hop_sec for i in range(n)]
    import numpy as np
    idx = np.linspace(0, n - 1, num=max_windows).astype(int)
    idx = sorted(set(int(x) for x in idx))
    return [float(i * hop_sec) for i in idx]

def build_query(audio_path: str, sr: int, max_seconds: float) -> Dict[str, Any]:
    wav = None
    try:
        wav = convert_to_wav_segment(audio_path, sr=sr, start_sec=0.0, max_seconds=max_seconds + 0.25)
        fp = run_fingerprint_on_wav(wav)

        qbh = {
            "melody_sha256": fp["melody"]["sha256"],
            "chroma_sha256": fp["chroma"]["sha256"],
            "melody_simhash64": fp["melody"]["simhash64"],
            "chroma_simhash64": fp["chroma"]["simhash64"],
            # optional, for debugging or later improved matching:
            "melody_b64": fp["melody"]["b64"],
            "chroma_b64": fp["chroma"]["b64"],
            "chroma_shape": fp["chroma"]["shape"],
            "meta": {
                "melody": fp["melody"]["meta"],
                "chroma": fp["chroma"]["meta"],
            }
        }

        return {
            "status": "ok",
            "mode": "query_extract",
            "version": ENGINE_VERSION,
            "audio": {"sr": int(fp.get("sr", sr)), "max_seconds": float(max_seconds)},
            "qbh": qbh
        }
    finally:
        if wav:
            try: os.unlink(wav)
            except Exception: pass

def build_index(audio_path: str, sr: int, window_sec: float, hop_sec: float, max_windows: int) -> Dict[str, Any]:
    dur = ffprobe_duration_seconds(audio_path)
    offs = window_offsets(dur, window_sec, hop_sec, max_windows)

    windows = []
    t0 = time.time()

    for off in offs:
        wav = None
        try:
            wav = convert_to_wav_segment(audio_path, sr=sr, start_sec=off, max_seconds=window_sec + 0.25)
            fp = run_fingerprint_on_wav(wav)

            windows.append({
                "t0": float(off),
                "window_sec": float(window_sec),
                "melody_sha256": fp["melody"]["sha256"],
                "chroma_sha256": fp["chroma"]["sha256"],
                "melody_simhash64": fp["melody"]["simhash64"],
                "chroma_simhash64": fp["chroma"]["simhash64"],
            })
        except Exception:
            continue
        finally:
            if wav:
                try: os.unlink(wav)
                except Exception: pass

    elapsed = time.time() - t0
    return {
        "status": "ok",
        "mode": "index_full",
        "version": ENGINE_VERSION,
        "audio": {"sr": int(sr), "duration_sec": None if dur is None else float(dur)},
        "index": {
            "window_sec": float(window_sec),
            "hop_sec": float(hop_sec),
            "max_windows": int(max_windows),
            "count": int(len(windows)),
            "elapsed_sec": float(elapsed),
            "windows": windows,
        }
    }

def main():
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        mode = normalize_mode(payload.get("mode"))

        if mode not in ("index_full", "query_extract"):
            fail("mode must be one of: index_full | query_extract")

        audio_path = payload.get("audio_path")
        if not audio_path:
            fail("audio_path is required")
        sr = _safe_int(payload.get("sr", DEFAULT_SR), DEFAULT_SR)

        if mode == "query_extract":
            max_seconds = _safe_float(payload.get("max_seconds", DEFAULT_QUERY_SECONDS), DEFAULT_QUERY_SECONDS)
            out = build_query(audio_path, sr=sr, max_seconds=max_seconds)
            sys.stdout.write(json.dumps(out, ensure_ascii=False))
            sys.stdout.flush()
            return

        window_sec = _safe_float(payload.get("window_seconds", payload.get("window_sec", DEFAULT_WINDOW_SECONDS)), DEFAULT_WINDOW_SECONDS)
        hop_sec = _safe_float(payload.get("hop_seconds", payload.get("hop_sec", DEFAULT_HOP_SECONDS)), DEFAULT_HOP_SECONDS)
        max_windows = _safe_int(payload.get("max_windows", DEFAULT_MAX_WINDOWS), DEFAULT_MAX_WINDOWS)

        out = build_index(audio_path, sr=sr, window_sec=window_sec, hop_sec=hop_sec, max_windows=max_windows)
        sys.stdout.write(json.dumps(out, ensure_ascii=False))
        sys.stdout.flush()

    except Exception:
        log("❌ Python exception:\n" + traceback.format_exc())
        fail("python_exception", extra={"traceback": traceback.format_exc()})

if __name__ == "__main__":
    main()
