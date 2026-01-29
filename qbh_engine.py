#!/usr/bin/env python3
# qbh_engine.py (Render-safe) — fingerprint-backed for index/query_extract
# IMPORTANT: matchmaking stays in Wix JSW (not here)

import os

# ============================================================
# 🔒 Guards BEFORE importing anything heavy
# ============================================================
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys, json, traceback, subprocess
from typing import Optional, List, Dict, Any

ENGINE_VERSION = "qbh_engine_v6_2026-01-29-fingerprint-backed"

HERE = os.path.dirname(os.path.abspath(__file__))
FINGERPRINT_PY = os.path.join(HERE, "fingerprint.py")

DEFAULT_WINDOW_SECONDS = 7.0
DEFAULT_HOP_SECONDS = 2.0
DEFAULT_MAX_WINDOWS = 140

DEFAULT_VOCAL_SECONDS = 8.0  # Recorder ~7s; fingerprint hum uses 8s internally

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
    try:
        return float(x)
    except Exception:
        return float(default)

def _safe_int(x, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)

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

def run_fingerprint(args: List[str]) -> Dict[str, Any]:
    if not os.path.exists(FINGERPRINT_PY):
        fail("fingerprint.py not found next to qbh_engine.py", extra={"fingerprint_path": FINGERPRINT_PY})

    cmd = [sys.executable, FINGERPRINT_PY] + args
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except Exception as e:
        fail("failed to spawn fingerprint.py", extra={"err": str(e), "cmd": cmd})

    out = (p.stdout or "").strip()

    if p.returncode != 0:
        fail(
            "fingerprint.py failed",
            extra={
                "returncode": p.returncode,
                "stderr_tail": (p.stderr or "")[-2000:],
                "stdout_tail": out[-2000:],
                "cmd": cmd,
            },
        )

    try:
        j = json.loads(out or "{}")
    except Exception:
        fail("fingerprint.py returned non-json", extra={"stdout_tail": out[-2000:], "stderr_tail": (p.stderr or "")[-2000:]})

    if isinstance(j, dict) and j.get("ok") is False:
        fail("fingerprint.py returned ok=false", extra={"fingerprint": j})

    return j

def build_qbh_query_from_fp_hum(fp: Dict[str, Any], *, max_seconds: float) -> Dict[str, Any]:
    match = fp.get("match") or {}
    melody = fp.get("melody") or {}
    meta = fp.get("meta") or {}

    return {
        "status": "ok",
        "mode": "query_extract",
        "version": ENGINE_VERSION,
        "audio": {
            "sr": meta.get("sr"),
            "duration_sec": meta.get("duration_sec"),
            "max_seconds": float(max_seconds),
        },
        "qbh": {
            "hum_hash": fp.get("hum_hash"),
            "match_hash": match.get("hash"),
            "melody_hash": melody.get("melody_hash"),
            "match": {
                "shape": match.get("shape"),
                "signature": match.get("signature"),
                "frames_per_sec": match.get("frames_per_sec"),
                "hop_length": match.get("hop_length"),
                "T": match.get("T"),
                "type": match.get("type"),
            },
            "melody": {
                "signature": melody.get("signature"),
                "voiced_ratio": melody.get("voiced_ratio"),
                "n_points": melody.get("n_points"),
                "hop_length": melody.get("hop_length"),
                "type": "melody_intervals_v1",
            },
        },
    }

def build_qbh_index_from_fp_index_full(fp: Dict[str, Any], *, window_sec: float, hop_sec: float, max_windows: int) -> Dict[str, Any]:
    meta = fp.get("meta") or {}
    windows = fp.get("windows") or []

    out_windows = []
    for w in windows:
        if not isinstance(w, dict):
            continue
        out_windows.append({
            "t0": _safe_float(w.get("t0"), 0.0),
            "window_sec": _safe_float(w.get("window_sec"), window_sec),
            "hum_hash": w.get("hum_hash"),
            "match_hash": w.get("match_hash"),
            "melody_hash": w.get("melody_hash"),
        })

    return {
        "status": "ok",
        "mode": "index_full",
        "version": ENGINE_VERSION,
        "audio": {
            "sr": meta.get("sr"),
            "duration_sec": meta.get("duration_sec"),
        },
        "index": {
            "window_sec": float(window_sec),
            "hop_sec": float(hop_sec),
            "max_windows": int(max_windows),
            "count": int(len(out_windows)),
            "windows": out_windows,
        },
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

        # ------------------------------------------------------------
        # index_full: for Wix imported tracks -> store into qbhIndex field
        # ------------------------------------------------------------
        if mode == "index_full":
            window_sec = _safe_float(payload.get("window_seconds", payload.get("window_sec", DEFAULT_WINDOW_SECONDS)), DEFAULT_WINDOW_SECONDS)
            hop_sec = _safe_float(payload.get("hop_seconds", payload.get("hop_sec", DEFAULT_HOP_SECONDS)), DEFAULT_HOP_SECONDS)
            max_windows = _safe_int(payload.get("max_windows", DEFAULT_MAX_WINDOWS), DEFAULT_MAX_WINDOWS)

            fp = run_fingerprint([
                "--mode", "index_full",
                "--window_seconds", str(window_sec),
                "--hop_seconds", str(hop_sec),
                "--max_windows", str(max_windows),
                audio_path,
            ])

            out = build_qbh_index_from_fp_index_full(fp, window_sec=window_sec, hop_sec=hop_sec, max_windows=max_windows)
            sys.stdout.write(json.dumps(out, ensure_ascii=False))
            sys.stdout.flush()
            return

        # ------------------------------------------------------------
        # query_extract: for Recorder voice (de-gamée / mauvais tempo)
        # ------------------------------------------------------------
        max_seconds = _safe_float(payload.get("max_seconds", DEFAULT_VOCAL_SECONDS), DEFAULT_VOCAL_SECONDS)

        fp = run_fingerprint([
            "--mode", "hum",
            audio_path,
        ])

        out = build_qbh_query_from_fp_hum(fp, max_seconds=max_seconds)
        sys.stdout.write(json.dumps(out, ensure_ascii=False))
        sys.stdout.flush()

    except Exception:
        log("❌ Python exception:\n" + traceback.format_exc())
        fail("python_exception", extra={"traceback": traceback.format_exc()})

if __name__ == "__main__":
    main()
