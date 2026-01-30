#!/usr/bin/env python3
# qbh_engine.py — Router/wrapper around fingerprint.py
# Modes (stdin JSON):
#   - {"mode":"index_full","audio_path": "...", "window_seconds":7, "hop_seconds":2, "max_windows":140, "sr":22050}
#   - {"mode":"query_extract","audio_path":"...", "max_seconds":7, "sr":22050}
#
# Output JSON on stdout. Logs on stderr only.

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys, json, traceback, subprocess
from typing import Optional, Dict, Any, List

ENGINE_VERSION = "qbh_engine_router_v1_2026-01-30"

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
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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

    if not isinstance(j, dict) or j.get("ok") is not True:
        fail("fingerprint.py returned ok!=true", extra={"payload": j})

    return j

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
            fp = run_fingerprint([
                "--mode", "query_extract",
                "--sr", str(sr),
                "--max_seconds", str(max_seconds),
                audio_path,
            ])

            out = {
                "status": "ok",
                "mode": "query_extract",
                "version": ENGINE_VERSION,
                "audio": fp.get("audio", {"sr": sr, "max_seconds": max_seconds}),
                "qbh": fp.get("qbh", {}),
            }
            sys.stdout.write(json.dumps(out, ensure_ascii=False))
            sys.stdout.flush()
            return

        window_sec = _safe_float(payload.get("window_seconds", payload.get("window_sec", DEFAULT_WINDOW_SECONDS)), DEFAULT_WINDOW_SECONDS)
        hop_sec = _safe_float(payload.get("hop_seconds", payload.get("hop_sec", DEFAULT_HOP_SECONDS)), DEFAULT_HOP_SECONDS)
        max_windows = _safe_int(payload.get("max_windows", DEFAULT_MAX_WINDOWS), DEFAULT_MAX_WINDOWS)

        fp = run_fingerprint([
            "--mode", "index_full",
            "--sr", str(sr),
            "--window_seconds", str(window_sec),
            "--hop_seconds", str(hop_sec),
            "--max_windows", str(max_windows),
            audio_path,
        ])

        out = {
            "status": "ok",
            "mode": "index_full",
            "version": ENGINE_VERSION,
            "audio": fp.get("audio", {"sr": sr}),
            "index": fp.get("index", {}),
        }
        sys.stdout.write(json.dumps(out, ensure_ascii=False))
        sys.stdout.flush()

    except Exception:
        log("❌ Python exception:\n" + traceback.format_exc())
        fail("python_exception", extra={"traceback": traceback.format_exc()})

if __name__ == "__main__":
    main()
