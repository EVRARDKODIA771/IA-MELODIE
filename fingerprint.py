#!/usr/bin/env python3
import sys
import json
import hashlib
import time
import traceback
import os
import tempfile
import subprocess
from typing import Optional, Dict, Any

# ============================================================
# CONFIG (Render-safe)
# ============================================================
TARGET_SR = 22050
HOP_LENGTH = 512

# full (legacy)
MAX_SECONDS_MAIN = 25.0
MAX_SECONDS_MATCH = 25.0
MAX_SECONDS_MELODY = 20.0

# hum (Recorder ~7s)
MAX_SECONDS_HUM = 8.0

# index_full (windowed across track)
DEFAULT_WINDOW_SECONDS = 7.0
DEFAULT_HOP_SECONDS = 2.0
DEFAULT_MAX_WINDOWS = 140

# ----------------------------
# Logging: stderr only
# ----------------------------
def log(msg: str):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()

def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _safe_float(x, default=0.0):
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)

# ----------------------------
# stdout JSON only helpers
# ----------------------------
def ok(payload: dict, code: int = 0):
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))
    sys.stdout.flush()
    return code

def fail(msg: str, *, code: int = 1, extra: Optional[Dict[str, Any]] = None):
    payload = {"ok": False, "error": msg}
    if extra:
        payload["extra"] = extra
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))
    sys.stdout.flush()
    return code

# ============================================================
# Args parsing (no argparse)
# - modes: full | hum | index_full
# - index_full accepts:
#   --window_seconds 7 --hop_seconds 2 --max_windows 140
# ============================================================
def parse_args(argv):
    mode = "full"
    file_path = None

    window_seconds = DEFAULT_WINDOW_SECONDS
    hop_seconds = DEFAULT_HOP_SECONDS
    max_windows = DEFAULT_MAX_WINDOWS

    i = 1
    while i < len(argv):
        a = argv[i]

        if a in ("--mode", "-m"):
            if i + 1 >= len(argv):
                return None, None, None, None, None, "missing value after --mode"
            mode = str(argv[i + 1]).strip().lower()
            i += 2
            continue
        if a.startswith("--mode="):
            mode = a.split("=", 1)[1].strip().lower()
            i += 1
            continue

        if a in ("--window_seconds", "--window_sec"):
            if i + 1 >= len(argv):
                return None, None, None, None, None, "missing value after --window_seconds"
            window_seconds = float(argv[i + 1])
            i += 2
            continue

        if a in ("--hop_seconds", "--hop_sec"):
            if i + 1 >= len(argv):
                return None, None, None, None, None, "missing value after --hop_seconds"
            hop_seconds = float(argv[i + 1])
            i += 2
            continue

        if a == "--max_windows":
            if i + 1 >= len(argv):
                return None, None, None, None, None, "missing value after --max_windows"
            max_windows = int(argv[i + 1])
            i += 2
            continue

        # positional
        if not a.startswith("-"):
            file_path = a
        i += 1

    if not file_path:
        return None, None, None, None, None, "missing filePath"
    if mode not in ("full", "hum", "index_full"):
        return None, None, None, None, None, f"unsupported mode: {mode}"

    # sanitize
    if not (0.5 <= window_seconds <= 30.0):
        return None, None, None, None, None, "window_seconds out of range (0.5..30)"
    if not (0.25 <= hop_seconds <= window_seconds):
        return None, None, None, None, None, "hop_seconds out of range (0.25..window_seconds)"
    if not (1 <= max_windows <= 500):
        return None, None, None, None, None, "max_windows out of range (1..500)"

    return file_path, mode, float(window_seconds), float(hop_seconds), int(max_windows), None

# ============================================================
# 0) Convert to WAV (cut early)
# ============================================================
def convert_to_wav(input_path: str, target_sr: int, max_seconds: float) -> Optional[str]:
    out_path = tempfile.mktemp(suffix=".wav")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-t", str(float(max_seconds)),
        "-ac", "1",
        "-ar", str(int(target_sr)),
        out_path,
    ]
    try:
        log(f"[fingerprint.py] ffmpeg convert -> wav: {' '.join(cmd)}")
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return out_path
        return None
    except Exception as e:
        log(f"[fingerprint.py] ffmpeg convert failed: {e}")
        try:
            if os.path.exists(out_path):
                os.unlink(out_path)
        except Exception:
            pass
        return None

# ============================================================
# index_full helpers: duration + segment convert
# ============================================================
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

def convert_to_wav_segment(input_path: str, target_sr: int, start_sec: float, max_seconds: float) -> Optional[str]:
    out_path = tempfile.mktemp(suffix=".wav")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", str(float(start_sec)),
        "-i", input_path,
        "-t", str(float(max_seconds)),
        "-ac", "1",
        "-ar", str(int(target_sr)),
        out_path,
    ]
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return out_path
        return None
    except Exception:
        try:
            if os.path.exists(out_path):
                os.unlink(out_path)
        except Exception:
            pass
        return None

def window_offsets(duration: Optional[float], window_sec: float, hop_sec: float, max_windows: int):
    if duration is None or duration <= window_sec:
        return [0.0]
    n = int((duration - window_sec) / hop_sec) + 1
    if n <= max_windows:
        return [i * hop_sec for i in range(n)]
    import numpy as np
    idx = np.linspace(0, n - 1, num=max_windows).astype(int)
    idx = sorted(set(int(x) for x in idx))
    return [float(i * hop_sec) for i in idx]

# ============================================================
# Helper: trim y to max_seconds
# ============================================================
def trim_audio(y, sr: int, max_seconds: float):
    if max_seconds and max_seconds > 0:
        max_len = int(sr * float(max_seconds))
        if y is not None and getattr(y, "size", 0) > max_len:
            y = y[:max_len]
    return y

# ============================================================
# 1) MATCH SIGNATURE (polyphonique): chroma sequence -> hash
# ============================================================
def build_chroma_match_signature_from_chroma(chroma_12T, sr, *, hop_length=HOP_LENGTH, frames_per_sec=4, max_seconds=MAX_SECONDS_MATCH):
    import numpy as np

    if chroma_12T is None or getattr(chroma_12T, "size", 0) == 0:
        return [], [0, 12], {"match_ok": False, "reason": "empty_chroma"}

    eps = 1e-9
    chroma = chroma_12T.astype(np.float32)

    chroma = chroma / (np.sum(chroma, axis=0, keepdims=True) + eps)

    fps_actual = sr / float(hop_length)
    target_fps = float(frames_per_sec) if frames_per_sec and frames_per_sec > 0 else 4.0
    step = int(max(1, round(fps_actual / target_fps)))

    chroma_ds = chroma[:, ::step]
    T = chroma_ds.shape[1]
    if T <= 0:
        return [], [0, 12], {"match_ok": False, "reason": "downsampled_empty"}

    q = np.clip(np.round(chroma_ds * 127.0), 0, 127).astype(np.uint8)

    q_T12 = q.T
    flat = q_T12.reshape(-1)

    match_hash = sha256_hex(flat.tobytes())
    meta = {
        "match_ok": True,
        "type": "chroma_cqt_seq_v1",
        "hop_length": int(hop_length),
        "frames_per_sec": float(target_fps),
        "max_seconds": float(max_seconds),
        "T": int(T),
        "quant_scale": 127.0,
        "hash": match_hash,
    }
    return flat.tolist(), [int(T), 12], meta

# ============================================================
# 2) MELODY SIGNATURE (monophonique): pitch intervals (pyin)
# ============================================================
def build_melody_signature(y, sr, *, hop_length=HOP_LENGTH, n_points=120, max_seconds=MAX_SECONDS_MELODY):
    import numpy as np
    import librosa

    y = trim_audio(y, sr, max_seconds)

    fmin = librosa.note_to_hz("C2")
    fmax = librosa.note_to_hz("C6")

    f0, _voiced_flag, _voiced_prob = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length
    )

    if f0 is None:
        return [], {"melody_ok": False, "reason": "pyin_returned_none"}

    midi = librosa.hz_to_midi(f0)
    voiced = (midi == midi)
    voiced_ratio = float(voiced.sum() / len(voiced)) if len(voiced) else 0.0

    if voiced_ratio < 0.08:
        return [], {"melody_ok": False, "reason": "too_few_voiced_frames", "voiced_ratio": voiced_ratio}

    midi_v = midi[voiced].astype(float)

    if len(midi_v) >= 7:
        k = 7
        pad = k // 2
        mv = [midi_v[0]] * pad + list(midi_v) + [midi_v[-1]] * pad
        sm = []
        for i in range(len(midi_v)):
            sm.append(sorted(mv[i:i+k])[k // 2])
        midi_v = sm

    midi_v = np.array(midi_v, dtype=np.float32)

    intervals = np.diff(midi_v)
    if intervals.size == 0:
        return [], {"melody_ok": False, "reason": "intervals_empty", "voiced_ratio": voiced_ratio}

    intervals = np.clip(intervals, -12.0, 12.0)
    q_int = np.round(intervals * 10.0).astype(np.int16)

    L = q_int.size
    if L >= n_points:
        idx = np.linspace(0, L - 1, num=n_points).astype(int)
        sig = q_int[idx]
    else:
        sig = np.zeros((n_points,), dtype=np.int16)
        sig[:L] = q_int

    melody_hash = sha256_hex(sig.tobytes())

    melody_meta = {
        "melody_ok": True,
        "voiced_ratio": voiced_ratio,
        "hop_length": int(hop_length),
        "n_points": int(n_points),
        "max_seconds": float(max_seconds),
        "quant_scale_semitone_x10": 10.0,
        "interval_clip_semitones": 12.0,
        "melody_hash": melody_hash,
    }
    return sig.tolist(), melody_meta

# ============================================================
# HUM CORE from (y,sr) -> hashes/signatures (used by hum + index_full)
# ============================================================
def hum_from_y(y, sr, *, window_sec: float):
    import numpy as np
    import librosa

    y = trim_audio(y, sr, window_sec)

    eps = 1e-9
    y = y.astype(np.float32)
    y = y / (np.max(np.abs(y)) + eps)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)

    match_sig, match_shape, match_meta = build_chroma_match_signature_from_chroma(
        chroma, sr, hop_length=HOP_LENGTH, frames_per_sec=4, max_seconds=window_sec
    )
    match_hash = match_meta.get("hash") if isinstance(match_meta, dict) else None

    melody_sig, melody_meta = build_melody_signature(
        y, sr, hop_length=HOP_LENGTH, n_points=120, max_seconds=min(MAX_SECONDS_MELODY, window_sec)
    )
    melody_hash = melody_meta.get("melody_hash") if isinstance(melody_meta, dict) else None

    combo_src = f"{match_hash or ''}|{melody_hash or ''}".encode("utf-8", errors="ignore")
    hum_hash = sha256_hex(combo_src) if (match_hash or melody_hash) else None

    return {
        "hum_hash": hum_hash,
        "match_hash": match_hash,
        "melody_hash": melody_hash,
        "match": {"shape": match_shape, "signature": match_sig, **match_meta},
        "melody": {"signature": melody_sig, **melody_meta},
    }

# ============================================================
# FULL MODE (unchanged behavior)
# ============================================================
def process_full(input_path: str, wav_tmp: Optional[str], *, t0: float):
    import numpy as np
    import librosa

    load_path = wav_tmp or input_path

    log("[fingerprint.py] Loading audio (full)...")
    y, sr = librosa.load(load_path, sr=TARGET_SR, mono=True)

    y = trim_audio(y, sr, MAX_SECONDS_MAIN)

    duration = float(librosa.get_duration(y=y, sr=sr))
    log(f"[fingerprint.py] Loaded+trim(full). sr={sr}, samples={len(y)}, duration={duration:.2f}s")

    eps = 1e-9
    y = y.astype(np.float32)
    y = y / (np.max(np.abs(y)) + eps)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=HOP_LENGTH)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)

    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    tempo = float(_safe_float(tempo, 0.0))

    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    onset_mean = float(np.mean(onset_env)) if getattr(onset_env, "size", 0) else 0.0
    onset_std  = float(np.std(onset_env)) if getattr(onset_env, "size", 0) else 0.0

    vec = np.concatenate([
        chroma_mean, chroma_std,
        mfcc_mean, mfcc_std,
        np.array([onset_mean, onset_std], dtype=np.float32)
    ]).astype(np.float32)

    q = np.round(vec * 1000.0).astype(np.int16)
    fingerprint = sha256_hex(q.tobytes())
    short_fp = fingerprint[:16]

    match_sig, match_shape, match_meta = build_chroma_match_signature_from_chroma(
        chroma, sr, hop_length=HOP_LENGTH, frames_per_sec=4, max_seconds=MAX_SECONDS_MATCH
    )

    melody_sig, melody_meta = build_melody_signature(
        y, sr, hop_length=HOP_LENGTH, n_points=120, max_seconds=MAX_SECONDS_MELODY
    )

    elapsed = time.time() - t0
    log(f"[fingerprint.py] Done(full) in {elapsed:.2f}s. fingerprint={short_fp}...")

    return {
        "ok": True,
        "mode": "full",
        "fingerprint": fingerprint,
        "fingerprint_short": short_fp,
        "meta": {
            "sr": int(sr),
            "duration_sec": float(duration),
            "tempo_bpm_est": float(tempo),
            "vector_dim": int(q.shape[0]),
            "quant_scale": 1000.0,
            "used_ffmpeg_wav": bool(wav_tmp),
            "max_seconds_main": float(MAX_SECONDS_MAIN),
        },
        "match": {"shape": match_shape, "signature": match_sig, **match_meta},
        "melody": {"signature": melody_sig, **melody_meta},
    }

# ============================================================
# HUM MODE (7-8s from start) for Recorder
# ============================================================
def process_hum(input_path: str, wav_tmp: Optional[str], *, t0: float):
    import librosa

    load_path = wav_tmp or input_path

    log("[fingerprint.py] Loading audio (hum)...")
    y, sr = librosa.load(load_path, sr=TARGET_SR, mono=True)
    y = trim_audio(y, sr, MAX_SECONDS_HUM)

    duration = float(len(y) / max(float(sr), 1.0))
    log(f"[fingerprint.py] Loaded+trim(hum). sr={sr}, samples={len(y)}, duration={duration:.2f}s")

    h = hum_from_y(y, sr, window_sec=MAX_SECONDS_HUM)

    elapsed = time.time() - t0
    log(f"[fingerprint.py] Done(hum) in {elapsed:.2f}s. hum_hash={(h['hum_hash'] or '')[:16]}...")

    return {
        "ok": True,
        "mode": "hum",
        "hum_hash": h["hum_hash"],
        "meta": {
            "sr": int(sr),
            "duration_sec": float(duration),
            "used_ffmpeg_wav": bool(wav_tmp),
            "max_seconds_hum": float(MAX_SECONDS_HUM),
        },
        "match": h["match"],
        "melody": h["melody"],
    }

# ============================================================
# INDEX_FULL MODE (windowed across entire track)
# - produces window hashes compatible with hum() later
# ============================================================
def process_index_full(input_path: str, *, t0: float, window_sec: float, hop_sec: float, max_windows: int):
    import librosa

    dur = ffprobe_duration_seconds(input_path)
    offs = window_offsets(dur, window_sec, hop_sec, max_windows)

    windows = []
    tmp_files = []

    log(f"[fingerprint.py] index_full: duration={dur} window_sec={window_sec} hop_sec={hop_sec} max_windows={max_windows} offs={len(offs)}")

    for off in offs:
        wav = convert_to_wav_segment(input_path, TARGET_SR, off, window_sec + 0.25)
        if not wav:
            continue
        tmp_files.append(wav)

        try:
            y, sr = librosa.load(wav, sr=TARGET_SR, mono=True)
            if y is None or len(y) < int(0.6 * sr):
                continue

            h = hum_from_y(y, sr, window_sec=window_sec)

            windows.append({
                "t0": float(off),
                "window_sec": float(window_sec),
                "hum_hash": h["hum_hash"],
                "match_hash": h.get("match_hash"),
                "melody_hash": h.get("melody_hash"),
            })
        except Exception:
            continue

    for p in tmp_files:
        try:
            os.unlink(p)
        except Exception:
            pass

    elapsed = time.time() - t0
    log(f"[fingerprint.py] Done(index_full) in {elapsed:.2f}s. windows={len(windows)}")

    return {
        "ok": True,
        "mode": "index_full",
        "meta": {
            "sr": int(TARGET_SR),
            "duration_sec": None if dur is None else float(dur),
            "window_seconds": float(window_sec),
            "hop_seconds": float(hop_sec),
            "max_windows": int(max_windows),
            "count": int(len(windows)),
            "elapsed_sec": float(elapsed),
        },
        "windows": windows,
    }

# ============================================================
# MAIN
# ============================================================
def main():
    file_path, mode, window_sec, hop_sec, max_windows, err = parse_args(sys.argv)
    if err:
        return fail(err, code=2)

    input_path = file_path
    t0 = time.time()
    log(f"[fingerprint.py] Start. mode={mode} file={input_path}")

    wav_tmp = None
    try:
        if mode == "index_full":
            out = process_index_full(
                input_path,
                t0=t0,
                window_sec=window_sec,
                hop_sec=hop_sec,
                max_windows=max_windows,
            )
            return ok(out, code=0)

        cut_seconds = MAX_SECONDS_MAIN if mode == "full" else MAX_SECONDS_HUM
        wav_tmp = convert_to_wav(input_path, target_sr=TARGET_SR, max_seconds=cut_seconds)
        if wav_tmp:
            log(f"[fingerprint.py] Using converted wav: {wav_tmp}")
        else:
            log("[fingerprint.py] Using original file (no conversion).")

        if mode == "full":
            out = process_full(input_path, wav_tmp, t0=t0)
        else:
            out = process_hum(input_path, wav_tmp, t0=t0)

        return ok(out, code=0)

    except Exception as e:
        log("[fingerprint.py] ERROR: " + str(e))
        log(traceback.format_exc())
        return fail(
            "python_exception",
            code=1,
            extra={"type": str(type(e)), "msg": str(e), "mode": mode}
        )
    finally:
        if wav_tmp:
            try:
                os.unlink(wav_tmp)
                log(f"[fingerprint.py] Deleted temp wav: {wav_tmp}")
            except Exception:
                pass

if __name__ == "__main__":
    sys.exit(main())
