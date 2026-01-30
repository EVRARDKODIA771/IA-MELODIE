#!/usr/bin/env python3
# fingerprint.py — QBH feature extractor for BOTH:
#   - query_extract: (Recorder.jsx file, usually 7s) -> features+hashes
#   - index_full: (Wix imported full track) -> sliding windows 7s across track
#
# Output JSON only on stdout. Logs only on stderr.

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("LIBROSA_DISABLE_NUMBA", "1")

import sys, json, time, tempfile, subprocess, hashlib, traceback, base64
from typing import Optional, List, Dict, Any, Tuple

import numpy as np

ENGINE_VERSION = "fingerprint_qbh_v2_2026-01-30"

DEFAULT_SR = 22050
DEFAULT_MAX_SECONDS_QUERY = 7.0

DEFAULT_WINDOW_SECONDS = 7.0
DEFAULT_HOP_SECONDS = 2.0
DEFAULT_MAX_WINDOWS = 140

MELODY_FPS = 15   # ~15 frames/sec
CHROMA_FPS = 4    # ~4 frames/sec

# -------------------------------------------------
# stderr logger
# -------------------------------------------------
def log(msg: str):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()

def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def ok(payload: dict, code: int = 0) -> int:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))
    sys.stdout.flush()
    return code

def fail(msg: str, *, code: int = 1, extra: Optional[dict] = None) -> int:
    payload = {"ok": False, "error": msg, "version": ENGINE_VERSION}
    if extra:
        payload["extra"] = extra
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))
    sys.stdout.flush()
    return code

# -------------------------------------------------
# args parser (no argparse)
# -------------------------------------------------
def parse_args(argv: List[str]):
    mode = "query_extract"
    file_path = None
    sr = DEFAULT_SR
    max_seconds = DEFAULT_MAX_SECONDS_QUERY

    window_seconds = DEFAULT_WINDOW_SECONDS
    hop_seconds = DEFAULT_HOP_SECONDS
    max_windows = DEFAULT_MAX_WINDOWS

    i = 1
    while i < len(argv):
        a = argv[i]

        if a in ("--mode", "-m"):
            mode = (argv[i + 1] if i + 1 < len(argv) else "").strip()
            i += 2
            continue
        if a.startswith("--mode="):
            mode = a.split("=", 1)[1].strip()
            i += 1
            continue

        if a == "--sr":
            sr = int(argv[i + 1])
            i += 2
            continue

        if a in ("--max_seconds", "--max_sec"):
            max_seconds = float(argv[i + 1])
            i += 2
            continue

        if a in ("--window_seconds", "--window_sec"):
            window_seconds = float(argv[i + 1])
            i += 2
            continue

        if a in ("--hop_seconds", "--hop_sec"):
            hop_seconds = float(argv[i + 1])
            i += 2
            continue

        if a == "--max_windows":
            max_windows = int(argv[i + 1])
            i += 2
            continue

        if not a.startswith("-"):
            file_path = a
        i += 1

    mode = (mode or "").strip().lower().replace("-", "_")
    if not file_path:
        return None, None, None, None, None, None, None, "missing file path"
    if mode not in ("query_extract", "index_full"):
        return None, None, None, None, None, None, None, f"unsupported mode: {mode}"

    if sr < 8000 or sr > 48000:
        return None, None, None, None, None, None, None, "sr out of range"
    if max_seconds <= 0.5 or max_seconds > 30:
        return None, None, None, None, None, None, None, "max_seconds out of range"
    if window_seconds <= 0.5 or window_seconds > 30:
        return None, None, None, None, None, None, None, "window_seconds out of range"
    if hop_seconds < 0.25 or hop_seconds > window_seconds:
        return None, None, None, None, None, None, None, "hop_seconds out of range"
    if max_windows < 1 or max_windows > 500:
        return None, None, None, None, None, None, None, "max_windows out of range"

    return file_path, mode, int(sr), float(max_seconds), float(window_seconds), float(hop_seconds), int(max_windows), None

# -------------------------------------------------
# ffprobe / ffmpeg
# -------------------------------------------------
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

def load_wav_mono_float32(path: str, sr_expected: int, max_seconds: float) -> Tuple[np.ndarray, int]:
    y = None
    sr = sr_expected
    try:
        import soundfile as sf
        y, sr = sf.read(path, dtype="float32", always_2d=False)
        sr = int(sr) if sr else int(sr_expected)
        if y is None:
            y = np.zeros((0,), dtype=np.float32)
        if getattr(y, "ndim", 1) > 1:
            y = y[:, 0]
    except Exception:
        from scipy.io import wavfile
        sr, y = wavfile.read(path)
        sr = int(sr) if sr else int(sr_expected)
        if getattr(y, "ndim", 1) > 1:
            y = y[:, 0]
        if y.dtype.kind in ("i", "u"):
            y = y.astype(np.float32) / max(np.iinfo(y.dtype).max, 1)
        else:
            y = y.astype(np.float32, copy=False)

    if max_seconds is not None:
        n = int(float(max_seconds) * float(sr))
        if y.size > n:
            y = y[:n]

    if y.size:
        m = float(np.max(np.abs(y)))
        if m > 1e-8:
            y = y / m
    return y.astype(np.float32, copy=False), sr

# -------------------------------------------------
# base64 helpers (optional debug transport)
# -------------------------------------------------
def b64_encode_bytes(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")

def b64_encode_int8(arr: np.ndarray) -> str:
    arr = np.asarray(arr, dtype=np.int8)
    return b64_encode_bytes(arr.tobytes())

def b64_encode_uint8(arr: np.ndarray) -> str:
    arr = np.asarray(arr, dtype=np.uint8)
    return b64_encode_bytes(arr.tobytes())

# -------------------------------------------------
# FAST, DETERMINISTIC 64-bit "simhash-like" sketch
# (no huge random matrices; stable + cheap)
# -------------------------------------------------
def _mix64(x: np.uint64) -> np.uint64:
    x ^= x >> np.uint64(33)
    x *= np.uint64(0xff51afd7ed558ccd)
    x ^= x >> np.uint64(33)
    x *= np.uint64(0xc4ceb9fe1a85ec53)
    x ^= x >> np.uint64(33)
    return x

def sketch64_from_ints(v: np.ndarray, seed: int) -> str:
    """
    Deterministic 64-bit sketch:
    accumulate signed contributions per bit using hashed indices.
    """
    v = np.asarray(v).reshape(-1)
    if v.size == 0:
        return "0x0000000000000000"

    acc = np.zeros((64,), dtype=np.int64)
    s = np.uint64(seed)

    # limit cost: if too long, sample uniformly
    if v.size > 2048:
        idx = np.linspace(0, v.size - 1, num=2048).astype(np.int64)
        v = v[idx]

    for i, val in enumerate(v):
        # hash(index, seed) -> 64-bit
        h = _mix64(np.uint64(i) ^ s)
        # sign from value
        w = int(val)
        if w == 0:
            continue
        sign = 1 if w > 0 else -1
        # contribute per bit of h
        for b in range(64):
            if (h >> np.uint64(b)) & np.uint64(1):
                acc[b] += sign
            else:
                acc[b] -= sign

    out = np.uint64(0)
    for b in range(64):
        if acc[b] >= 0:
            out |= (np.uint64(1) << np.uint64(b))
    return "0x" + format(int(out), "016x")

def sketch64_from_floats(v: np.ndarray, seed: int) -> str:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    if v.size == 0:
        return "0x0000000000000000"
    # quantize to small ints
    q = np.rint(np.clip(v, -3.0, 3.0) * 20.0).astype(np.int16)
    return sketch64_from_ints(q, seed)

# -------------------------------------------------
# Feature extractors
# melody: piptrack -> cents -> delta-steps int8
# chroma: chroma_stft -> uint8 (T,12)
# -------------------------------------------------
def extract_melody_steps_int8(y: np.ndarray, sr: int, frames_per_sec=MELODY_FPS, fmin=80.0, fmax=1000.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    import librosa

    hop = int(sr / float(frames_per_sec))
    hop = max(256, min(hop, 2048))
    n_fft = 2048

    if y is None or y.size < int(0.25 * sr):
        steps = np.zeros((0,), dtype=np.int8)
        meta = {"type":"melody_delta_steps_int8","backend":"piptrack","T":0,"sr":int(sr),"hop":int(hop),"frames_per_sec":float(frames_per_sec),"voiced_ratio":0.0,"sha256":sha256_hex(b"")}
        return steps, meta

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, center=True))
    pitches, mags = librosa.piptrack(S=S, sr=sr, n_fft=n_fft, hop_length=hop, fmin=fmin, fmax=fmax)

    T = pitches.shape[1] if pitches.ndim == 2 else 0
    if T <= 0:
        steps = np.zeros((0,), dtype=np.int8)
        meta = {"type":"melody_delta_steps_int8","backend":"piptrack","T":0,"sr":int(sr),"hop":int(hop),"frames_per_sec":float(frames_per_sec),"voiced_ratio":0.0,"sha256":sha256_hex(b"")}
        return steps, meta

    idx = np.argmax(mags, axis=0)
    f0 = pitches[idx, np.arange(T)]
    f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)

    maxmag = mags[idx, np.arange(T)].astype(np.float32)
    thr = float(np.median(maxmag) * 0.15) if maxmag.size else 0.0
    voiced = ((f0 > 0.0) & (maxmag > thr)).astype(np.float32)
    voiced_ratio = float(np.mean(voiced)) if voiced.size else 0.0

    ref = 55.0
    cents = np.zeros_like(f0, dtype=np.float32)
    nz = (f0 > 0.0) & (voiced > 0.0)
    cents[nz] = 1200.0 * np.log2(f0[nz] / ref)

    # median smooth
    if cents.size >= 5:
        k = 5
        pad = k // 2
        cents_pad = np.pad(cents, (pad, pad), mode="edge")
        cents_sm = np.empty_like(cents)
        for i in range(cents.size):
            cents_sm[i] = np.median(cents_pad[i:i+k])
        cents = cents_sm

    # remove absolute pitch (center)
    if np.any(nz):
        med = np.median(cents[nz])
        cents = cents - float(med)

    dc = np.diff(cents, prepend=cents[:1])
    dc = np.clip(dc, -400.0, 400.0)

    steps = np.rint(dc / 50.0).astype(np.int32)
    steps = np.clip(steps, -127, 127).astype(np.int8)
    steps = (steps.astype(np.int16) * voiced.astype(np.int16)).astype(np.int8)

    sig_hash = sha256_hex(steps.tobytes())
    meta = {
        "type":"melody_delta_steps_int8",
        "backend":"piptrack",
        "sr":int(sr),
        "hop":int(hop),
        "frames_per_sec":float(frames_per_sec),
        "T":int(steps.shape[0]),
        "voiced_ratio":float(voiced_ratio),
        "sha256":sig_hash,
        "mag_thr":float(thr),
        "quant_cents_per_unit":50,
    }
    return steps, meta

def extract_chroma_uint8(y: np.ndarray, sr: int, frames_per_sec=CHROMA_FPS) -> Tuple[np.ndarray, Dict[str, Any]]:
    import librosa

    hop = int(sr / float(frames_per_sec))
    hop = max(512, min(hop, 4096))
    n_fft = 4096

    if y is None or y.size < int(0.25 * sr):
        chroma_q = np.zeros((0, 12), dtype=np.uint8)
        meta = {"type":"chroma_stft_uint8","backend":"stft","T":0,"sr":int(sr),"hop":int(hop),"frames_per_sec":float(frames_per_sec),"sha256":sha256_hex(b"")}
        return chroma_q, meta

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop, n_fft=n_fft).T
    chroma = chroma / (np.sum(chroma, axis=1, keepdims=True) + 1e-8)

    # small smooth
    if chroma.shape[0] >= 5:
        k = 5
        pad = k // 2
        chroma_pad = np.pad(chroma, ((pad, pad), (0, 0)), mode="edge")
        chroma_sm = np.empty_like(chroma)
        for t in range(chroma.shape[0]):
            chroma_sm[t] = chroma_pad[t:t+k].mean(axis=0)
        chroma = chroma_sm

    chroma_q = np.clip(np.rint(chroma * 255.0), 0, 255).astype(np.uint8)
    sig_hash = sha256_hex(chroma_q.tobytes())
    meta = {
        "type":"chroma_stft_uint8",
        "backend":"stft",
        "sr":int(sr),
        "hop":int(hop),
        "frames_per_sec":float(frames_per_sec),
        "T":int(chroma_q.shape[0]),
        "sha256":sig_hash,
    }
    return chroma_q, meta

def build_qbh_from_audio(y: np.ndarray, sr: int, *, include_b64: bool) -> Dict[str, Any]:
    mel_steps, mel_meta = extract_melody_steps_int8(y, sr)
    chr_q, chr_meta = extract_chroma_uint8(y, sr)

    mel_sha = mel_meta.get("sha256")
    chr_sha = chr_meta.get("sha256")

    combo = f"{mel_sha or ''}|{chr_sha or ''}".encode("utf-8", errors="ignore")
    key_sha = sha256_hex(combo) if (mel_sha or chr_sha) else None

    # cheap stable sketches
    mel_sim = sketch64_from_ints(mel_steps.astype(np.int16), seed=777)
    chr_sim = sketch64_from_floats((chr_q.astype(np.float32) / 255.0).reshape(-1), seed=888)

    melody_obj = {
        "sha256": mel_sha,
        "simhash64": mel_sim,
        "T": int(mel_steps.shape[0]),
        "meta": mel_meta,
    }
    chroma_obj = {
        "sha256": chr_sha,
        "simhash64": chr_sim,
        "shape": [int(chr_q.shape[0]), 12],
        "meta": chr_meta,
    }

    # only include b64 for query_extract (debug) to keep index small
    if include_b64:
        melody_obj["b64"] = b64_encode_int8(mel_steps)
        chroma_obj["b64"] = b64_encode_uint8(chr_q)

    return {
        "qbh_key_sha256": key_sha,
        "melody": melody_obj,
        "chroma": chroma_obj,
    }

def window_offsets(duration: Optional[float], window_sec: float, hop_sec: float, max_windows: int) -> List[float]:
    if duration is None or duration <= window_sec:
        return [0.0]
    n = int((duration - window_sec) / hop_sec) + 1
    if n <= max_windows:
        return [i * hop_sec for i in range(n)]
    idx = np.linspace(0, n - 1, num=max_windows).astype(int)
    idx = sorted(set(int(x) for x in idx))
    return [float(i * hop_sec) for i in idx]

# -------------------------------------------------
# modes
# -------------------------------------------------
def process_query_extract(input_path: str, *, sr: int, max_seconds: float) -> Dict[str, Any]:
    wav = None
    try:
        wav = convert_to_wav_segment(input_path, sr=sr, start_sec=0.0, max_seconds=max_seconds + 0.25)
        y, sr2 = load_wav_mono_float32(wav, sr_expected=sr, max_seconds=max_seconds)
        dur = float(y.size / max(float(sr2), 1.0))

        qbh = build_qbh_from_audio(y, sr2, include_b64=True)

        return {
            "ok": True,
            "mode": "query_extract",
            "version": ENGINE_VERSION,
            "audio": {"sr": int(sr2), "duration_sec": float(dur), "max_seconds": float(max_seconds)},
            "qbh": qbh,
        }
    finally:
        if wav:
            try:
                os.unlink(wav)
            except Exception:
                pass

def process_index_full(input_path: str, *, sr: int, window_sec: float, hop_sec: float, max_windows: int) -> Dict[str, Any]:
    dur = ffprobe_duration_seconds(input_path)
    offs = window_offsets(dur, window_sec, hop_sec, max_windows)

    windows = []
    t0 = time.time()

    for off in offs:
        wav = None
        try:
            wav = convert_to_wav_segment(input_path, sr=sr, start_sec=off, max_seconds=window_sec + 0.25)
            y, sr2 = load_wav_mono_float32(wav, sr_expected=sr, max_seconds=window_sec)
            if y.size < int(0.6 * sr2):
                continue

            qbh = build_qbh_from_audio(y, sr2, include_b64=False)

            windows.append({
                "t0": float(off),
                "window_sec": float(window_sec),
                "qbh_key_sha256": qbh.get("qbh_key_sha256"),
                "melody_sha256": qbh["melody"]["sha256"],
                "chroma_sha256": qbh["chroma"]["sha256"],
                "melody_simhash64": qbh["melody"]["simhash64"],
                "chroma_simhash64": qbh["chroma"]["simhash64"],
            })
        except Exception:
            continue
        finally:
            if wav:
                try:
                    os.unlink(wav)
                except Exception:
                    pass

    elapsed = time.time() - t0

    return {
        "ok": True,
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
        },
    }

def main():
    parsed = parse_args(sys.argv)
    if parsed[-1] is not None:
        return fail(parsed[-1], code=2)

    file_path, mode, sr, max_seconds, window_seconds, hop_seconds, max_windows, _ = parsed

    try:
        if mode == "query_extract":
            out = process_query_extract(file_path, sr=sr, max_seconds=max_seconds)
            return ok(out, 0)

        out = process_index_full(
            file_path,
            sr=sr,
            window_sec=window_seconds,
            hop_sec=hop_seconds,
            max_windows=max_windows,
        )
        return ok(out, 0)

    except Exception as e:
        log("❌ Exception:\n" + traceback.format_exc())
        return fail("python_exception", code=1, extra={"msg": str(e)})

if __name__ == "__main__":
    sys.exit(main())
