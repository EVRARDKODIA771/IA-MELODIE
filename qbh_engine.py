#!/usr/bin/env python3
# qbh_engine.py — standalone QBH engine (NO dependency on fingerprint.py)
# Modes (stdin JSON):
#   - {"mode":"index_full","audio_path":"...", "window_seconds":7, "hop_seconds":1, "max_windows":400, "sr":22050}
#   - {"mode":"query_extract","audio_path":"...", "max_seconds":7, "sr":22050}
#
# Output JSON on stdout. Logs on stderr only.

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("LIBROSA_DISABLE_NUMBA", "1")

import sys, json, traceback, subprocess, tempfile, time, hashlib, base64
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

ENGINE_VERSION = "qbh_engine_standalone_v1_2026-01-30"

DEFAULT_SR = 22050
DEFAULT_QUERY_SECONDS = 7.0
DEFAULT_WINDOW_SECONDS = 7.0
DEFAULT_HOP_SECONDS = 1.0          # IMPORTANT: 1s hop improves “anywhere in track”
DEFAULT_MAX_WINDOWS = 400          # allow more windows, but you can cap in Wix

MELODY_FPS = 15
CHROMA_FPS = 4


# -----------------------------
# stderr logger
# -----------------------------
def log(msg: str):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()

def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

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

# -----------------------------
# ffprobe / ffmpeg
# -----------------------------
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
    # soundfile is fastest; fallback to scipy
    try:
        import soundfile as sf
        y, sr = sf.read(path, dtype="float32", always_2d=False)
        if y is None:
            return np.zeros((0,), dtype=np.float32), sr_expected
        if getattr(y, "ndim", 1) > 1:
            y = y[:, 0]
        sr = int(sr) if sr else int(sr_expected)
    except Exception:
        from scipy.io import wavfile
        sr, y = wavfile.read(path)
        if y.ndim > 1:
            y = y[:, 0]
        if y.dtype.kind in ("i", "u"):
            y = y.astype(np.float32) / max(np.iinfo(y.dtype).max, 1)
        else:
            y = y.astype(np.float32, copy=False)
        sr = int(sr)

    # trim
    n = int(float(max_seconds) * float(sr))
    if y.size > n:
        y = y[:n]

    # normalize
    if y.size:
        m = float(np.max(np.abs(y)))
        if m > 1e-8:
            y = y / m

    return y.astype(np.float32, copy=False), sr

def b64_encode_bytes(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")

def b64_encode_int8(arr: np.ndarray) -> str:
    arr = np.asarray(arr, dtype=np.int8)
    return b64_encode_bytes(arr.tobytes())

def b64_encode_uint8(arr: np.ndarray) -> str:
    arr = np.asarray(arr, dtype=np.uint8)
    return b64_encode_bytes(arr.tobytes())

# -----------------------------
# Feature extractors
# -----------------------------
def extract_melody_steps_int8(y: np.ndarray, sr: int, frames_per_sec=MELODY_FPS, fmin=80.0, fmax=1000.0):
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

    # smooth
    if cents.size >= 5:
        k = 5
        pad = k // 2
        cents_pad = np.pad(cents, (pad, pad), mode="edge")
        cents_sm = np.empty_like(cents)
        for i in range(cents.size):
            cents_sm[i] = np.median(cents_pad[i:i+k])
        cents = cents_sm

    # center
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

def extract_chroma_uint8(y: np.ndarray, sr: int, frames_per_sec=CHROMA_FPS):
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

    # smooth
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

# -----------------------------
# SimHash 64-bit (tolerant)
# -----------------------------
def _resample_1d_to(x: np.ndarray, L: int) -> np.ndarray:
    x = np.asarray(x)
    if x.size == 0:
        return np.zeros((L,), dtype=np.float32)
    if x.size == L:
        return x.astype(np.float32)
    xp = np.linspace(0.0, 1.0, num=x.size, dtype=np.float32)
    xq = np.linspace(0.0, 1.0, num=L, dtype=np.float32)
    y = np.interp(xq, xp, x.astype(np.float32))
    return y.astype(np.float32)

def _resample_2d_time_to(X: np.ndarray, T: int) -> np.ndarray:
    X = np.asarray(X)
    if X.shape[0] == 0:
        return np.zeros((T, X.shape[1]), dtype=np.float32)
    if X.shape[0] == T:
        return X.astype(np.float32)
    xp = np.linspace(0.0, 1.0, num=X.shape[0], dtype=np.float32)
    xq = np.linspace(0.0, 1.0, num=T, dtype=np.float32)
    out = np.zeros((T, X.shape[1]), dtype=np.float32)
    for j in range(X.shape[1]):
        out[:, j] = np.interp(xq, xp, X[:, j].astype(np.float32))
    return out

def simhash64(vec: np.ndarray, seed: int) -> str:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if v.size == 0:
        return "0x0000000000000000"
    rng = np.random.default_rng(seed)
    H = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(64, v.size), replace=True)
    s = H @ v
    bits = (s >= 0).astype(np.uint64)
    out = np.uint64(0)
    for i in range(64):
        out |= (bits[i] << np.uint64(i))
    return "0x" + format(int(out), "016x")

def build_qbh(y: np.ndarray, sr: int) -> Dict[str, Any]:
    mel_steps, mel_meta = extract_melody_steps_int8(y, sr)
    chr_q, chr_meta = extract_chroma_uint8(y, sr)

    mel_sha = mel_meta.get("sha256")
    chr_sha = chr_meta.get("sha256")
    key_sha = sha256_hex(f"{mel_sha or ''}|{chr_sha or ''}".encode("utf-8")) if (mel_sha or chr_sha) else None

    mel_fixed = _resample_1d_to(mel_steps.astype(np.float32), 128)
    chr_fixed = _resample_2d_time_to(chr_q.astype(np.float32) / 255.0, 48).reshape(-1)

    return {
        "qbh_key_sha256": key_sha,
        "melody": {
            "sha256": mel_sha,
            "simhash64": simhash64(mel_fixed, seed=777),
            "b64": b64_encode_int8(mel_steps),
            "T": int(mel_steps.shape[0]),
            "meta": mel_meta,
        },
        "chroma": {
            "sha256": chr_sha,
            "simhash64": simhash64(chr_fixed, seed=888),
            "b64": b64_encode_uint8(chr_q),
            "shape": [int(chr_q.shape[0]), 12],
            "meta": chr_meta,
        },
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

def query_extract(audio_path: str, sr: int, max_seconds: float) -> Dict[str, Any]:
    wav = None
    try:
        wav = convert_to_wav_segment(audio_path, sr=sr, start_sec=0.0, max_seconds=max_seconds + 0.25)
        y, sr2 = load_wav_mono_float32(wav, sr_expected=sr, max_seconds=max_seconds)
        dur = float(y.size / max(float(sr2), 1.0))
        qbh = build_qbh(y, sr2)
        return {
            "status": "ok",
            "mode": "query_extract",
            "version": ENGINE_VERSION,
            "audio": {"sr": int(sr2), "duration_sec": dur, "max_seconds": float(max_seconds)},
            "qbh": qbh,
        }
    finally:
        if wav:
            try: os.unlink(wav)
            except Exception: pass

def index_full(audio_path: str, sr: int, window_sec: float, hop_sec: float, max_windows: int) -> Dict[str, Any]:
    dur = ffprobe_duration_seconds(audio_path)
    offs = window_offsets(dur, window_sec, hop_sec, max_windows)

    windows = []
    tstart = time.time()

    debug = {
        "duration_sec": None if dur is None else float(dur),
        "offsets": int(len(offs)),
        "tried": 0, "ok": 0, "ffmpeg_fail": 0, "short_audio": 0, "extract_fail": 0,
        "error_samples": [],
    }

    for off in offs:
        debug["tried"] += 1
        wav = None
        try:
            wav = convert_to_wav_segment(audio_path, sr=sr, start_sec=off, max_seconds=window_sec + 0.25)
        except Exception as e:
            debug["ffmpeg_fail"] += 1
            if len(debug["error_samples"]) < 5:
                debug["error_samples"].append({"t0": float(off), "step": "ffmpeg", "err": str(e)[:400]})
            continue

        try:
            y, sr2 = load_wav_mono_float32(wav, sr_expected=sr, max_seconds=window_sec)
            if y.size < int(0.6 * sr2):
                debug["short_audio"] += 1
                continue

            qbh = build_qbh(y, sr2)
            debug["ok"] += 1

            windows.append({
                "t0": float(off),
                "window_sec": float(window_sec),
                "qbh_key_sha256": qbh.get("qbh_key_sha256"),
                "melody_sha256": qbh["melody"]["sha256"],
                "chroma_sha256": qbh["chroma"]["sha256"],
                "melody_simhash64": qbh["melody"]["simhash64"],
                "chroma_simhash64": qbh["chroma"]["simhash64"],
            })

        except Exception as e:
            debug["extract_fail"] += 1
            if len(debug["error_samples"]) < 5:
                debug["error_samples"].append({"t0": float(off), "step": "extract", "err": str(e)[:400]})
            continue
        finally:
            if wav:
                try: os.unlink(wav)
                except Exception: pass

    elapsed = time.time() - tstart
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
        },
        "debug": debug,
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
            out = query_extract(audio_path, sr=sr, max_seconds=max_seconds)
            sys.stdout.write(json.dumps(out, ensure_ascii=False))
            sys.stdout.flush()
            return

        window_sec = _safe_float(payload.get("window_seconds", payload.get("window_sec", DEFAULT_WINDOW_SECONDS)), DEFAULT_WINDOW_SECONDS)
        hop_sec = _safe_float(payload.get("hop_seconds", payload.get("hop_sec", DEFAULT_HOP_SECONDS)), DEFAULT_HOP_SECONDS)
        max_windows = _safe_int(payload.get("max_windows", DEFAULT_MAX_WINDOWS), DEFAULT_MAX_WINDOWS)

        out = index_full(audio_path, sr=sr, window_sec=window_sec, hop_sec=hop_sec, max_windows=max_windows)
        sys.stdout.write(json.dumps(out, ensure_ascii=False))
        sys.stdout.flush()

    except Exception:
        log("❌ Python exception:\n" + traceback.format_exc())
        fail("python_exception", extra={"traceback": traceback.format_exc()})

if __name__ == "__main__":
    main()
