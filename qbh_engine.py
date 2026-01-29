#!/usr/bin/env python3
# qbh_engine.py / qbh.py (patched to avoid SIGSEGV on Render-like envs)

import os

# ============================================================
# 🔒 Anti-SIGSEGV guards (must be set BEFORE importing numpy/librosa)
# ============================================================
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")  # librosa can trigger numba paths in some envs

import sys, json, base64, traceback, tempfile, subprocess, hashlib
import numpy as np
from typing import Optional, List

DEFAULT_SR = 22050
DEFAULT_MAX_SECONDS = 12.0
MELODY_FPS = 15
CHROMA_FPS = 4

ENGINE_VERSION = "qbh_engine_v4_2026-01-29-safe"

INDEX_WINDOW_SECONDS = 7.0
INDEX_HOP_SECONDS = 1.0
INDEX_MAX_WINDOWS = 140

def log(msg: str):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()

def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def fail(msg: str, code=1):
    log("❌ " + msg)
    sys.stdout.write(json.dumps({"status": "error", "message": msg}))
    sys.stdout.flush()
    sys.exit(code)

def b64_encode_int8(arr: np.ndarray) -> str:
    arr = np.asarray(arr, dtype=np.int8)
    return base64.b64encode(arr.tobytes()).decode("ascii")

def b64_decode_int8(s: str, shape) -> np.ndarray:
    raw = base64.b64decode(s.encode("ascii"))
    arr = np.frombuffer(raw, dtype=np.int8)
    return arr.reshape(shape)

def b64_encode_uint8(arr: np.ndarray) -> str:
    arr = np.asarray(arr, dtype=np.uint8)
    return base64.b64encode(arr.tobytes()).decode("ascii")

def b64_decode_uint8(s: str, shape) -> np.ndarray:
    raw = base64.b64decode(s.encode("ascii"))
    arr = np.frombuffer(raw, dtype=np.uint8)
    return arr.reshape(shape)

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

def convert_to_wav(input_path: str, sr: int, max_seconds: float, start_sec: float = 0.0) -> str:
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

def load_audio(path: str, sr=DEFAULT_SR, mono=True, max_seconds=DEFAULT_MAX_SECONDS):
    import librosa
    y, sr = librosa.load(path, sr=sr, mono=mono)
    if max_seconds is not None:
        y = y[: int(float(max_seconds) * sr)]
    return y.astype(np.float32, copy=False), sr

# ============================================================
# ✅ Melody extraction: use piptrack (avoid librosa.yin -> numba/segfault risk)
# Output stays compatible: int8 delta steps with voiced mask
# ============================================================
def extract_melody_sig(y: np.ndarray, sr: int, *, frames_per_sec=MELODY_FPS, fmin=80.0, fmax=1000.0):
    import librosa

    hop = int(sr / float(frames_per_sec))
    hop = max(256, min(hop, 2048))
    n_fft = 2048

    # Magnitude spectrogram
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, center=True))
    pitches, mags = librosa.piptrack(S=S, sr=sr, n_fft=n_fft, hop_length=hop, fmin=fmin, fmax=fmax)

    # For each frame: pick pitch bin with max magnitude
    T = pitches.shape[1] if pitches.ndim == 2 else 0
    if T <= 0:
        steps = np.zeros((0,), dtype=np.int8)
        meta = {
            "type": "melody_delta_steps_int8",
            "sr": int(sr),
            "hop": int(hop),
            "frames_per_sec": float(frames_per_sec),
            "T": 0,
            "quant_cents_per_unit": 50,
            "backend": "piptrack",
            "voiced_ratio": 0.0,
            "sha256": sha256_hex(b""),
        }
        return b64_encode_int8(steps), [0], meta

    idx = np.argmax(mags, axis=0)               # (T,)
    f0 = pitches[idx, np.arange(T)]             # (T,)
    f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)

    # voiced decision: magnitude threshold relative to median of max-mags
    maxmag = mags[idx, np.arange(T)].astype(np.float32)
    thr = float(np.median(maxmag) * 0.15) if maxmag.size else 0.0
    voiced = ((f0 > 0.0) & (maxmag > thr)).astype(np.float32)
    voiced_ratio = float(np.mean(voiced)) if voiced.size else 0.0

    # cents representation (relative to ref)
    ref = 55.0
    cents = np.zeros_like(f0, dtype=np.float32)
    nz = (f0 > 0.0) & (voiced > 0.0)
    cents[nz] = 1200.0 * np.log2(f0[nz] / ref)

    # smooth cents a bit (median filter length 5)
    if cents.size >= 5:
        k = 5
        pad = k // 2
        cents_pad = np.pad(cents, (pad, pad), mode="edge")
        cents_sm = np.empty_like(cents)
        for i in range(cents.size):
            cents_sm[i] = np.median(cents_pad[i:i+k])
        cents = cents_sm

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
        "type": "melody_delta_steps_int8",
        "sr": int(sr),
        "hop": int(hop),
        "frames_per_sec": float(frames_per_sec),
        "T": int(steps.shape[0]),
        "quant_cents_per_unit": 50,
        "backend": "piptrack",
        "voiced_ratio": float(voiced_ratio),
        "sha256": sig_hash,
        "mag_thr": float(thr),
    }
    return b64_encode_int8(steps), [int(steps.shape[0])], meta

def extract_chroma_sig(y: np.ndarray, sr: int, *, frames_per_sec=CHROMA_FPS):
    import librosa

    hop = int(sr / float(frames_per_sec))
    hop = max(512, min(hop, 4096))
    n_fft = 4096

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop, n_fft=n_fft)
    chroma = chroma.T  # (T,12)
    chroma = chroma / (np.sum(chroma, axis=1, keepdims=True) + 1e-8)

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
        "type": "chroma_stft_uint8",
        "sr": int(sr),
        "hop": int(hop),
        "frames_per_sec": float(frames_per_sec),
        "T": int(chroma_q.shape[0]),
        "backend": "stft",
        "sha256": sig_hash,
    }
    return b64_encode_uint8(chroma_q), [int(chroma_q.shape[0]), 12], meta

def parse_candidates(raw):
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        try:
            return json.loads(s)
        except Exception:
            return []
    return []

def normalize_mode(mode_raw: str) -> str:
    m = (mode_raw or "").strip()
    if not m:
        return ""
    m = m.replace("-", "_").replace(" ", "_").lower()

    aliases = {
        "extract_query": "query_extract",
        "queryextract": "query_extract",
        "query_extract": "query_extract",
        "query_extraction": "query_extract",
        "extractquery": "query_extract",
        "index": "index",
        "query": "query",
        "index_full": "index_full",
        "full_index": "index_full",
    }
    return aliases.get(m, m)

def window_offsets(duration: Optional[float], window_sec: float, hop_sec: float, max_windows: int) -> List[float]:
    if duration is None or duration <= window_sec:
        return [0.0]
    n = int((duration - window_sec) / hop_sec) + 1
    if n <= max_windows:
        return [i * hop_sec for i in range(n)]
    idx = np.linspace(0, n - 1, num=max_windows).astype(int)
    idx = sorted(set(int(x) for x in idx))
    return [float(i * hop_sec) for i in idx]

def main():
    wav_tmp = None
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        mode = normalize_mode(payload.get("mode"))

        if mode not in ("index", "query", "query_extract", "index_full"):
            fail("mode must be one of: index | query | query_extract | index_full")

        audio_path = payload.get("audio_path")
        if not audio_path:
            fail("audio_path is required")

        sr = int(payload.get("sr", DEFAULT_SR))
        max_seconds = float(payload.get("max_seconds", DEFAULT_MAX_SECONDS))

        # ============================================================
        # index_full: windowed hashes across full track (for 7s anywhere)
        # ============================================================
        if mode == "index_full":
            window_sec = float(payload.get("window_seconds", INDEX_WINDOW_SECONDS))
            hop_sec = float(payload.get("hop_seconds", INDEX_HOP_SECONDS))
            max_w = int(payload.get("max_windows", INDEX_MAX_WINDOWS))

            dur = ffprobe_duration_seconds(audio_path)
            offs = window_offsets(dur, window_sec, hop_sec, max_w)

            windows = []
            tmp_files = []

            for off in offs:
                try:
                    wav = convert_to_wav(audio_path, sr=sr, max_seconds=window_sec + 0.25, start_sec=off)
                except Exception:
                    continue
                tmp_files.append(wav)

                y, sr2 = load_audio(wav, sr=sr, mono=True, max_seconds=window_sec)
                if y.size < int(0.6 * sr2):
                    continue

                # melody
                try:
                    mel_b64, mel_shape, mel_meta = extract_melody_sig(y, sr2)
                    mel_sha = mel_meta.get("sha256") if isinstance(mel_meta, dict) else None
                    voiced_ratio = float(mel_meta.get("voiced_ratio", 0.0)) if isinstance(mel_meta, dict) else 0.0
                except Exception:
                    mel_sha = None
                    voiced_ratio = 0.0

                # chroma
                try:
                    chr_b64, chr_shape, chr_meta = extract_chroma_sig(y, sr2)
                    chr_sha = chr_meta.get("sha256") if isinstance(chr_meta, dict) else None
                except Exception:
                    chr_sha = None

                combo = f"{mel_sha or ''}|{chr_sha or ''}".encode("utf-8", errors="ignore")
                win_hash = sha256_hex(combo) if (mel_sha or chr_sha) else None

                windows.append({
                    "t0": float(off),
                    "window_sec": float(window_sec),
                    "melody_sha256": mel_sha,
                    "chroma_sha256": chr_sha,
                    "voiced_ratio": float(voiced_ratio),
                    "win_hash": win_hash,
                })

            for p in tmp_files:
                try:
                    os.unlink(p)
                except Exception:
                    pass

            out = {
                "status": "ok",
                "mode": "index_full",
                "version": ENGINE_VERSION,
                "audio": {"sr": int(sr), "duration_sec": None if dur is None else float(dur)},
                "index": {
                    "window_sec": float(window_sec),
                    "hop_sec": float(hop_sec),
                    "max_windows": int(max_w),
                    "count": int(len(windows)),
                    "windows": windows,
                },
            }
            sys.stdout.write(json.dumps(out))
            sys.stdout.flush()
            return

        # ============================================================
        # index / query_extract / query : single segment path
        # ============================================================
        log(f"🎧 Preparing audio: {audio_path} (mode={mode})")
        wav_tmp = convert_to_wav(audio_path, sr=sr, max_seconds=max_seconds, start_sec=0.0)
        y, sr = load_audio(wav_tmp, sr=sr, mono=True, max_seconds=max_seconds)
        dur = float(len(y) / max(float(sr), 1.0))

        try:
            melody_b64, melody_shape, melody_meta = extract_melody_sig(y, sr)
        except Exception as e:
            melody_b64, melody_shape, melody_meta = "", [0], {
                "type": "melody_failed",
                "error": str(e),
                "sha256": None,
                "voiced_ratio": 0.0,
                "backend": "piptrack",
            }

        chroma_b64, chroma_shape, chroma_meta = extract_chroma_sig(y, sr)

        if mode in ("index", "query_extract"):
            out = {
                "status": "ok",
                "mode": mode,
                "version": ENGINE_VERSION,
                "audio": {"sr": int(sr), "duration_sec": float(dur), "max_seconds": float(max_seconds)},
                "melodySig": {"b64": melody_b64, "shape": melody_shape, "meta": melody_meta},
                "chromaSig": {"b64": chroma_b64, "shape": chroma_shape, "meta": chroma_meta},
                "qbh_hash": {
                    "melody_sha256": melody_meta.get("sha256") if isinstance(melody_meta, dict) else None,
                    "chroma_sha256": chroma_meta.get("sha256") if isinstance(chroma_meta, dict) else None,
                },
            }
            sys.stdout.write(json.dumps(out))
            sys.stdout.flush()
            return

        candidates = parse_candidates(payload.get("candidates", []))
        if not isinstance(candidates, list) or len(candidates) == 0:
            fail("query mode requires non-empty candidates list")

        # ============================================================
        # DTW matching (unchanged)
        # ============================================================
        def dtw_distance(A: np.ndarray, B: np.ndarray, *, window=None) -> float:
            A = np.asarray(A)
            B = np.asarray(B)
            if A.ndim == 1: A = A[:, None]
            if B.ndim == 1: B = B[:, None]
            TA, _ = A.shape
            TB, _ = B.shape
            if window is None:
                window = max(TA, TB)
            window = int(window)

            INF = 1e18
            dp = np.full((TA + 1, TB + 1), INF, dtype=np.float64)
            dp[0, 0] = 0.0

            for i in range(1, TA + 1):
                j0 = max(1, i - window)
                j1 = min(TB, i + window)
                ai = A[i - 1]
                for j in range(j0, j1 + 1):
                    bj = B[j - 1]
                    cost = float(np.sum(np.abs(ai - bj)))
                    dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

            cost = dp[TA, TB]
            norm = (TA + TB) / 2.0
            return float(cost / max(norm, 1.0))

        def rotate_chroma(chroma_T12: np.ndarray, shift: int) -> np.ndarray:
            return np.roll(chroma_T12, shift=shift, axis=1)

        def score_chroma(query_chroma: np.ndarray, track_chroma: np.ndarray):
            q = np.asarray(query_chroma, dtype=np.float32)
            t = np.asarray(track_chroma, dtype=np.float32)
            w = int(max(q.shape[0], t.shape[0]) * 0.10) + 5
            best = 1e18
            best_shift = 0
            for s in range(12):
                ts = rotate_chroma(t, s)
                d = dtw_distance(q, ts, window=w)
                if d < best:
                    best = d
                    best_shift = s
            return float(best), int(best_shift)

        def score_melody(query_steps: np.ndarray, track_steps: np.ndarray):
            q = np.asarray(query_steps, dtype=np.int16)
            t = np.asarray(track_steps, dtype=np.int16)
            w = int(max(q.shape[0], t.shape[0]) * 0.10) + 5
            return float(dtw_distance(q, t, window=w))

        q_mel = None
        if melody_b64 and melody_shape and melody_shape[0] > 0:
            q_mel = b64_decode_int8(melody_b64, (melody_shape[0],))

        q_chr = b64_decode_uint8(chroma_b64, (chroma_shape[0], 12)).astype(np.float32) / 255.0

        results = []
        for i, c in enumerate(candidates):
            if not isinstance(c, dict):
                continue

            cid = c.get("id", f"cand_{i}")
            tm = c.get("melodySig") or {}
            tc = c.get("chromaSig") or {}

            s_mel = None
            s_chr = None
            best_shift = None

            if q_mel is not None and isinstance(tm, dict) and tm.get("b64") and tm.get("shape") and tm["shape"][0] > 0:
                try:
                    t_mel = b64_decode_int8(tm["b64"], (tm["shape"][0],))
                    s_mel = score_melody(q_mel, t_mel)
                except Exception:
                    s_mel = None

            if isinstance(tc, dict) and tc.get("b64") and tc.get("shape") and tc["shape"][0] > 0:
                try:
                    t_chr = b64_decode_uint8(tc["b64"], (tc["shape"][0], 12)).astype(np.float32) / 255.0
                    s_chr, best_shift = score_chroma(q_chr, t_chr)
                except Exception:
                    s_chr = None
                    best_shift = None

            if s_mel is None and s_chr is None:
                continue

            fused = min([x for x in (s_mel, s_chr) if x is not None])

            results.append({
                "id": cid,
                "score": float(fused),
                "score_melody": None if s_mel is None else float(s_mel),
                "score_chroma": None if s_chr is None else float(s_chr),
                "chroma_best_shift": best_shift,
            })

        results.sort(key=lambda r: r["score"])
        top_k = int(payload.get("top_k", 10))

        out = {
            "status": "ok",
            "mode": "query",
            "version": ENGINE_VERSION,
            "audio": {"sr": int(sr), "duration_sec": float(dur), "max_seconds": float(max_seconds)},
            "count": int(len(results)),
            "top": results[:top_k],
        }
        sys.stdout.write(json.dumps(out))
        sys.stdout.flush()

    except Exception as e:
        log("❌ Python exception:\n" + traceback.format_exc())
        sys.stdout.write(json.dumps({"status": "error", "message": str(e)}))
        sys.stdout.flush()
        sys.exit(1)
    finally:
        if wav_tmp:
            try:
                os.unlink(wav_tmp)
            except Exception:
                pass

if __name__ == "__main__":
    main()
