#!/usr/bin/env python3
# qbh_engine.py
import sys, json, base64, traceback, os, tempfile, subprocess
import numpy as np

DEFAULT_SR = 22050
DEFAULT_MAX_SECONDS = 12.0
MELODY_FPS = 15
CHROMA_FPS = 4

def log(msg: str):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()

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

def convert_to_wav(input_path: str, sr: int, max_seconds: float) -> str:
    out_path = tempfile.mktemp(suffix=".wav")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-t", str(float(max_seconds)),
        "-ac", "1",
        "-ar", str(int(sr)),
        out_path,
    ]
    log(f"🎚️ ffmpeg -> wav: {' '.join(cmd)}")
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

def extract_melody_sig(y: np.ndarray, sr: int, *, frames_per_sec=MELODY_FPS, fmin=80.0, fmax=1000.0):
    """
    ✅ Stable: librosa.yin (pas de numba JIT)
    """
    import librosa

    hop = int(sr / frames_per_sec)
    hop = max(256, min(hop, 2048))
    log(f"   A) yin hop={hop} fps={frames_per_sec}")

    f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop)
    f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)

    voiced = (f0 > 0.0).astype(np.float32)

    ref = 55.0
    cents = np.zeros_like(f0, dtype=np.float32)
    nz = f0 > 0
    cents[nz] = 1200.0 * np.log2(f0[nz] / ref)

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
        cents = cents - med

    dc = np.diff(cents, prepend=cents[:1])
    dc = np.clip(dc, -400.0, 400.0)

    steps = np.rint(dc / 50.0).astype(np.int32)
    steps = np.clip(steps, -127, 127).astype(np.int8)

    steps = (steps.astype(np.int16) * voiced.astype(np.int16)).astype(np.int8)

    meta = {
        "type": "melody_delta_steps_int8",
        "sr": sr,
        "hop": hop,
        "frames_per_sec": frames_per_sec,
        "T": int(steps.shape[0]),
        "quant_cents_per_unit": 50,
        "backend": "yin",
    }
    return b64_encode_int8(steps), [int(steps.shape[0])], meta

def extract_chroma_sig(y: np.ndarray, sr: int, *, frames_per_sec=CHROMA_FPS):
    """
    ✅ Stable: chroma_stft (pas CQT)
    """
    import librosa

    hop = int(sr / frames_per_sec)
    hop = max(512, min(hop, 4096))
    n_fft = 4096
    log(f"   B) chroma_stft hop={hop} fps={frames_per_sec} n_fft={n_fft}")

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop, n_fft=n_fft)
    chroma = chroma.T  # (T, 12)

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

    meta = {
        "type": "chroma_stft_uint8",
        "sr": sr,
        "hop": hop,
        "frames_per_sec": frames_per_sec,
        "T": int(chroma_q.shape[0]),
        "backend": "stft",
    }
    return b64_encode_uint8(chroma_q), [int(chroma_q.shape[0]), 12], meta

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

def main():
    wav_tmp = None
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        mode = payload.get("mode", "")
        if mode not in ("index", "query", "query_extract"):
            fail("mode must be one of: index | query | query_extract")

        audio_path = payload.get("audio_path")
        if not audio_path:
            fail("audio_path is required")

        sr = int(payload.get("sr", DEFAULT_SR))
        max_seconds = float(payload.get("max_seconds", DEFAULT_MAX_SECONDS))

        log(f"🎧 Preparing audio: {audio_path}")
        wav_tmp = convert_to_wav(audio_path, sr=sr, max_seconds=max_seconds)

        log(f"🎧 Loading audio wav: {wav_tmp}")
        y, sr = load_audio(wav_tmp, sr=sr, mono=True, max_seconds=max_seconds)
        log(f"✅ Loaded: {len(y)/sr:.2f}s @ sr={sr}")

        log("🧠 Extracting Option A (melodySig)...")
        try:
            melody_b64, melody_shape, melody_meta = extract_melody_sig(y, sr)
        except Exception as e:
            log("⚠️ Melody extraction failed (fallback empty): " + str(e))
            melody_b64, melody_shape, melody_meta = "", [0], {"type": "melody_failed", "error": str(e)}

        log("🎼 Extracting Option B (chromaSig)...")
        chroma_b64, chroma_shape, chroma_meta = extract_chroma_sig(y, sr)

        if mode in ("index", "query_extract"):
            out = {
                "status": "ok",
                "mode": mode,
                "melodySig": {"b64": melody_b64, "shape": melody_shape, "meta": melody_meta},
                "chromaSig": {"b64": chroma_b64, "shape": chroma_shape, "meta": chroma_meta},
            }
            sys.stdout.write(json.dumps(out))
            sys.stdout.flush()
            return

        candidates = payload.get("candidates", [])
        if not isinstance(candidates, list) or len(candidates) == 0:
            fail("query mode requires non-empty candidates list")

        q_mel = None
        if melody_b64 and melody_shape and melody_shape[0] > 0:
            q_mel = b64_decode_int8(melody_b64, (melody_shape[0],))

        q_chr = b64_decode_uint8(chroma_b64, (chroma_shape[0], 12)).astype(np.float32) / 255.0

        log(f"🔎 Matching against {len(candidates)} candidates...")

        results = []
        for i, c in enumerate(candidates):
            cid = c.get("id", f"cand_{i}")

            tm = c.get("melodySig") or {}
            tc = c.get("chromaSig") or {}

            s_mel = None
            s_chr = None
            best_shift = None

            if q_mel is not None and tm.get("b64") and tm.get("shape") and tm["shape"][0] > 0:
                t_mel = b64_decode_int8(tm["b64"], (tm["shape"][0],))
                s_mel = score_melody(q_mel, t_mel)

            if tc.get("b64") and tc.get("shape"):
                t_chr = b64_decode_uint8(tc["b64"], (tc["shape"][0], 12)).astype(np.float32) / 255.0
                s_chr, best_shift = score_chroma(q_chr, t_chr)

            if s_mel is None and s_chr is None:
                continue

            fused = min([x for x in [s_mel, s_chr] if x is not None])

            results.append({
                "id": cid,
                "score": float(fused),
                "score_melody": None if s_mel is None else float(s_mel),
                "score_chroma": None if s_chr is None else float(s_chr),
                "chroma_best_shift": best_shift,
            })

            if (i + 1) % 10 == 0:
                log(f"… matched {i+1}/{len(candidates)}")

        results.sort(key=lambda r: r["score"])
        top_k = int(payload.get("top_k", 10))

        out = {"status": "ok", "mode": "query", "count": len(results), "top": results[:top_k]}
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
                log(f"🧹 Deleted temp wav: {wav_tmp}")
            except Exception:
                pass

if __name__ == "__main__":
    main()
