#!/usr/bin/env python3
# qbh_engine.py
import sys, json, base64, traceback, math, time
import numpy as np

# ----------------------------
# Streaming logs (Node suit stderr)
# ----------------------------
def log(msg: str):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()

def die(msg: str, code=1):
    log("❌ " + msg)
    sys.exit(code)

# ----------------------------
# Helpers: encode/decode signatures (compact for Wix DB)
# ----------------------------
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

# ----------------------------
# Audio loading
# ----------------------------
def load_audio(path: str, sr=22050, mono=True, max_seconds=30.0):
    import librosa
    y, sr = librosa.load(path, sr=sr, mono=mono)
    if max_seconds is not None:
        y = y[: int(max_seconds * sr)]
    return y, sr

# ----------------------------
# OPTION B (chorale/polyphonique): Chroma sequence (CQT-based)
# ----------------------------
def extract_chroma_sig(y: np.ndarray, sr: int, hop_length=512, frames_per_sec=4, max_seconds=30.0):
    """
    Returns:
      chroma_sig_b64: base64(uint8) for T x 12
      shape: [T, 12]
      meta
    """
    import librosa

    # downsample time resolution to frames_per_sec
    # hop_length target: sr/frames_per_sec  (approx)
    hop = int(sr / frames_per_sec)
    hop = max(128, min(hop, 4096))

    # Use CQT chroma (more stable musically)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    # chroma shape: (12, T)
    chroma = chroma.T  # (T,12)

    # normalize per frame (robust to loudness)
    chroma = chroma / (np.sum(chroma, axis=1, keepdims=True) + 1e-8)

    # smoothing (helps noise/chorale)
    # simple moving average over time
    if chroma.shape[0] >= 5:
        k = 5
        pad = k // 2
        chroma_pad = np.pad(chroma, ((pad, pad), (0, 0)), mode="edge")
        chroma_sm = np.empty_like(chroma)
        for t in range(chroma.shape[0]):
            chroma_sm[t] = chroma_pad[t:t+k].mean(axis=0)
        chroma = chroma_sm

    # quantize to uint8 (0..255)
    chroma_q = np.clip(np.rint(chroma * 255.0), 0, 255).astype(np.uint8)

    meta = {
        "sr": sr,
        "hop": hop,
        "frames_per_sec": frames_per_sec,
        "type": "chroma_cqt_uint8",
        "T": int(chroma_q.shape[0]),
    }
    return b64_encode_uint8(chroma_q), [int(chroma_q.shape[0]), 12], meta

# ----------------------------
# OPTION A (fredonnement/voix): Melody contour (relative, tolerant)
# ----------------------------
def extract_melody_sig(y: np.ndarray, sr: int, frames_per_sec=20, fmin=80.0, fmax=1000.0, max_seconds=30.0):
    """
    Robust for humming: extract pitch with pyin, convert to relative contour in cents,
    then to delta steps (in semitone-ish units), quantized int8.
    Returns:
      melody_sig_b64: base64(int8) for length T
      shape: [T]
      meta
    """
    import librosa

    hop = int(sr / frames_per_sec)
    hop = max(128, min(hop, 2048))

    # pyin returns f0 per frame (Hz) or nan if unvoiced
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop
    )
    # replace nan with 0
    f0 = np.nan_to_num(f0, nan=0.0)

    # voiced mask
    voiced = (f0 > 0.0).astype(np.float32)

    # Convert Hz to cents (log scale): cents = 1200*log2(f0/ref)
    # Use ref=55Hz to keep numbers moderate, then center later.
    ref = 55.0
    cents = np.zeros_like(f0, dtype=np.float32)
    nz = f0 > 0
    cents[nz] = 1200.0 * np.log2(f0[nz] / ref)

    # Smooth cents (reduce pitch bends/vibrato jitter)
    if cents.size >= 5:
        k = 5
        pad = k // 2
        cents_pad = np.pad(cents, (pad, pad), mode="edge")
        cents_sm = np.empty_like(cents)
        for i in range(cents.size):
            cents_sm[i] = np.median(cents_pad[i:i+k])
        cents = cents_sm

    # Make invariant to absolute pitch (voice grave/aigue):
    # subtract median of voiced cents
    if np.any(nz):
        med = np.median(cents[nz])
        cents = cents - med

    # Convert to delta contour (tempo invariant handled by DTW later)
    # delta in "semitone steps": 100 cents ~ 1 semitone
    dc = np.diff(cents, prepend=cents[:1])
    # clamp huge jumps (noise)
    dc = np.clip(dc, -400.0, 400.0)

    # quantize to int8: scale 100 cents -> 1 unit
    steps = np.rint(dc / 50.0).astype(np.int32)  # 50 cents per unit (tolerant to bends)
    steps = np.clip(steps, -127, 127).astype(np.int8)

    # Optionally damp unvoiced frames (set to 0)
    steps = (steps.astype(np.int16) * voiced.astype(np.int16)).astype(np.int8)

    meta = {
        "sr": sr,
        "hop": hop,
        "frames_per_sec": frames_per_sec,
        "type": "melody_delta_steps_int8",
        "T": int(steps.shape[0]),
        "quant_cents_per_unit": 50,
    }
    return b64_encode_int8(steps), [int(steps.shape[0])], meta

# ----------------------------
# DTW (simple, robust) for matching
# ----------------------------
def dtw_distance(seqA: np.ndarray, seqB: np.ndarray, window=None):
    """
    seqA: (TA, D) or (TA,)
    seqB: (TB, D) or (TB,)
    returns normalized cost
    """
    A = np.asarray(seqA)
    B = np.asarray(seqB)
    if A.ndim == 1:
        A = A[:, None]
    if B.ndim == 1:
        B = B[:, None]

    TA, D = A.shape
    TB, _ = B.shape

    # Sakoe-Chiba band window
    if window is None:
        window = max(TA, TB)  # no constraint
    window = int(window)

    INF = 1e18
    dp = np.full((TA + 1, TB + 1), INF, dtype=np.float64)
    dp[0, 0] = 0.0

    for i in range(1, TA + 1):
        j_start = max(1, i - window)
        j_end = min(TB, i + window)
        ai = A[i - 1]
        for j in range(j_start, j_end + 1):
            bj = B[j - 1]
            # L1 distance (more robust than L2)
            cost = np.sum(np.abs(ai - bj))
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    # normalize by path length approx
    cost = dp[TA, TB]
    norm = (TA + TB) / 2.0
    return float(cost / max(norm, 1.0))

def rotate_chroma(chroma_T12: np.ndarray, shift: int):
    # chroma_T12: (T,12)
    return np.roll(chroma_T12, shift=shift, axis=1)

def score_chroma(query_chroma: np.ndarray, track_chroma: np.ndarray):
    """
    Invariant to transposition: min over 12 rotations.
    DTW handles tempo.
    """
    best = 1e18
    best_shift = 0
    # Use window to speed: 10% band
    w = int(max(query_chroma.shape[0], track_chroma.shape[0]) * 0.10) + 5
    for s in range(12):
        tc = rotate_chroma(track_chroma, s)
        d = dtw_distance(query_chroma, tc, window=w)
        if d < best:
            best = d
            best_shift = s
    return best, best_shift

def score_melody(query_steps: np.ndarray, track_steps: np.ndarray):
    """
    Both are int8 sequences of delta steps (already pitch-invariant).
    DTW handles tempo; L1 distance handles bends.
    """
    w = int(max(query_steps.shape[0], track_steps.shape[0]) * 0.10) + 5
    return dtw_distance(query_steps.astype(np.int16), track_steps.astype(np.int16), window=w)

# ----------------------------
# Main
# ----------------------------
def main():
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        mode = payload.get("mode", "")
        if mode not in ("index", "query", "query_extract"):
            die("mode must be one of: index | query | query_extract")

        audio_path = payload.get("audio_path")
        if not audio_path:
            die("audio_path is required")

        max_seconds = float(payload.get("max_seconds", 30.0))
        sr = int(payload.get("sr", 22050))

        log(f"🎧 Loading audio: {audio_path}")
        y, sr = load_audio(audio_path, sr=sr, mono=True, max_seconds=max_seconds)
        log(f"✅ Loaded: {len(y)/sr:.2f}s @ sr={sr}")

        # Extract both signatures (A+B) always
        log("🧠 Extracting Option A (melodySig)...")
        melody_b64, melody_shape, melody_meta = extract_melody_sig(y, sr)

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

        # mode == query
        # candidates: list of {id, melodySig{b64,shape}, chromaSig{b64,shape}}
        candidates = payload.get("candidates", [])
        if not isinstance(candidates, list) or len(candidates) == 0:
            die("query mode requires non-empty candidates list")

        # Decode query
        q_mel = b64_decode_int8(melody_b64, (melody_shape[0],))
        q_chr = b64_decode_uint8(chroma_b64, (chroma_shape[0], 12)).astype(np.float32) / 255.0

        log(f"🔎 Matching against {len(candidates)} candidates...")

        results = []
        for idx, c in enumerate(candidates):
            cid = c.get("id", f"cand_{idx}")

            # Decode track chroma
            tc_info = c.get("chromaSig") or {}
            tm_info = c.get("melodySig") or {}

            # Skip if missing both
            if not tc_info.get("b64") and not tm_info.get("b64"):
                continue

            # Scores default large
            s_mel = None
            s_chr = None
            best_shift = None

            if tm_info.get("b64"):
                t_mel = b64_decode_int8(tm_info["b64"], (tm_info["shape"][0],))
                s_mel = score_melody(q_mel, t_mel)

            if tc_info.get("b64"):
                t_chr = b64_decode_uint8(tc_info["b64"], (tc_info["shape"][0], 12)).astype(np.float32) / 255.0
                s_chr, best_shift = score_chroma(q_chr, t_chr)

            # Fusion: take min of available (you can tune weights later)
            # smaller score = better
            score_list = []
            if s_mel is not None:
                score_list.append(float(s_mel))
            if s_chr is not None:
                score_list.append(float(s_chr))
            fused = min(score_list) if score_list else 1e18

            results.append({
                "id": cid,
                "score": float(fused),
                "score_melody": None if s_mel is None else float(s_mel),
                "score_chroma": None if s_chr is None else float(s_chr),
                "chroma_best_shift": best_shift,
            })

            if (idx + 1) % 10 == 0:
                log(f"… matched {idx+1}/{len(candidates)}")

        results.sort(key=lambda r: r["score"])
        top_k = int(payload.get("top_k", 10))
        out = {
            "status": "ok",
            "mode": "query",
            "top": results[:top_k],
            "count": len(results),
        }
        sys.stdout.write(json.dumps(out))
        sys.stdout.flush()

    except Exception as e:
        log("❌ Python exception:\n" + traceback.format_exc())
        sys.stdout.write(json.dumps({"status": "error", "message": str(e)}))
        sys.stdout.flush()
        sys.exit(1)

if __name__ == "__main__":
    main()
