#!/usr/bin/env python3
import sys
import json
import hashlib
import time
import traceback
import os
import tempfile
import subprocess

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

def fail(msg: str, *, code: int = 1, extra: dict | None = None):
    payload = {"ok": False, "error": msg}
    if extra:
        payload["extra"] = extra
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))
    sys.stdout.flush()
    return code

# ============================================================
# 0) Convert input audio to WAV (handles webm/opus from MediaRecorder)
# ============================================================
def convert_to_wav(input_path: str, target_sr: int = 22050) -> str | None:
    """
    Returns path to a temporary wav file, or None if conversion fails.
    Needs ffmpeg.
    """
    out_path = tempfile.mktemp(suffix=".wav")
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        input_path,
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        out_path,
    ]
    try:
        log(f"[fingerprint.py] ffmpeg convert -> wav: {' '.join(cmd)}")
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return out_path
        return None
    except FileNotFoundError:
        log("[fingerprint.py] ffmpeg not found on system.")
        try:
            if os.path.exists(out_path):
                os.unlink(out_path)
        except Exception:
            pass
        return None
    except subprocess.CalledProcessError as e:
        log("[fingerprint.py] ffmpeg failed:")
        try:
            log(e.output.decode("utf-8", errors="ignore")[-2000:])
        except Exception:
            pass
        try:
            if os.path.exists(out_path):
                os.unlink(out_path)
        except Exception:
            pass
        return None

# ============================================================
# 1) MATCH SIGNATURE (polyphonique) : chroma SEQUENCE compressée
# ============================================================
def build_chroma_match_signature(y, sr, *, hop_length=512, frames_per_sec=4, max_seconds=30):
    """
    Retour:
      match_sig_flat: list[int]  (uint8 0..127) de taille T*12
      match_shape: [T,12]
      match_meta: dict
    """
    import numpy as np
    import librosa

    if max_seconds and max_seconds > 0:
        max_len = int(sr * max_seconds)
        if y.size > max_len:
            y = y[:max_len]

    log("[fingerprint.py] Computing chroma (match signature)...")

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)  # (12, T)
    if chroma is None or chroma.size == 0:
        return [], [0, 12], {"match_ok": False, "reason": "empty_chroma"}

    eps = 1e-9
    chroma = chroma.astype(np.float32)
    chroma = chroma / (np.sum(chroma, axis=0, keepdims=True) + eps)

    fps_actual = sr / hop_length
    target_fps = float(frames_per_sec) if frames_per_sec and frames_per_sec > 0 else 4.0
    step = int(max(1, round(fps_actual / target_fps)))
    chroma_ds = chroma[:, ::step]  # (12, T2)

    T = chroma_ds.shape[1]
    if T == 0:
        return [], [0, 12], {"match_ok": False, "reason": "downsampled_empty"}

    q = np.clip(np.round(chroma_ds * 127.0), 0, 127).astype(np.uint8)  # (12, T)
    q_T12 = q.T  # (T, 12)
    flat = q_T12.reshape(-1)  # (T*12,)

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
# 2) MELODY SIGNATURE (monophonique) : pitch intervals (pyin)
# ============================================================
def build_melody_signature(y, sr, *, hop_length=512, n_points=120, max_seconds=20):
    import numpy as np
    import librosa

    if max_seconds and max_seconds > 0:
        max_len = int(sr * max_seconds)
        if y.size > max_len:
            y = y[:max_len]

    fmin = librosa.note_to_hz("C2")
    fmax = librosa.note_to_hz("C6")

    log("[fingerprint.py] Extracting melody (pyin)...")
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        hop_length=hop_length
    )

    if f0 is None:
        return [], {"melody_ok": False, "reason": "pyin_returned_none"}

    midi = librosa.hz_to_midi(f0)
    voiced = np.isfinite(midi)
    voiced_ratio = float(np.mean(voiced)) if midi.size else 0.0

    if voiced_ratio < 0.08:
        return [], {"melody_ok": False, "reason": "too_few_voiced_frames", "voiced_ratio": voiced_ratio}

    midi_v = midi[voiced].astype(np.float32)

    if midi_v.size >= 7:
        k = 7
        pad = k // 2
        mv = np.pad(midi_v, (pad, pad), mode="edge")
        sm = np.empty_like(midi_v)
        for i in range(midi_v.size):
            sm[i] = np.median(mv[i:i+k])
        midi_v = sm

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
# MAIN
# ============================================================
def main():
    if len(sys.argv) < 2:
        return fail("missing filePath", code=2)

    input_path = sys.argv[1]
    t0 = time.time()
    log(f"[fingerprint.py] Start. file={input_path}")

    wav_tmp = None
    try:
        import numpy as np
        import librosa

        target_sr = 22050

        # 0) Convert (best effort)
        wav_tmp = convert_to_wav(input_path, target_sr=target_sr)
        load_path = wav_tmp or input_path
        if wav_tmp:
            log(f"[fingerprint.py] Using converted wav: {wav_tmp}")
        else:
            log("[fingerprint.py] Using original file (no conversion). This may fail for webm/opus.")

        # 1) Load audio
        log("[fingerprint.py] Loading audio...")
        y, sr = librosa.load(load_path, sr=target_sr, mono=True)

        duration = float(librosa.get_duration(y=y, sr=sr))
        log(f"[fingerprint.py] Loaded. sr={sr}, samples={len(y)}, duration={duration:.2f}s")

        if len(y) < sr * 2:
            log("[fingerprint.py] Warning: audio is very short; fingerprint may be less robust.")

        # 2) Normalize
        eps = 1e-9
        y = y.astype(np.float32)
        y = y / (np.max(np.abs(y)) + eps)

        # 3) Features fingerprint "global"
        log("[fingerprint.py] Computing chroma...")
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

        log("[fingerprint.py] Computing MFCC...")
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

        log("[fingerprint.py] Computing onset strength...")
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)

        log("[fingerprint.py] Estimating tempo...")
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        tempo = float(_safe_float(tempo, 0.0))

        log("[fingerprint.py] Building fingerprint vector...")
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)

        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        onset_mean = float(np.mean(onset_env)) if onset_env.size else 0.0
        onset_std  = float(np.std(onset_env)) if onset_env.size else 0.0

        vec = np.concatenate([
            chroma_mean, chroma_std,
            mfcc_mean, mfcc_std,
            np.array([onset_mean, onset_std], dtype=np.float32)
        ]).astype(np.float32)

        q = np.round(vec * 1000.0).astype(np.int16)
        fingerprint = sha256_hex(q.tobytes())
        short_fp = fingerprint[:16]

        # 4) MATCH signature (polyphonique)
        match_sig, match_shape, match_meta = build_chroma_match_signature(
            y, sr,
            hop_length=512,
            frames_per_sec=4,
            max_seconds=30
        )

        # 5) Melody signature (mono)
        melody_sig, melody_meta = build_melody_signature(
            y, sr,
            hop_length=512,
            n_points=120,
            max_seconds=20
        )

        elapsed = time.time() - t0
        log(f"[fingerprint.py] Done in {elapsed:.2f}s. fingerprint={short_fp}...")

        out = {
            "ok": True,
            "fingerprint": fingerprint,
            "fingerprint_short": short_fp,
            "meta": {
                "sr": sr,
                "duration_sec": duration,
                "tempo_bpm_est": tempo,
                "vector_dim": int(q.shape[0]),
                "quant_scale": 1000.0,
                "used_ffmpeg_wav": bool(wav_tmp),
            },
            "match": {
                "shape": match_shape,
                "signature": match_sig,
                **match_meta
            },
            "melody": {
                "signature": melody_sig,
                **melody_meta
            }
        }

        return ok(out, code=0)

    except Exception as e:
        log("[fingerprint.py] ERROR: " + str(e))
        log(traceback.format_exc())
        return fail(
            "python_exception",
            code=1,
            extra={"type": str(type(e)), "msg": str(e)}
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
