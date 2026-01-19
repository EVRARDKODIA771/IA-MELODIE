#!/usr/bin/env python3
import sys
import json
import hashlib
import time
import traceback

def log(msg: str):
    # logs continus pour server.js (stderr)
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

def build_melody_signature(y, sr, *, hop_length=512, n_points=120):
    """
    Construit une signature mélodique robuste au:
    - changement de tonalité (on utilise des INTERVALLES)
    - tempo différent (on downsample à n_points fixes)
    - petits écarts (on médiane + nettoyage)

    Retour:
      melody_sig: list[int] (intervals quantifiés en demi-tons*10)
      melody_meta: dict
    """
    import numpy as np
    import librosa

    # pyin est plus stable pour la voix que pitch_tuning basique
    # fmin/fmax: voix humaine typique ~ 80-400 Hz (homme/femme), on élargit un peu
    fmin = librosa.note_to_hz("C2")  # ~65 Hz
    fmax = librosa.note_to_hz("C6")  # ~1046 Hz

    log("[fingerprint.py] Extracting melody (pyin)...")
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        hop_length=hop_length
    )

    # f0: array length T with NaN where unvoiced
    if f0 is None:
        return [], {"melody_ok": False, "reason": "pyin_returned_none"}

    # Convert to midi for intervals (midi handles ratios nicely)
    midi = librosa.hz_to_midi(f0)  # NaN stays NaN

    # Keep only voiced frames
    voiced = np.isfinite(midi)
    voiced_ratio = float(np.mean(voiced)) if midi.size else 0.0

    if voiced_ratio < 0.05:
        # presque pas de pitch détectable -> pas fiable
        return [], {"melody_ok": False, "reason": "too_few_voiced_frames", "voiced_ratio": voiced_ratio}

    midi_v = midi[voiced].astype(np.float32)

    # Smoothing léger (robustesse)
    # median filter manual (petit)
    if midi_v.size >= 5:
        k = 5
        pad = k // 2
        mv = np.pad(midi_v, (pad, pad), mode="edge")
        sm = np.empty_like(midi_v)
        for i in range(midi_v.size):
            sm[i] = np.median(mv[i:i+k])
        midi_v = sm

    # Intervals (différences) -> invariant à la transposition
    intervals = np.diff(midi_v)  # len-1
    if intervals.size == 0:
        return [], {"melody_ok": False, "reason": "intervals_empty", "voiced_ratio": voiced_ratio}

    # Clip + quantize: demi-tons * 10 (plus fin)
    intervals = np.clip(intervals, -12.0, 12.0)  # max une octave par pas (robuste)
    q_int = np.round(intervals * 10.0).astype(np.int16)

    # Downsample à n_points fixes (tempo-invariant grossier)
    # Si trop court, on pad; si trop long, on sample uniformément
    L = q_int.size
    if L >= n_points:
        idx = np.linspace(0, L - 1, num=n_points).astype(int)
        sig = q_int[idx]
    else:
        sig = np.zeros((n_points,), dtype=np.int16)
        sig[:L] = q_int

    melody_bytes = sig.tobytes()
    melody_hash = sha256_hex(melody_bytes)

    melody_meta = {
        "melody_ok": True,
        "voiced_ratio": voiced_ratio,
        "hop_length": int(hop_length),
        "n_points": int(n_points),
        "quant_scale_semitone_x10": 10.0,
        "interval_clip_semitones": 12.0
    }

    return sig.tolist(), {"melody_hash": melody_hash, **melody_meta}

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "missing filePath"}))
        return 2

    file_path = sys.argv[1]
    t0 = time.time()
    log(f"[fingerprint.py] Start. file={file_path}")

    try:
        import numpy as np
        import librosa

        # 1) Load audio
        target_sr = 22050
        log("[fingerprint.py] Loading audio...")
        y, sr = librosa.load(file_path, sr=target_sr, mono=True)
        duration = float(librosa.get_duration(y=y, sr=sr))
        log(f"[fingerprint.py] Loaded. sr={sr}, samples={len(y)}, duration={duration:.2f}s")

        if len(y) < sr * 2:
            log("[fingerprint.py] Warning: audio is very short; fingerprint may be less robust.")

        # 2) Normalize (robustness)
        eps = 1e-9
        y = y.astype(np.float32)
        y = y / (np.max(np.abs(y)) + eps)

        # 3) Features (robustes pour identification "audio")
        #    - Chroma: signature harmonique
        #    - MFCC: timbre
        #    - Onset strength: dynamique
        log("[fingerprint.py] Computing chroma...")
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)  # (12, T)

        log("[fingerprint.py] Computing MFCC...")
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)  # (20, T)

        log("[fingerprint.py] Computing onset strength...")
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)  # (T,)

        # Tempo (indicatif)
        log("[fingerprint.py] Estimating tempo...")
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        tempo = float(tempo)

        # 4) Statistiques résumées + quantification
        log("[fingerprint.py] Building fingerprint vector...")

        chroma_mean = np.mean(chroma, axis=1)
        chroma_std  = np.std(chroma, axis=1)

        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std  = np.std(mfcc, axis=1)

        onset_mean = float(np.mean(onset_env)) if onset_env.size else 0.0
        onset_std  = float(np.std(onset_env)) if onset_env.size else 0.0

        # Vector: 12*2 + 20*2 + 2 = 66 dims
        vec = np.concatenate([
            chroma_mean, chroma_std,
            mfcc_mean, mfcc_std,
            np.array([onset_mean, onset_std], dtype=np.float32)
        ]).astype(np.float32)

        # Quantification stable
        q = np.round(vec * 1000.0).astype(np.int16)  # int16 stable
        fp_bytes = q.tobytes()

        fingerprint = sha256_hex(fp_bytes)
        short_fp = fingerprint[:16]

        # ✅ 5) AJOUT : Melody signature (pour fredonnement)
        # On calcule sur un sous-échantillonnage (et c'est ok)
        # NOTE: si tu veux encore plus rapide, on peut ne prendre qu'un segment.
        melody_sig, melody_info = build_melody_signature(y, sr, hop_length=512, n_points=120)

        elapsed = time.time() - t0
        log(f"[fingerprint.py] Done in {elapsed:.2f}s. fingerprint={short_fp}...")

        out = {
            "fingerprint": fingerprint,
            "fingerprint_short": short_fp,
            "meta": {
                "sr": sr,
                "duration_sec": duration,
                "tempo_bpm_est": tempo,
                "vector_dim": int(q.shape[0]),
                "quant_scale": 1000.0
            },

            # ✅ Nouveau bloc pour "humming"
            "melody": {
                "signature": melody_sig,     # list[int] longueur fixe (120)
                **melody_info               # melody_ok, voiced_ratio, melody_hash, etc.
            }
        }

        print(json.dumps(out, ensure_ascii=False))
        return 0

    except Exception as e:
        log("[fingerprint.py] ERROR: " + str(e))
        log(traceback.format_exc())
        print(json.dumps({"error": str(e)}))
        return 1

if __name__ == "__main__":
    sys.exit(main())
