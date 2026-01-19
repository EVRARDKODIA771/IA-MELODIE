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

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "missing filePath"}))
        return 2

    file_path = sys.argv[1]
    t0 = time.time()
    log(f"[fingerprint.py] Start. file={file_path}")

    try:
        # Dépendances à installer côté Render:
        # pip install numpy librosa soundfile
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

        # 3) Features (robustes pour identification)
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
        #    On fait une empreinte stable:
        #    - moyennes + variances (par bande)
        #    - quantifiées en int16
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
        # (on scale pour capturer info tout en restant stable)
        q = np.round(vec * 1000.0).astype(np.int16)  # int16 stable
        fp_bytes = q.tobytes()

        fingerprint = sha256_hex(fp_bytes)

        # 5) Extra: “short fingerprint” lisible (pas obligatoire)
        short_fp = fingerprint[:16]

        elapsed = time.time() - t0
        log(f"[fingerprint.py] Done in {elapsed:.2f}s. fingerprint={short_fp}...")

        # JSON final (stdout uniquement)
        out = {
            "fingerprint": fingerprint,
            "fingerprint_short": short_fp,
            "meta": {
                "sr": sr,
                "duration_sec": duration,
                "tempo_bpm_est": tempo,
                "vector_dim": int(q.shape[0]),
                "quant_scale": 1000.0
            }
        }
        print(json.dumps(out, ensure_ascii=False))

        return 0

    except Exception as e:
        log("[fingerprint.py] ERROR: " + str(e))
        log(traceback.format_exc())
        # stdout: JSON erreur (même si server.js va marquer error)
        print(json.dumps({"error": str(e)}))
        return 1

if __name__ == "__main__":
    sys.exit(main())
