import sys
import json
import os
import tempfile

# Ici tu dois importer tes modules d'IA
from audio.preprocess import load_audio
from melody.extract import extract_pitch
from melody.fingerprint import pitch_to_fingerprint
from utils.logger import log

def process_audio(file_path):
    try:
        log(f"📥 Traitement du fichier {file_path}")

        # Exemple : charger audio, extraire pitch, transformer en "paroles" fictives
        y, sr = load_audio(file_path)
        pitch = extract_pitch(y, sr)
        fingerprint = pitch_to_fingerprint(pitch)

        # Simuler une transformation en paroles de chanson
        paroles = " ".join([f"note{p}" for p in fingerprint[:50]])

        response = {
            "status": "ok",
            "paroles": paroles,
            "length": len(fingerprint)
        }

        return response

    except Exception as e:
        log(f"❌ Erreur IA : {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "message": "Aucun fichier fourni"}))
        sys.exit(1)

    file_path = sys.argv[1]

    # Appeler le traitement
    result = process_audio(file_path)

    # Retourner le JSON à Node
    print(json.dumps(result))
