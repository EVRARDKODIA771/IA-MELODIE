import sys
import json
import os
import tempfile

# Importer les modules d'IA
from audio.preprocess import load_audio
from melody.extract import extract_pitch
from melody.fingerprint import pitch_to_fingerprint
from utils.logger import log

def process_audio(file_path):
    try:
        log(f"📥 Traitement du fichier {file_path}")

        # Charger audio, extraire pitch, transformer en "paroles" fictives
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
    # Vérifier si un fichier a été fourni
    if len(sys.argv) >= 2:
        file_path = sys.argv[1]
        result = process_audio(file_path)
        print(json.dumps(result))
    else:
        # Aucun fichier fourni → juste loguer et continuer
        log("⚠️ Aucun fichier fourni, app.py lancé sans traitement. Prêt à recevoir des fichiers via Node.")
        # Ne pas planter Render
        # Si tu veux, tu peux rester en mode serveur ou juste passer
        pass
