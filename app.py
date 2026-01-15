from fastapi import FastAPI, UploadFile, File
import tempfile, os

from audio.preprocess import load_audio
from melody.extract import extract_pitch
from melody.fingerprint import pitch_to_fingerprint
from utils.logger import log

app = FastAPI()

@app.post("/fingerprint")
async def fingerprint(file: UploadFile = File(...)):
    log("📥 Audio reçu depuis Node.js")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        path = tmp.name

    try:
        y, sr = load_audio(path)
        pitch = extract_pitch(y, sr)
        fingerprint = pitch_to_fingerprint(pitch)

        return {
            "status": "ok",
            "fingerprint": fingerprint,
            "length": len(fingerprint)
        }

    except Exception as e:
        log(f"❌ Erreur IA : {e}")
        return {"status": "error", "message": str(e)}

    finally:
        os.remove(path)
