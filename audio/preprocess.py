# audio/preprocess.py
import librosa

def load_audio(file_path, sr=22050):
    """
    Charge un fichier audio et renvoie le signal et le taux d'échantillonnage.
    """
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    return y, sr
