# melody/extract.py
import numpy as np

def extract_pitch(y, sr):
    """
    Fonction simulée pour extraire les pitches d'un signal audio.
    Pour l'instant renvoie une liste de notes fictives.
    """
    # TODO : remplacer par vraie extraction de pitch
    length = min(len(y)//1000, 100)
    pitches = np.random.randint(60, 72, size=length).tolist()  # notes MIDI aléatoires
    return pitches
