"""Utilidades comunes: semillas, E/S de audio y rutas del proyecto."""
from __future__ import annotations
import random
from pathlib import Path

import numpy as np
import soundfile as sf

# Parámetros globales del proyecto (consistentes en todo el pipeline)
SAMPLE_RATE = 16_000      # Hz
N_FFT = 512               # tamaño de ventana STFT
HOP_LENGTH = 128          # salto entre ventanas (75% de solape)
WIN_LENGTH = 512

ROOT = Path(__file__).resolve().parents[1]


def get_device():
    """Selecciona el mejor dispositivo disponible: CUDA (NVIDIA) > MPS
    (GPU de Apple Silicon, p. ej. M4 Pro) > CPU."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """Fija la semilla aleatoria para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_audio(path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Carga un WAV, lo convierte a mono y lo remuestrea a `sr`. Rango [-1, 1]."""
    import librosa
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


def save_audio(path, y: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    """Guarda un arreglo como WAV (16 bits PCM)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    y = np.clip(y, -1.0, 1.0)
    sf.write(str(path), y, sr, subtype="PCM_16")


def match_length(a: np.ndarray, b: np.ndarray):
    """Recorta dos señales a la misma longitud (la menor)."""
    n = min(len(a), len(b))
    return a[:n], b[:n]
