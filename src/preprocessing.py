"""Preprocesamiento tiempo-frecuencia: STFT / ISTFT y reconstrucción.

Antes de entrenar, se verifica que ISTFT(STFT(x)) reconstruye x sin pérdida:

    python src/preprocessing.py
"""
from __future__ import annotations
import numpy as np
import librosa

from utils import N_FFT, HOP_LENGTH, WIN_LENGTH


def stft(y: np.ndarray) -> np.ndarray:
    return librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH,
                        win_length=WIN_LENGTH, window="hann", center=True)


def istft(S: np.ndarray, length: int | None = None) -> np.ndarray:
    return librosa.istft(S, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                         window="hann", center=True, length=length)


def magnitude_phase(S: np.ndarray):
    return np.abs(S), np.angle(S)


def to_complex(magnitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
    return magnitude * np.exp(1j * phase)


def log_magnitude(magnitude: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return 20.0 * np.log10(magnitude + eps)


def reconstruction_error(y: np.ndarray) -> float:
    y_rec = istft(stft(y), length=len(y))
    return float(np.sqrt(np.mean((y - y_rec) ** 2)))


if __name__ == "__main__":
    # Prueba de sanidad sobre un archivo real: ISTFT(STFT(x)) debe recuperar x.
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from utils import load_audio

    default = Path(__file__).resolve().parents[1] / "data" / "raw" / "voicebank" / "clean_testset"
    wavs = sorted(default.glob("*.wav")) if default.exists() else []
    if not wavs:
        raise SystemExit(f"Coloca WAV reales en {default} para ejecutar la prueba.")
    x = load_audio(wavs[0])
    err = reconstruction_error(x)
    print(f"Archivo: {wavs[0].name}")
    print(f"Error de reconstruccion STFT/ISTFT (RMS): {err:.2e}")
    assert err < 1e-3, "La reconstruccion no es fiel; revisa n_fft/hop/ventana."
    print("OK: la reconstruccion es fiel (< 1e-3).")
