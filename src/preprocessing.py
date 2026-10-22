from __future__ import annotations
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

SAMPLE_RATE = 16_000
ROOT = Path(__file__).resolve().parents[1]

def stft(y: np.ndarray) -> np.ndarray:
    return librosa.stft(y, n_fft = 512)

def istft(S: np.ndarray, length: int | None = None) -> np.ndarray:
    return librosa.istft(S, length=length)

def magnitude_phase(S: np.ndarray):
    return np.abs(S), np.angle(S)

def to_complex(magnitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
    return magnitude * np.exp(1j * phase)

def log_magnitude(magnitude: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return 20.0 * np.log10(magnitude + eps)

def reconstruction_error(y: np.ndarray) -> float:
    y_rec = istft(stft(y), length=len(y))
    return float(np.sqrt(np.mean((y - y_rec) ** 2)))

def load_audio(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    import librosa
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)

def save_audio(path, y: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    """Guarda un arreglo como WAV (16 bits PCM)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    y = np.clip(y, -1.0, 1.0)
    sf.write(str(path), y, sr, subtype="PCM_16")

def match_length(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    return a[:n], b[:n]

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

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    default = Path(__file__).resolve().parents[1] / "data" / "raw" / "voicebank" / "clean_testset"
    wavs = list(default.glob("*.wav"))
    x = load_audio(wavs[0])
    err = reconstruction_error(x)
    print(f"Archivo: {wavs[0].name}")
    print(f"Error de reconstruccion STFT/ISTFT (RMS): {err:.2e}")