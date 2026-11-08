"""Métricas objetivas de calidad e inteligibilidad de voz: SNR, SI-SDR, PESQ, STOI."""
from __future__ import annotations
import numpy as np

from utils import SAMPLE_RATE, match_length


def snr(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference, estimate = match_length(reference, estimate)
    noise = reference - estimate
    return 10.0 * np.log10((np.sum(reference ** 2) + 1e-12) / (np.sum(noise ** 2) + 1e-12))


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference, estimate = match_length(reference, estimate)
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    alpha = np.dot(estimate, reference) / (np.dot(reference, reference) + 1e-12)
    target = alpha * reference
    noise = estimate - target
    return 10.0 * np.log10((np.sum(target ** 2) + 1e-12) / (np.sum(noise ** 2) + 1e-12))


def pesq_score(reference: np.ndarray, estimate: np.ndarray, sr: int = SAMPLE_RATE) -> float | None:
    try:
        from pesq import pesq
        reference, estimate = match_length(reference, estimate)
        return float(pesq(sr, reference, estimate, "wb"))
    except Exception as e:  # p. ej. señal demasiado corta o sin voz detectable
        print(f"[PESQ] no disponible: {e}")
        return None


def stoi_score(reference: np.ndarray, estimate: np.ndarray, sr: int = SAMPLE_RATE) -> float | None:
    try:
        from pystoi import stoi
        reference, estimate = match_length(reference, estimate)
        return float(stoi(reference, estimate, sr, extended=False))
    except Exception as e:
        print(f"[STOI] no disponible: {e}")
        return None


def evaluate_all(reference: np.ndarray, estimate: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    return {
        "SNR": snr(reference, estimate),
        "SI-SDR": si_sdr(reference, estimate),
        "PESQ": pesq_score(reference, estimate, sr),
        "STOI": stoi_score(reference, estimate, sr),
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from utils import load_audio, match_length

    root = Path(__file__).resolve().parents[1] / "data" / "raw" / "voicebank"
    clean_wavs = sorted((root / "clean_testset").glob("*.wav")) if (root / "clean_testset").exists() else []
    if not clean_wavs:
        raise SystemExit(f"Coloca WAV reales en {root} para ejecutar la prueba.")
    name0 = clean_wavs[0].name
    clean = load_audio(root / "clean_testset" / name0)
    noisy = load_audio(root / "noisy_testset" / name0)
    clean, noisy = match_length(clean, noisy)
    print(f"Archivo: {name0}")
    print("Metricas (ruidoso vs limpio):")
    print(evaluate_all(clean, noisy))
