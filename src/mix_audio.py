"""Generación de mezclas voz + ruido a una relación señal-ruido (SNR) objetivo."""
from __future__ import annotations
import argparse
import numpy as np

from utils import load_audio, save_audio, match_length


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)) + 1e-12)


def tile_to_length(noise: np.ndarray, length: int, rng=None) -> np.ndarray:
    """Ajusta el ruido a `length`: lo repite si es corto, o toma un segmento
    aleatorio si es largo."""
    if len(noise) >= length:
        if rng is not None and len(noise) > length:
            start = int(rng.integers(0, len(noise) - length + 1))
        else:
            start = 0
        return noise[start:start + length]
    reps = int(np.ceil(length / len(noise)))
    return np.tile(noise, reps)[:length]


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float, rng=None):
    """Mezcla voz y ruido al SNR indicado (en dB). Devuelve (mezcla, ruido_escalado).

    SNR = 10*log10(P_voz / P_ruido)  ->  P_ruido = P_voz / 10^(SNR/10).
    """
    noise = tile_to_length(noise, len(clean), rng)
    clean, noise = match_length(clean, noise)

    target_noise_power = (_rms(clean) ** 2) / (10 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_power) / _rms(noise)
    noise_scaled = noise * scale
    mixture = clean + noise_scaled

    peak = np.max(np.abs(mixture))
    if peak > 1.0:
        mixture = mixture / peak
        clean = clean / peak
        noise_scaled = noise_scaled / peak
    return mixture.astype(np.float32), noise_scaled.astype(np.float32)


def measured_snr(clean: np.ndarray, noise: np.ndarray) -> float:
    return 10.0 * np.log10((_rms(clean) ** 2) / (_rms(noise) ** 2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Mezcla voz + ruido a un SNR objetivo.")
    ap.add_argument("--clean", required=True)
    ap.add_argument("--noise", required=True)
    ap.add_argument("--snr", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    c = load_audio(args.clean); n = load_audio(args.noise)
    mix, n_scaled = mix_at_snr(c, n, args.snr, rng=np.random.default_rng(42))
    save_audio(args.out, mix)
    print(f"Mezcla guardada en {args.out}")
