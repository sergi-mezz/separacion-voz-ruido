"""Baseline clásico de reducción de ruido (sin red neuronal).

Implementa la compuerta espectral (spectral gating) como baseline principal, la
sustracción espectral como alternativa, y un envoltorio sobre la compuerta de la
biblioteca noisereduce como implementación de referencia. Sirve como punto de comparación frente al autoencoder y la U-Net. El perfil de ruido se
estima de los fotogramas de menor energía, sin necesidad de un segmento de
"solo ruido" marcado a mano.

Los valores por defecto (cuantil 0.40 en la sustracción, n_std = 17 en la compuerta)
se eligieron por SI-SDR sobre 200 pares de ENTRENAMIENTO con src/ajuste_baselines.py.

    python src/baseline.py  # prueba sobre un par real de VoiceBank
"""
from __future__ import annotations
import numpy as np

from preprocessing import stft, istft, magnitude_phase, to_complex


def _estimate_noise_magnitude(mag: np.ndarray, quantile: float = 0.40) -> np.ndarray:
    """Estima la magnitud del ruido por banda con un cuantil de |Y| a lo largo del tiempo."""
    return np.quantile(mag, quantile, axis=1, keepdims=True)


def spectral_subtraction(noisy: np.ndarray, over_subtraction: float = 1.5,
                         spectral_floor: float = 0.05, quantile: float = 0.40) -> np.ndarray:
    """Sustracción espectral clásica: |X| = max(|Y| - alpha*|D|, beta*|Y|).

    |D| se estima por banda como el cuantil `quantile` de |Y| a lo largo del tiempo.
    """
    S = stft(noisy)
    mag, phase = magnitude_phase(S)
    noise_mag = _estimate_noise_magnitude(mag, quantile)
    clean_mag = mag - over_subtraction * noise_mag
    floor = spectral_floor * mag
    clean_mag = np.maximum(clean_mag, floor)
    S_clean = to_complex(clean_mag, phase)
    return istft(S_clean, length=len(noisy)).astype(np.float32)


def _smooth2d(m: np.ndarray, k: int = 3) -> np.ndarray:
    """Suavizado separable (media móvil) en frecuencia y tiempo, solo con numpy."""
    ker = np.ones(k) / k
    m = np.apply_along_axis(lambda r: np.convolve(r, ker, mode="same"), 1, m)
    m = np.apply_along_axis(lambda r: np.convolve(r, ker, mode="same"), 0, m)
    return m


def spectral_gating(noisy: np.ndarray, n_std: float = 17.0,
                    gain_floor: float = 0.05, smooth: bool = True,
                    quantile: float = 0.20) -> np.ndarray:
    """Compuerta espectral (spectral gating): baseline clásico basado en máscara.

    Estima un perfil de ruido por banda (media y desviación de los fotogramas más
    silenciosos), fija un umbral T(k) = mu + n_std*sigma, y construye una máscara:
    deja pasar los puntos por encima del umbral (voz) y atenúa los de abajo (ruido)
    hasta una ganancia mínima gain_floor. Opcionalmente suaviza la máscara.
    Los fotogramas silenciosos son los que quedan por debajo del cuantil `quantile`.

    n_std es grande porque sigma sale solo del 20 % más bajo (una distribución
    truncada), y es unas 4 veces menor que la desviación real del ruido.
    """
    S = stft(noisy)
    mag, phase = magnitude_phase(S)

    corte = np.quantile(mag, quantile, axis=1, keepdims=True)
    quiet = np.where(mag <= corte, mag, np.nan)
    noise_mean = np.nanmean(quiet, axis=1, keepdims=True)
    noise_std = np.nanstd(quiet, axis=1, keepdims=True)
    threshold = noise_mean + n_std * noise_std

    mask = (mag >= threshold).astype(np.float32)
    mask = mask + (1.0 - mask) * gain_floor
    if smooth:
        mask = _smooth2d(mask, k=3).astype(np.float32)
        mask = np.clip(mask, gain_floor, 1.0)

    S_clean = to_complex(mag * mask, phase)
    return istft(S_clean, length=len(noisy)).astype(np.float32)


def spectral_gating_nr(noisy: np.ndarray, sr: int = 16000,
                       stationary: bool = True, prop_decrease: float = 0.4) -> np.ndarray:
    """Compuerta espectral con la biblioteca de referencia noisereduce (Sainburg et al.).

    Envoltorio sobre noisereduce.reduce_noise, la implementación canónica del
    spectral gating. La configuración por defecto (modo estacionario,
    prop_decrease = 0.4) se eligió con src/ajuste_noisereduce.py sobre 200 pares
    del conjunto de ENTRENAMIENTO, maximizando el SI-SDR en una rejilla de ambos
    modos y prop_decrease de 0.1 a 1.0 (nunca con el conjunto de prueba).
    `prop_decrease` (0-1) controla qué proporción del ruido estimado se atenúa;
    con 1.0 (supresión total) la voz se degrada. Requiere: pip install noisereduce.
    """
    import noisereduce as nr
    out = nr.reduce_noise(y=noisy.astype(np.float32), sr=sr,
                          stationary=stationary, prop_decrease=prop_decrease)
    return np.asarray(out, dtype=np.float32)[:len(noisy)]


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
    for name, fn in [("compuerta espectral", spectral_gating),
                     ("sustraccion espectral", spectral_subtraction),
                     ("compuerta (noisereduce)", spectral_gating_nr)]:
        est = fn(noisy)[:len(clean)]
        res_noise = np.sqrt(np.mean((clean - est) ** 2))
        print(f"{name}: RMS(residuo vs limpio) = {res_noise:.4f}")
    print("OK: baseline ejecutado.")
