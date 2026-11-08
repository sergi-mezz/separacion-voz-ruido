"""Dataset para realce de voz, con dos fuentes intercambiables.

Modo "pairs"  -> VoiceBank-DEMAND: pares (limpio, ruidoso) emparejados por nombre.
Modo "mix"    -> VCTK + DEMAND/MUSAN: mezcla al vuelo a un SNR aleatorio.

Entrega la MAGNITUD del espectrograma (entrada=ruidosa, objetivo=limpia) y la
fase de la ruidosa (para reconstruir con la ISTFT).

    python src/dataset.py   # autoprueba sobre datos reales de VoiceBank
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from utils import SAMPLE_RATE, load_audio
from preprocessing import stft, magnitude_phase
from mix_audio import mix_at_snr

FREQ_BINS = 256
SEGMENT_FRAMES = 256
SEGMENT_SAMPLES = (SEGMENT_FRAMES - 1) * 128


def pair_by_filename(clean_dir, noisy_dir):
    clean_dir, noisy_dir = Path(clean_dir), Path(noisy_dir)
    clean = {p.name: p for p in clean_dir.glob("*.wav")}
    noisy = {p.name: p for p in noisy_dir.glob("*.wav")}
    common = sorted(set(clean) & set(noisy))
    missing = sorted((set(clean) ^ set(noisy)))
    if missing:
        print(f"[dataset] aviso: {len(missing)} archivos sin par (se ignoran)")
    return [(clean[n], noisy[n]) for n in common]


def list_sources(clean_dir, noise_dir):
    clean = sorted(Path(clean_dir).glob("*.wav"))
    noise = sorted(Path(noise_dir).glob("*.wav"))
    return clean, noise


def random_segment(y, n_samples, rng):
    if len(y) >= n_samples:
        start = int(rng.integers(0, len(y) - n_samples + 1))
        return y[start:start + n_samples]
    pad = np.zeros(n_samples, dtype=y.dtype)
    pad[:len(y)] = y
    return pad


def fit_spectrogram(mag):
    mag = mag[:FREQ_BINS, :]
    if mag.shape[1] >= SEGMENT_FRAMES:
        mag = mag[:, :SEGMENT_FRAMES]
    else:
        mag = np.pad(mag, ((0, 0), (0, SEGMENT_FRAMES - mag.shape[1])))
    if mag.shape[0] < FREQ_BINS:
        mag = np.pad(mag, ((0, FREQ_BINS - mag.shape[0]), (0, 0)))
    return mag


def make_example_from_pair(clean_path, noisy_path, rng, segment=True, with_wave=False):
    clean = load_audio(clean_path)
    noisy = load_audio(noisy_path)
    n = min(len(clean), len(noisy))
    clean, noisy = clean[:n], noisy[:n]
    if segment:
        if n >= SEGMENT_SAMPLES:
            start = int(rng.integers(0, n - SEGMENT_SAMPLES + 1))
            clean = clean[start:start + SEGMENT_SAMPLES]
            noisy = noisy[start:start + SEGMENT_SAMPLES]
        else:
            clean = random_segment(clean, SEGMENT_SAMPLES, np.random.default_rng(0))
            noisy = random_segment(noisy, SEGMENT_SAMPLES, np.random.default_rng(0))
    return _spectra(clean, noisy, with_wave)


def make_example_from_mix(clean_path, noise_path, rng, snr_range=(-5, 10), segment=True, with_wave=False):
    clean = load_audio(clean_path)
    noise = load_audio(noise_path)
    if segment:
        clean = random_segment(clean, SEGMENT_SAMPLES, rng)
    snr = float(rng.uniform(*snr_range))
    noisy, _ = mix_at_snr(clean, noise, snr, rng=rng)
    clean, noisy = clean[:len(noisy)], noisy[:len(clean)]
    return _spectra(clean, noisy, with_wave)


def _fit_frames(m):
    """Recorta o rellena solo el eje de tiempo a SEGMENT_FRAMES (conserva las 257 bandas)."""
    if m.shape[1] >= SEGMENT_FRAMES:
        return m[:, :SEGMENT_FRAMES]
    return np.pad(m, ((0, 0), (0, SEGMENT_FRAMES - m.shape[1])))


def _spectra(clean, noisy, with_wave=False):
    Sn = stft(noisy); Sc = stft(clean)
    noisy_mag, noisy_phase = magnitude_phase(Sn)
    clean_mag, _ = magnitude_phase(Sc)
    out = (fit_spectrogram(noisy_mag).astype(np.float32),
           fit_spectrogram(clean_mag).astype(np.float32),
           fit_spectrogram(noisy_phase).astype(np.float32))
    if not with_wave:
        return out
    # Para la pérdida con SI-SDR: forma de onda limpia del segmento, fase ruidosa con las
    # 257 bandas y la banda de Nyquist de la magnitud ruidosa (la red estima solo 256).
    extra = {"wave_clean": clean.astype(np.float32),
             "phase_full": _fit_frames(noisy_phase).astype(np.float32),
             "nyq": _fit_frames(noisy_mag[FREQ_BINS:FREQ_BINS + 1]).astype(np.float32)}
    return out + (extra,)


try:
    import torch
    from torch.utils.data import Dataset

    class SpeechEnhancementDataset(Dataset):
        def __init__(self, mode, clean_dir, other_dir, snr_range=(-5, 10),
                     segment=True, log_compress=True, seed=42, random_crop=False, with_wave=False):
            assert mode in ("pairs", "mix")
            self.mode = mode; self.segment = segment; self.snr_range = snr_range
            # with_wave=True: también entrega la forma de onda limpia y la fase completa
            # (solo la necesita la pérdida combinada MSE + SI-SDR).
            self.with_wave = with_wave
            self.log_compress = log_compress; self.base_seed = seed
            # random_crop=True (entrenamiento): el segmento cambia en cada época.
            # random_crop=False (validación): segmento fijo, pérdida comparable entre épocas.
            self.random_crop = random_crop
            self.epoch = 0
            if mode == "pairs":
                self.items = pair_by_filename(clean_dir, other_dir)
            else:
                clean, noise = list_sources(clean_dir, other_dir)
                self.items = clean; self.noise = noise
                if not noise:
                    raise ValueError("No hay archivos de ruido para el modo 'mix'.")
            if not self.items:
                raise ValueError("Dataset vacío: revisa las rutas.")

        def __len__(self):
            return len(self.items)

        def _maybe_log(self, m):
            return np.log1p(m) if self.log_compress else m

        def set_epoch(self, epoch: int) -> None:
            """Fija la época actual (para que el recorte aleatorio cambie de una época a otra)."""
            self.epoch = int(epoch)

        def __getitem__(self, idx):
            # Semilla reproducible: depende de (semilla, época, índice) si el recorte es
            # aleatorio por época, o solo de (semilla, índice) si es fijo.
            if self.random_crop:
                rng = np.random.default_rng([self.base_seed, self.epoch, idx])
            else:
                rng = np.random.default_rng(self.base_seed + idx)
            if self.mode == "pairs":
                res = make_example_from_pair(
                    self.items[idx][0], self.items[idx][1], rng, self.segment, self.with_wave)
            else:
                noise_path = self.noise[int(rng.integers(0, len(self.noise)))]
                res = make_example_from_mix(
                    self.items[idx], noise_path, rng, self.snr_range, self.segment, self.with_wave)
            noisy_mag, clean_mag, phase = res[:3]
            x = torch.from_numpy(self._maybe_log(noisy_mag)).unsqueeze(0)
            y = torch.from_numpy(self._maybe_log(clean_mag)).unsqueeze(0)
            ph = torch.from_numpy(phase).unsqueeze(0)
            item = {"noisy": x, "clean": y, "phase": ph}
            if self.with_wave:
                item.update({k: torch.from_numpy(v) for k, v in res[3].items()})
            return item

except ImportError:
    SpeechEnhancementDataset = None


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "data" / "raw" / "voicebank"
    cdir, ndir = root / "clean_testset", root / "noisy_testset"
    if not (cdir.exists() and ndir.exists()):
        raise SystemExit(f"Coloca los WAV de VoiceBank en {root} para ejecutar la prueba.")
    rng = np.random.default_rng(0)
    pairs = pair_by_filename(cdir, ndir)
    print(f"pares emparejados: {len(pairs)}")
    nm, cm, ph = make_example_from_pair(pairs[0][0], pairs[0][1], rng)
    print(f"par: {Path(pairs[0][0]).name}")
    print(f"forma entrada: {nm.shape}  (esperado ({FREQ_BINS}, {SEGMENT_FRAMES}))")
    assert nm.shape == (FREQ_BINS, SEGMENT_FRAMES)
    assert cm.shape == (FREQ_BINS, SEGMENT_FRAMES)
    print("OK: emparejamiento y formas correctos.")
