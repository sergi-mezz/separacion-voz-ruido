"""Dataset para realce de voz, con dos fuentes intercambiables.

"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from preprocessing import stft, magnitude_phase, SAMPLE_RATE, load_audio

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


def make_example_from_pair(clean_path, noisy_path, rng, segment=True):
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
    return _spectra(clean, noisy)


def _spectra(clean, noisy):
    Sn = stft(noisy); Sc = stft(clean)
    noisy_mag, noisy_phase = magnitude_phase(Sn)
    clean_mag, _ = magnitude_phase(Sc)
    return (fit_spectrogram(noisy_mag).astype(np.float32),
            fit_spectrogram(clean_mag).astype(np.float32),
            fit_spectrogram(noisy_phase).astype(np.float32))


try:
    import torch
    from torch.utils.data import Dataset

    class SpeechEnhancementDataset(Dataset):
        def __init__(self, clean_dir, other_dir, snr_range=(-5, 10),
                     mode="Pairs", segment=True, log_compress=True, seed=42, random_crop=False):
            self.mode = mode; self.segment = segment; self.snr_range = snr_range
            self.log_compress = log_compress; self.base_seed = seed
            # random_crop=True (entrenamiento): el segmento cambia en cada época.
            # random_crop=False (validación): segmento fijo, pérdida comparable entre épocas.
            self.random_crop = random_crop
            self.epoch = 0
            if mode == "pairs":
                self.items = pair_by_filename(clean_dir, other_dir)
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
                noisy_mag, clean_mag, phase = make_example_from_pair(
                    self.items[idx][0], self.items[idx][1], rng, self.segment)
            x = torch.from_numpy(self._maybe_log(noisy_mag)).unsqueeze(0)
            y = torch.from_numpy(self._maybe_log(clean_mag)).unsqueeze(0)
            ph = torch.from_numpy(phase).unsqueeze(0)
            return {"noisy": x, "clean": y, "phase": ph}

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
