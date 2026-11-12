"""Experimentos de generalización: evaluación a distintos SNR y con
ruido visto/no visto.

Extrae el ruido real de cada par (ruido = ruidoso - limpio), lo reescala a SNR
objetivo (-5, 0, 5, 10 dB) y evalúa ruidoso/baseline/modelo en cada nivel.

Uso:
    python src/generalizacion.py \
      --clean data/raw/voicebank/clean_testset \
      --noisy data/raw/voicebank/noisy_testset \
      --checkpoint experiments/unet_pairs/checkpoints/best.pt --model unet
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils import load_audio, match_length, ROOT, SAMPLE_RATE
from preprocessing import stft, istft, magnitude_phase, to_complex
from baseline import spectral_gating, spectral_subtraction, spectral_gating_nr
from metrics import evaluate_all
from mix_audio import mix_at_snr
from dataset import pair_by_filename, FREQ_BINS


def _pad_time_multiple(mag, multiple=8):
    T = mag.shape[1]
    pad = (multiple - T % multiple) % multiple
    if pad:
        mag = np.pad(mag, ((0, 0), (0, pad)))
    return mag, T


def enhance_with_model(noisy, model, device, log_compress=True):
    import torch
    S = stft(noisy)
    mag, phase = magnitude_phase(S)
    net_in = mag[:FREQ_BINS, :]
    net_in, T = _pad_time_multiple(net_in, 8)
    x = np.log1p(net_in) if log_compress else net_in
    x = torch.from_numpy(x.astype(np.float32))[None, None].to(device)
    with torch.no_grad():
        pred = model(x).squeeze().cpu().numpy()
    est_mag = np.expm1(pred) if log_compress else pred
    est_mag = np.clip(est_mag, 0.0, None)[:, :T]
    full = np.vstack([est_mag, mag[FREQ_BINS:FREQ_BINS + 1, :]])
    return istft(to_complex(full, phase), length=len(noisy)).astype(np.float32)


def load_model(checkpoint, model_name, device):
    import torch
    if model_name == "ae":
        from models.autoencoder import ConvAutoencoder as M
    elif model_name == "ae_strided":
        from models.autoencoder_strided import ConvAutoencoderStrided as M
    elif model_name == "unet_noskip":
        from models.unet import UNetNoSkip as M
    else:
        from models.unet import UNet as M
    model = M().to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def run(clean_dir, noisy_dir, snrs, checkpoint=None, model_name="unet",
        limit=None, seen_clean=None, seen_noisy=None, seed=42):
    pairs = pair_by_filename(clean_dir, noisy_dir)
    if limit:
        pairs = pairs[:limit]
    rng = np.random.default_rng(seed)

    seen_noises = None
    if seen_clean and seen_noisy:
        seen_pairs = pair_by_filename(seen_clean, seen_noisy)
        rng.shuffle(seen_pairs)
        seen_pairs = seen_pairs[:max(50, (limit or 200))]
        seen_noises = []
        for cp, npth in seen_pairs:
            cl = load_audio(cp); no = load_audio(npth)
            cl, no = match_length(cl, no)
            seen_noises.append((no - cl).astype(np.float32))
        print(f"[ruido visto] {len(seen_noises)} muestras de ruido de entrenamiento")

    device, model = None, None
    systems = ["Ruidoso", "Compuerta (propia)", "Compuerta (noisereduce)", "Sustraccion espectral"]
    if checkpoint:
        import torch
        from utils import get_device
        device = get_device()
        model = load_model(checkpoint, model_name, device)
        systems.append("Modelo")
        print(f"Modelo {model_name} cargado en {device}")

    conditions = ["no_visto"] + (["visto"] if seen_noises is not None else [])
    acc = {(cond, snr, s): {"SI-SDR": [], "PESQ": [], "STOI": []}
           for cond in conditions for snr in snrs for s in systems}

    for i, (cpath, npath) in enumerate(pairs):
        clean = load_audio(cpath)
        noisy = load_audio(npath)
        clean, noisy = match_length(clean, noisy)
        noise_unseen = (noisy - clean).astype(np.float32)
        noise_by_cond = {"no_visto": noise_unseen}
        if seen_noises is not None:
            noise_by_cond["visto"] = seen_noises[int(rng.integers(0, len(seen_noises)))]
        for cond, noise in noise_by_cond.items():
            for snr in snrs:
                mix, _ = mix_at_snr(clean, noise, float(snr), rng=rng)
                mix = mix[:len(clean)]
                outs = {"Ruidoso": mix,
                        "Compuerta (propia)": spectral_gating(mix)[:len(clean)],
                        "Compuerta (noisereduce)": spectral_gating_nr(mix, SAMPLE_RATE)[:len(clean)],
                        "Sustraccion espectral": spectral_subtraction(mix)[:len(clean)]}
                if model is not None:
                    outs["Modelo"] = enhance_with_model(mix, model, device)[:len(clean)]
                for s, est in outs.items():
                    m = evaluate_all(clean, est)
                    for k in ("SI-SDR", "PESQ", "STOI"):
                        if m[k] is not None:
                            acc[(cond, snr, s)][k].append(m[k])
        if (i + 1) % 25 == 0:
            print(f"  procesados {i+1}/{len(pairs)}")

    rows = []
    for (cond, snr, s), d in acc.items():
        if not d["SI-SDR"]:
            continue
        rows.append({"Condicion": cond, "SNR_entrada": snr, "Sistema": s,
                     "SI-SDR": np.mean(d["SI-SDR"]),
                     "PESQ": np.mean(d["PESQ"]) if d["PESQ"] else float("nan"),
                     "STOI": np.mean(d["STOI"]) if d["STOI"] else float("nan")})
    return pd.DataFrame(rows).sort_values(["Condicion", "SNR_entrada", "Sistema"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", required=True)
    ap.add_argument("--noisy", required=True)
    ap.add_argument("--snrs", type=float, nargs="+", default=[-5, 0, 5, 10])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--model", choices=["ae", "ae_strided", "unet", "unet_noskip"], default="unet")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seen-clean", default=None)
    ap.add_argument("--seen-noisy", default=None)
    args = ap.parse_args()

    df = run(args.clean, args.noisy, args.snrs, checkpoint=args.checkpoint,
             model_name=args.model, limit=args.limit,
             seen_clean=args.seen_clean, seen_noisy=args.seen_noisy)
    print("\n=== Generalización: SI-SDR por SNR de entrada ===")
    for cond in df["Condicion"].unique():
        print(f"\n--- Ruido {cond.replace('_', ' ')} ---")
        sub = df[df["Condicion"] == cond]
        print(sub.pivot_table(index="SNR_entrada", columns="Sistema", values="SI-SDR").round(2).to_string())
    out = ROOT / "results" / "metrics" / "generalizacion.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(out, index=False)
    print(f"\nTabla completa guardada en {out}")


if __name__ == "__main__":
    main()
