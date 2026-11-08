"""Evaluación sobre el conjunto de prueba.

Para cada par (limpio, ruidoso) calcula SNR, SI-SDR, PESQ y STOI de:
  - la señal ruidosa (entrada), el baseline clásico, y el modelo (si --checkpoint).
Agrega promedios en un CSV, guarda audios de ejemplo y una figura comparativa.

Ejemplos
--------
Solo baseline:
    python main.py --clean data/raw/voicebank/clean_testset \
        --noisy data/raw/voicebank/noisy_testset

Baseline + modelo:
    python main.py --clean ... --noisy ... \
        --checkpoint experiments/unet_pairs/checkpoints/best.pt --model unet
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# main.py vive en la raíz; añade src/ al path para importar los módulos.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from utils import SAMPLE_RATE, load_audio, save_audio, match_length, ROOT, get_device
from preprocessing import stft, istft, magnitude_phase, to_complex, log_magnitude
from baseline import spectral_gating, spectral_subtraction
from metrics import evaluate_all
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
    S_est = to_complex(full, phase)
    return istft(S_est, length=len(noisy)).astype(np.float32)


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


def evaluate_dataset(clean_dir, noisy_dir, checkpoint=None, model_name="unet",
                     limit=None, n_examples=3, out_dir=None, tag=""):
    pairs = pair_by_filename(clean_dir, noisy_dir)
    if not pairs:
        raise ValueError("No se encontraron pares. Revisa las rutas.")
    if limit:
        pairs = pairs[:limit]

    device, model = None, None
    if checkpoint:
        import torch
        device = get_device()
        model = load_model(checkpoint, model_name, device)
        print(f"Modelo cargado desde {checkpoint} ({model_name}) en {device}")

    systems = ["Ruidoso", "Compuerta (propia)", "Compuerta (noisereduce)", "Sustraccion espectral"]
    if model is not None:
        systems.append("Modelo")
    accum = {s: {"SNR": [], "SI-SDR": [], "PESQ": [], "STOI": []} for s in systems}
    rows_utt = []   # métricas por locución (para analizar la consistencia entre sistemas)

    saved_examples = []
    for i, (cpath, npath) in enumerate(pairs):
        clean = load_audio(cpath)
        noisy = load_audio(npath)
        clean, noisy = match_length(clean, noisy)
        outputs = {
            "Ruidoso": noisy,
            "Compuerta (propia)": spectral_gating(noisy),
            "Sustraccion espectral": spectral_subtraction(noisy),
        }
        if model is not None:
            outputs["Modelo"] = enhance_with_model(noisy, model, device)
        for name, est in outputs.items():
            est = est[:len(clean)]
            m = evaluate_all(clean, est)
            rows_utt.append({"archivo": Path(cpath).name, "sistema": name,
                             **{k: (np.nan if v is None else v) for k, v in m.items()}})
            for k, v in m.items():
                if v is not None:
                    accum[name][k].append(v)
        if i < n_examples and out_dir is not None:
            ex = Path(out_dir) / "audio_examples" / tag if tag else Path(out_dir) / "audio_examples"
            save_audio(ex / f"{i:02d}_clean.wav", clean)
            save_audio(ex / f"{i:02d}_noisy.wav", noisy)
            for name, est in outputs.items():
                if name == "Ruidoso":
                    continue
                nombre = name.lower().replace(" ", "_").replace("ó", "o")
                save_audio(ex / f"{i:02d}_{nombre}.wav", est[:len(clean)])
            saved_examples.append((clean, outputs))
        if (i + 1) % 25 == 0:
            print(f"  procesados {i+1}/{len(pairs)}")

    rows = {s: {k: (np.mean(v) if v else float("nan")) for k, v in accum[s].items()}
            for s in systems}
    df = pd.DataFrame(rows).T[["SNR", "SI-SDR", "PESQ", "STOI"]]
    return df, saved_examples, pd.DataFrame(rows_utt)


def save_comparison_figure(clean, outputs, path):
    keys = ["Ruidoso"] + [k for k in outputs if k != "Ruidoso"]
    sigs = {"Voz limpia": clean, **{k: outputs[k] for k in keys}}
    n = len(sigs)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.2), dpi=120)
    for ax, (name, y) in zip(axes, sigs.items()):
        mag, _ = magnitude_phase(stft(y[:len(clean)]))
        ax.imshow(log_magnitude(mag), origin="lower", aspect="auto", cmap="magma")
        ax.set_title(name, fontsize=9); ax.set_xlabel("Fotogramas")
    axes[0].set_ylabel("Frecuencia")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", required=True)
    ap.add_argument("--noisy", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--model", choices=["ae", "ae_strided", "unet", "unet_noskip"], default="unet")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-examples", type=int, default=3)
    ap.add_argument("--tag", default="",
                    help="sufijo para los archivos de salida (p. ej. ae_mse), para no sobrescribir")
    args = ap.parse_args()
    sfx = f"_{args.tag}" if args.tag else ""

    out_dir = ROOT / "results"
    df, examples, df_utt = evaluate_dataset(
        args.clean, args.noisy, checkpoint=args.checkpoint, model_name=args.model,
        limit=args.limit, n_examples=args.n_examples, out_dir=out_dir, tag=args.tag)

    print("\n=== Métricas promedio (referencia = voz limpia) ===")
    print(df.round(3).to_string())
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metrics" / f"evaluacion_test{sfx}.csv"
    df.round(4).to_csv(csv_path)
    utt_path = out_dir / "metrics" / f"metricas_por_locucion{sfx}.csv"
    df_utt.to_csv(utt_path, index=False)
    print(f"\nTabla guardada en {csv_path}\nMétricas por locución en {utt_path}")

    if examples:
        clean, outputs = examples[0]
        fig_path = out_dir / "figures" / f"comparacion_espectrogramas{sfx}.png"
        save_comparison_figure(clean, outputs, fig_path)
        print(f"Figura comparativa en {fig_path}")


if __name__ == "__main__":
    main()
