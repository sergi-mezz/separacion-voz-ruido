"""Norma del gradiente: ¿cuántas veces actúa el recorte en 5.0?

El entrenamiento recorta la norma del gradiente a MAX_GRAD_NORM = 5.0 (train.py).
Pascanu et al. (2013) recomiendan fijar ese umbral mirando la norma típica durante
muchas actualizaciones. Este script la mide en dos momentos del entrenamiento:

- Inicio: lee grad_norms.npy de corridas cortas hechas con --log-grad-norm. Usan la
  misma semilla, así que repiten el comienzo del entrenamiento original; el script
  lo comprueba comparando sus pérdidas con las de la corrida original.
- Final: carga el modelo entrenado (last.pt) y calcula la norma del gradiente sobre
  una época de lotes de entrenamiento, con el mismo código de train.py pero sin
  actualizar los pesos (optimizador con tasa 0).

    # 1) corridas cortas con registro de la norma (5 épocas)
    TRAIN=(--mode pairs --clean data/raw/voicebank/clean_trainset
           --other data/raw/voicebank/noisy_trainset --batch-size 16 --num-workers 4
           --epochs 5 --log-grad-norm)
    python src/train.py "${TRAIN[@]}" --model ae          --loss mse --exp-name gn_ae
    python src/train.py "${TRAIN[@]}" --model ae_strided  --loss mse --exp-name gn_ae_strided
    python src/train.py "${TRAIN[@]}" --model unet        --loss mse --exp-name gn_unet
    python src/train.py "${TRAIN[@]}" --model unet_noskip --loss mse --exp-name gn_unet_noskip
    python src/train.py "${TRAIN[@]}" --model unet --loss mse_sisdr --lambda-sisdr 0.001 \\
        --exp-name gn_unet_mse_sisdr

    # 2) medición al final del entrenamiento y resumen
    python src/norma_gradiente.py

Salidas: results/metrics/norma_gradiente.csv y results/figures/norma_gradiente.png.
"""
from __future__ import annotations
import argparse
import itertools
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import set_seed, ROOT, get_device
from dataset import SpeechEnhancementDataset
from train import build_model, get_loss, run_epoch, MAX_GRAD_NORM

# (nombre, corrida completa original, corrida corta con --log-grad-norm)
MODELOS = [
    ("Autoencoder (max-pooling)", "ae_pairs", "gn_ae"),
    ("Autoencoder (con paso)", "ae_strided", "gn_ae_strided"),
    ("U-Net", "unet_pairs", "gn_unet"),
    ("U-Net sin saltos", "unet_noskip", "gn_unet_noskip"),
    ("U-Net, pérdida combinada", "unet_mse_sisdr", "gn_unet_mse_sisdr"),
]


def normas_finales(exp: str, n_lotes: int, device) -> np.ndarray:
    """Norma del gradiente del modelo entrenado sobre lotes de entrenamiento, sin
    actualizar los pesos. Repite la partición, el tamaño de lote y el recorte de train.py."""
    ck = torch.load(ROOT / "experiments" / exp / "checkpoints" / "last.pt",
                    map_location="cpu", weights_only=False)
    a = ck["args"]
    set_seed(a["seed"])
    model = build_model(a["model"])
    model.load_state_dict(ck["model"])
    model.to(device)
    loss_fn = get_loss(a["loss"], a.get("lambda_sisdr", 1e-3))
    if isinstance(loss_fn, torch.nn.Module):
        loss_fn = loss_fn.to(device)
    ds = SpeechEnhancementDataset(mode=a["mode"], clean_dir=a["clean"], other_dir=a["other"],
                                  snr_range=tuple(a["snr_range"]), segment=True, log_compress=True,
                                  seed=a["seed"], random_crop=True, with_wave=(a["loss"] == "mse_sisdr"))
    ds.set_epoch(a["epochs"])   # recortes de la última época
    n_total = len(ds)
    n_train = n_total - max(1, int(n_total * a["val_split"]))
    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(a["seed"])).tolist()
    loader = DataLoader(Subset(ds, perm[:n_train]), batch_size=a["batch_size"], shuffle=True,
                        num_workers=2, generator=torch.Generator().manual_seed(a["seed"]))
    if n_lotes:
        loader = itertools.islice(loader, n_lotes)
    normas: list[float] = []
    sin_paso = torch.optim.SGD(model.parameters(), lr=0.0)   # calcula el gradiente, no mueve los pesos
    run_epoch(model, loader, loss_fn, device, sin_paso, normas)
    return np.asarray(normas, dtype=np.float32)


def resumen(nombre: str, fase: str, g: np.ndarray) -> dict:
    return {"modelo": nombre, "fase": fase, "pasos": int(g.size),
            "mediana": float(np.median(g)), "p99": float(np.percentile(g, 99)),
            "max": float(g.max()), "recortados": int((g > MAX_GRAD_NORM).sum()),
            "pct_recortados": float(100 * (g > MAX_GRAD_NORM).mean())}


def reproduce(corta: str, original: str) -> float | None:
    """Mayor diferencia relativa (%) entre las pérdidas de la corrida corta y las mismas
    épocas de la original: si es pequeña, la corrida corta repite el inicio real."""
    try:
        hc = json.load(open(ROOT / "experiments" / corta / "history.json"))
        ho = json.load(open(ROOT / "experiments" / original / "history.json"))
    except FileNotFoundError:
        return None
    n = len(hc["train"])
    dif = [abs(c - o) / abs(o) for k in ("train", "val") for c, o in zip(hc[k], ho[k][:n])]
    return 100 * max(dif)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lotes", type=int, default=0,
                    help="lotes para la medición final (0 = una época completa)")
    ap.add_argument("--recalcular", action="store_true",
                    help="vuelve a medir aunque exista grad_norms_final.npy")
    args = ap.parse_args()
    device = get_device()
    print(f"Dispositivo: {device}  |  umbral de recorte: {MAX_GRAD_NORM}")

    filas, curvas = [], {}
    for nombre, original, corta in MODELOS:
        f_ini = ROOT / "experiments" / corta / "grad_norms.npy"
        if f_ini.exists():
            g = np.load(f_ini)
            curvas[nombre] = g
            fila = resumen(nombre, "inicio", g)
            fila["dif_perdida_pct"] = reproduce(corta, original)
            filas.append(fila)
        else:
            print(f"  (sin corrida corta {corta}: se omite el inicio de {nombre})")
        f_fin = ROOT / "experiments" / original / "grad_norms_final.npy"
        if f_fin.exists() and not args.recalcular:
            g = np.load(f_fin)
        else:
            print(f"Midiendo {nombre} al final del entrenamiento...")
            g = normas_finales(original, args.lotes, device)
            np.save(f_fin, g)
        filas.append(resumen(nombre, "final", g))

    df = pd.DataFrame(filas)
    out = ROOT / "results" / "metrics" / "norma_gradiente.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.4f")
    with pd.option_context("display.width", 140, "display.max_columns", 20):
        print(df.round(3).to_string(index=False))
    print(f"\nTabla en {out}")

    if curvas:
        plt.figure(figsize=(7, 4), dpi=120)
        for nombre, g in curvas.items():
            pasos_x = np.arange(1, g.size + 1)
            # valores de cada paso, tenues (los picos son los que activarían el recorte),
            # y su mediana móvil, para que se lea la tendencia
            linea, = plt.plot(pasos_x, g, lw=0.5, alpha=0.25)
            suave = pd.Series(g).rolling(25, min_periods=1, center=True).median()
            plt.plot(pasos_x, suave, color=linea.get_color(), lw=1.3, label=nombre)
        plt.axhline(MAX_GRAD_NORM, color="k", ls="--", lw=1, label=f"umbral de recorte ({MAX_GRAD_NORM:g})")
        plt.yscale("log")
        plt.xlabel("Paso de entrenamiento"); plt.ylabel("Norma del gradiente (antes del recorte)")
        plt.title("Norma del gradiente al inicio del entrenamiento")
        plt.legend(fontsize=8); plt.grid(alpha=0.3, which="both"); plt.tight_layout()
        fig = ROOT / "results" / "figures" / "norma_gradiente.png"
        fig.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig); plt.close()
        print(f"Figura en {fig}")


if __name__ == "__main__":
    main()
