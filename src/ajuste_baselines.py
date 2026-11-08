"""Ajuste de los baselines propios sobre un subconjunto de ENTRENAMIENTO.

Se ajusta un parámetro por método, el que controla cuánto ruido se elimina:
- Sustracción espectral: el cuantil con el que se estima el ruido por banda
  (alpha=1.5 y beta=0.05 fijos).
- Compuerta espectral: n_std, el número de desviaciones por encima de la media del
  ruido que fija el umbral (corte de fotogramas silenciosos en el cuantil 0.20 y
  piso de ganancia 0.05 fijos).

Se usan los mismos 200 pares de entrenamiento que en el ajuste de noisereduce, para
no ajustar los baselines sobre el conjunto de prueba.

    python src/ajuste_baselines.py \
      --clean data/raw/voicebank/clean_trainset \
      --noisy data/raw/voicebank/noisy_trainset --n 200
"""
from __future__ import annotations
import argparse

import numpy as np
import pandas as pd

from utils import load_audio, match_length, ROOT
from baseline import spectral_subtraction, spectral_gating
from metrics import evaluate_all
from dataset import pair_by_filename

METRICS = ("SNR", "SI-SDR", "PESQ", "STOI")


def _mean_metrics(audios, fn):
    acc = {k: [] for k in METRICS}
    for clean, noisy in audios:
        m = evaluate_all(clean, fn(noisy)[:len(clean)])
        for k, v in m.items():
            if v is not None:
                acc[k].append(v)
    return {k: float(np.mean(v)) if v else float("nan") for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", required=True, help="voz limpia de ENTRENAMIENTO")
    ap.add_argument("--noisy", required=True, help="voz ruidosa de ENTRENAMIENTO")
    ap.add_argument("--n", type=int, default=200, help="número de pares a usar")
    ap.add_argument("--criterio", default="SI-SDR", choices=list(METRICS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cuantiles", type=float, nargs="+",
                    default=[0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
                    help="cuantiles a probar en la sustracción espectral")
    ap.add_argument("--n-std", type=float, nargs="+",
                    default=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
                    help="valores de n_std a probar en la compuerta espectral")
    ap.add_argument("--tag", default="", help="sufijo del archivo de salida")
    args = ap.parse_args()

    pairs = pair_by_filename(args.clean, args.noisy)
    rng = np.random.default_rng(args.seed)
    idx = sorted(rng.choice(len(pairs), size=min(args.n, len(pairs)), replace=False))
    audios = [match_length(load_audio(pairs[i][0]), load_audio(pairs[i][1])) for i in idx]
    print(f"{len(audios)} pares de entrenamiento cargados")

    barridos = [
        ("Sustraccion espectral", "cuantil", args.cuantiles,
         lambda y, v: spectral_subtraction(y, over_subtraction=1.5, spectral_floor=0.05, quantile=v)),
        ("Compuerta (propia)", "n_std", args.n_std,
         lambda y, v: spectral_gating(y, n_std=v, gain_floor=0.05, quantile=0.20)),
    ]
    rows = [{"metodo": "Ruidoso (referencia)", "parametro": "", "valor": np.nan,
             **_mean_metrics(audios, lambda y: y)}]
    for metodo, param, valores, fn in barridos:
        for v in valores:
            res = _mean_metrics(audios, lambda y: fn(y, v))
            rows.append({"metodo": metodo, "parametro": param, "valor": v, **res})
            print(f"  {metodo:<22} {param}={v:<5} " + "  ".join(f"{k} {res[k]:.3f}" for k in METRICS))

    df = pd.DataFrame(rows)
    out = ROOT / "results" / "metrics" / f"ajuste_baselines{'_' + args.tag if args.tag else ''}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(out, index=False)

    print("\n" + df.round(3).to_string(index=False))
    print(f"\nMejor valor según {args.criterio}:")
    for metodo, param, valores, _ in barridos:
        sub = df[df["metodo"] == metodo]
        best = sub.loc[sub[args.criterio].idxmax()]
        borde = best["valor"] in (min(valores), max(valores))
        print(f"  {metodo:<22} {param} = {best['valor']:g}"
              + ("  <- en el borde de la rejilla: amplíala" if borde else ""))
    print(f"Tabla guardada en {out}")


if __name__ == "__main__":
    main()
