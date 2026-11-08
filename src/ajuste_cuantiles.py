"""Ajuste del cuantil de estimación de ruido de los baselines propios, sobre ENTRENAMIENTO.

- Sustracción espectral: el cuantil ES la estimación del ruido por banda.
- Compuerta espectral: el cuantil es el corte que elige los fotogramas silenciosos
  sobre los que se calculan la media y la desviación del ruido.

Se usa un subconjunto de entrenamiento (no el de prueba) para no ajustar los
baselines sobre los mismos datos con los que se reportan los resultados. Los demás
parámetros se dejan fijos (alpha=1.5, beta=0.05; n_std=1.5, piso=0.05).

    python src/ajuste_cuantiles.py \
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
METODOS = {
    "Sustraccion espectral": lambda y, q: spectral_subtraction(y, quantile=q),
    "Compuerta (propia)": lambda y, q: spectral_gating(y, n_std=1.5, quantile=q),
}
# Valores por defecto actuales de baseline.py, para comparar con el óptimo de la rejilla.
ACTUAL = {"Sustraccion espectral": 0.40, "Compuerta (propia)": 0.20}


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
                    default=[0.05, 0.10, 0.15, 0.20, 0.30, 0.40],
                    help="valores de cuantil a probar (se prueban en ambos métodos)")
    ap.add_argument("--tag", default="", help="sufijo del archivo de salida")
    args = ap.parse_args()

    pairs = pair_by_filename(args.clean, args.noisy)
    rng = np.random.default_rng(args.seed)
    idx = sorted(rng.choice(len(pairs), size=min(args.n, len(pairs)), replace=False))
    audios = [match_length(load_audio(pairs[i][0]), load_audio(pairs[i][1])) for i in idx]
    print(f"{len(audios)} pares de entrenamiento cargados")

    rows = [{"metodo": "Ruidoso (referencia)", "cuantil": np.nan,
             **_mean_metrics(audios, lambda y: y)}]
    for metodo, fn in METODOS.items():
        for q in args.cuantiles:
            res = _mean_metrics(audios, lambda y: fn(y, q))
            rows.append({"metodo": metodo, "cuantil": q, **res})
            print(f"  {metodo:<22} q={q:<5} " + "  ".join(f"{k} {res[k]:.3f}" for k in METRICS))

    df = pd.DataFrame(rows)
    out = ROOT / "results" / "metrics" / f"ajuste_cuantiles{'_' + args.tag if args.tag else ''}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(out, index=False)

    print("\n" + df.round(3).to_string(index=False))
    print(f"\nMejor cuantil según {args.criterio} (valor actual entre paréntesis):")
    for metodo in METODOS:
        sub = df[df["metodo"] == metodo]
        best = sub.loc[sub[args.criterio].idxmax()]
        borde = best["cuantil"] in (min(args.cuantiles), max(args.cuantiles))
        print(f"  {metodo:<22} q={best['cuantil']:.2f}  ({ACTUAL[metodo]:.2f})"
              + ("  <- en el borde de la rejilla: amplíala" if borde else ""))
    print(f"Tabla guardada en {out}")


if __name__ == "__main__":
    main()
