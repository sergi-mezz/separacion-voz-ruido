"""Ajuste de la compuerta espectral de noisereduce sobre un subconjunto de ENTRENAMIENTO.

Elige el modo (estacionario / no estacionario) y la proporción de reducción
(prop_decrease) sin usar el conjunto de prueba, para no ajustar el baseline sobre
los mismos datos con los que se reportan los resultados.

    python src/ajuste_noisereduce.py \
      --clean data/raw/voicebank/clean_trainset \
      --noisy data/raw/voicebank/noisy_trainset --n 200
"""
from __future__ import annotations
import argparse

import numpy as np
import pandas as pd

from utils import load_audio, match_length, SAMPLE_RATE, ROOT
from baseline import spectral_gating_nr
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
    ap.add_argument("--props", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    help="valores de prop_decrease a probar")
    ap.add_argument("--tag", default="", help="sufijo del archivo de salida (p. ej. bajo)")
    args = ap.parse_args()

    pairs = pair_by_filename(args.clean, args.noisy)
    rng = np.random.default_rng(args.seed)
    idx = sorted(rng.choice(len(pairs), size=min(args.n, len(pairs)), replace=False))
    audios = [match_length(load_audio(pairs[i][0]), load_audio(pairs[i][1])) for i in idx]
    print(f"{len(audios)} pares de entrenamiento cargados")

    rows = [{"configuracion": "Ruidoso (referencia)", "stationary": None, "prop_decrease": 0.0,
             **_mean_metrics(audios, lambda y: y)}]
    for stationary in (False, True):
        for prop in args.props:
            res = _mean_metrics(
                audios, lambda y: spectral_gating_nr(y, SAMPLE_RATE, stationary=stationary,
                                                     prop_decrease=prop))
            rows.append({"configuracion": f"{'estacionario' if stationary else 'no estacionario'}, p={prop}",
                         "stationary": stationary, "prop_decrease": prop, **res})
            print(f"  {rows[-1]['configuracion']:<26} " +
                  "  ".join(f"{k} {res[k]:.3f}" for k in METRICS))

    df = pd.DataFrame(rows)
    out = ROOT / "results" / "metrics" / f"ajuste_noisereduce{'_' + args.tag if args.tag else ''}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(out, index=False)
    cand = df[df["stationary"].notna()]
    best = cand.loc[cand[args.criterio].idxmax()]
    print("\n" + df.drop(columns=["stationary"]).round(3).to_string(index=False))
    print(f"\nMejor configuración según {args.criterio}: {best['configuracion']}")
    print(f"Tabla guardada en {out}")


if __name__ == "__main__":
    main()
