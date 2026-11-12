"""Oráculo de fase: referencia para los métodos que estiman solo la magnitud.

Para cada par de prueba se reconstruye la señal con la magnitud LIMPIA y la fase RUIDOSA
(ISTFT) y se miden SNR, SI-SDR, PESQ y STOI contra la voz limpia. La diferencia entre el
oráculo y la voz limpia perfecta se debe solo a la fase. No es una cota estricta: con la
fase ruidosa, la magnitud que maximiza la SI-SDR no es exactamente la limpia (máscara
sensible a la fase, Erdogan et al., 2015), por lo que un modelo puede superarlo en
algunas locuciones.

    python src/oraculo_fase.py --clean data/raw/voicebank/clean_testset --noisy data/raw/voicebank/noisy_testset
"""
from __future__ import annotations
import argparse

import numpy as np
import pandas as pd

from utils import load_audio, match_length, ROOT
from preprocessing import stft, istft
from metrics import evaluate_all
from dataset import pair_by_filename


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", required=True)
    ap.add_argument("--noisy", required=True)
    ap.add_argument("--start", type=int, default=0, help="primer par (para procesar por tramos)")
    ap.add_argument("--end", type=int, default=None, help="último par, exclusivo")
    ap.add_argument("--out", default=str(ROOT / "results" / "metrics" / "oraculo_fase_por_locucion.csv"))
    args = ap.parse_args()

    pares = pair_by_filename(args.clean, args.noisy)[args.start:args.end]
    filas = []
    for i, (cp, npth) in enumerate(pares, start=args.start):
        limpia = load_audio(cp)
        ruidosa = load_audio(npth)
        limpia, ruidosa = match_length(limpia, ruidosa)
        S, Y = stft(limpia), stft(ruidosa)
        oraculo = istft(np.abs(S) * np.exp(1j * np.angle(Y)), length=len(limpia)).astype(np.float32)
        for sistema, est in (("Ruidoso", ruidosa), ("Oráculo de fase", oraculo)):
            m = evaluate_all(limpia, est)
            filas.append({"archivo": cp.name, "sistema": sistema, **m})
        if (i + 1) % 50 == 0:
            print(f"  procesados {i + 1}")
    df = pd.DataFrame(filas)
    escribir_encabezado = args.start == 0
    df.to_csv(args.out, mode="w" if escribir_encabezado else "a", header=escribir_encabezado, index=False)
    todo = pd.read_csv(args.out)
    print(f"\n{todo['archivo'].nunique()} locuciones acumuladas")
    print(todo.groupby("sistema")[["SNR", "SI-SDR", "PESQ", "STOI"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
