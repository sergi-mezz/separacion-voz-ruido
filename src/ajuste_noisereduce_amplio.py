"""Ajuste ampliado de la compuerta de noisereduce sobre ENTRENAMIENTO (sin tocar la prueba).

El ajuste original (ajuste_noisereduce.py) solo variaba el modo y prop_decrease. Aquí se
exploran también el umbral del modo estacionario (n_std_thresh_stationary), los del modo
no estacionario (thresh_n_mult_nonstationary, time_constant_s) y el tamaño de la FFT,
con los mismos 200 pares de entrenamiento y el mismo criterio (SI-SDR medio).

Como referencia se mide en el mismo subconjunto la compuerta propia (n_std = 17).

    python src/ajuste_noisereduce_amplio.py \
      --clean data/raw/voicebank/clean_trainset --noisy data/raw/voicebank/noisy_trainset
"""
from __future__ import annotations
import argparse
import itertools
from multiprocessing import Pool

import numpy as np
import pandas as pd
import noisereduce as nr

from utils import load_audio, match_length, SAMPLE_RATE, ROOT
from baseline import spectral_gating
from metrics import si_sdr, evaluate_all
from dataset import pair_by_filename


def nr_config(y, stationary, prop, n_fft, n_std=1.5, mult=2.0, tc=2.0):
    kw = dict(y=y.astype(np.float32), sr=SAMPLE_RATE, stationary=stationary, prop_decrease=prop, n_fft=n_fft)
    if stationary:
        kw["n_std_thresh_stationary"] = n_std
    else:
        kw["thresh_n_mult_nonstationary"] = mult
        kw["time_constant_s"] = tc
    return np.asarray(nr.reduce_noise(**kw), dtype=np.float32)[:len(y)]


AUDIOS = None


def _score(cfg):
    kind, params = cfg
    if kind == "ruidoso":
        fn = lambda y: y
    elif kind == "propia":
        fn = spectral_gating
    elif kind == "est":
        n_fft, n_std, prop = params
        fn = lambda y: nr_config(y, True, prop, n_fft, n_std=n_std)
    else:
        n_fft, mult, tc, prop = params
        fn = lambda y: nr_config(y, False, prop, n_fft, mult=mult, tc=tc)
    return float(np.mean([si_sdr(c, fn(y)[:len(c)]) for c, y in AUDIOS]))


def configs():
    cfg = [("ruidoso", ()), ("propia", ())]
    cfg += [("est", p) for p in itertools.product([512, 1024], [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0], [0.2, 0.4, 0.6, 0.8, 1.0])]
    cfg += [("noest", p) for p in itertools.product([512, 1024], [1.0, 2.0, 3.0, 4.0], [0.5, 1.0, 2.0, 4.0], [0.2, 0.4, 0.6, 0.8, 1.0])]
    return cfg


def _init(cache):
    global AUDIOS
    d = np.load(cache, allow_pickle=True)
    AUDIOS = list(zip(d["clean"], d["noisy"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", required=True, help="voz limpia de ENTRENAMIENTO")
    ap.add_argument("--noisy", required=True, help="voz ruidosa de ENTRENAMIENTO")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--desde", type=int, default=0, help="primera configuración (para correr por tramos)")
    ap.add_argument("--hasta", type=int, default=None)
    ap.add_argument("--procesos", type=int, default=1)
    args = ap.parse_args()

    cache = ROOT / "results" / "metrics" / "_cache_ajuste_nr.npz"
    if not cache.exists():
        # Mismo subconjunto que ajuste_noisereduce.py y ajuste_baselines.py
        pairs = pair_by_filename(args.clean, args.noisy)
        rng = np.random.default_rng(args.seed)
        idx = sorted(rng.choice(len(pairs), size=min(args.n, len(pairs)), replace=False))
        audios = [match_length(load_audio(pairs[i][0]), load_audio(pairs[i][1])) for i in idx]
        np.savez(cache, clean=np.array([a for a, _ in audios], dtype=object), noisy=np.array([b for _, b in audios], dtype=object))
        print(f"{len(audios)} pares de entrenamiento guardados en caché")
    cfg = configs()[args.desde:args.hasta]
    if args.procesos > 1:
        with Pool(args.procesos, initializer=_init, initargs=(str(cache),)) as pool:
            res = pool.map(_score, cfg)
    else:
        _init(str(cache))
        res = [_score(c) for c in cfg]
    rows = []
    for (kind, p), r in zip(cfg, res):
        row = {"metodo": {"ruidoso": "Ruidoso", "propia": "Compuerta propia (n_std = 17)",
                          "est": "noisereduce estacionario", "noest": "noisereduce no estacionario"}[kind], "SI-SDR": r}
        if kind == "est":
            row.update(n_fft=p[0], n_std=p[1], prop=p[2])
        elif kind == "noest":
            row.update(n_fft=p[0], mult=p[1], tc=p[2], prop=p[3])
        rows.append(row)
    out = ROOT / "results" / "metrics" / "ajuste_noisereduce_amplio.csv"
    df = pd.DataFrame(rows)
    if args.desde > 0 and out.exists():
        df = pd.concat([pd.read_csv(out), df], ignore_index=True)
    df.round(4).to_csv(out, index=False)
    print(f"{len(df)} configuraciones acumuladas en {out}")
    print(df.sort_values("SI-SDR", ascending=False).head(10).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
