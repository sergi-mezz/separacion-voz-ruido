"""Significancia estadística de las comparaciones del Capítulo 5.

Para cada comparación (sistema A frente a sistema B) y cada métrica, aplica la prueba de
Wilcoxon de rangos con signo, pareada sobre las 824 locuciones de prueba (cada locución
procesada por los dos sistemas) y bilateral. Los valores p se corrigen por comparaciones
múltiples con el método de Holm sobre el conjunto de todas las pruebas.

Además guarda, en victorias_unet.csv, el porcentaje de locuciones en que la U-Net supera a la
señal ruidosa, al mejor baseline de cada locución (el mejor de los tres en esa locución y
métrica), al autoencoder base y al autoencoder con paso (Tabla 5.4 de la tesis).

Usa las métricas por locución que guarda main.py (results/metrics/metricas_por_locucion_*.csv);
no requiere reentrenar ni reevaluar.

    python src/significancia.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from utils import ROOT

M = ROOT / "results" / "metrics"
METRICAS = ["SI-SDR", "PESQ", "STOI"]

# Etiqueta de main.py -> nombre del modelo en las tablas
MODELOS = {
    "ae_mse": "AE MaxPool",
    "ae_strided_mse": "AE con paso",
    "unet_noskip_mse": "U-Net sin saltos",
    "unet_mse": "U-Net",
    "unet_mse_sisdr": "U-Net (MSE + SI-SDR)",
}
BASELINES = {
    "Ruidoso": "Ruidoso",
    "Compuerta (propia)": "Compuerta (propia)",
    "Compuerta (noisereduce)": "noisereduce",
    "Sustraccion espectral": "Sustracción",
}

# (A, B, qué pone a prueba)
COMPARACIONES = [
    ("U-Net", "Ruidoso", "Mejora sobre la entrada"),
    ("U-Net", "Compuerta (propia)", "U-Net frente a los clásicos"),
    ("U-Net", "Sustracción", "U-Net frente a los clásicos"),
    ("U-Net", "noisereduce", "U-Net frente a los clásicos"),
    ("U-Net", "AE MaxPool", "U-Net frente al autoencoder base"),
    ("AE con paso", "AE MaxPool", "Ablación: submuestreo"),
    ("U-Net", "U-Net sin saltos", "Ablación: conexiones de salto"),
    ("U-Net sin saltos", "AE con paso", "Resto de la arquitectura"),
    ("U-Net (MSE + SI-SDR)", "U-Net", "Función de pérdida"),
    ("AE MaxPool", "Sustracción", "Autoencoder base frente a los clásicos"),
    ("AE MaxPool", "Compuerta (propia)", "Autoencoder base frente a los clásicos"),
]


def cargar() -> dict[str, pd.DataFrame]:
    """Devuelve, por métrica, una tabla locución × sistema."""
    tablas = {m: {} for m in METRICAS}
    for tag, nombre in MODELOS.items():
        d = pd.read_csv(M / f"metricas_por_locucion_{tag}.csv")
        for m in METRICAS:
            piv = d.pivot(index="archivo", columns="sistema", values=m)
            tablas[m][nombre] = piv["Modelo"]
            for col, b in BASELINES.items():
                tablas[m].setdefault(b, piv[col])
    return {m: pd.DataFrame(t).dropna() for m, t in tablas.items()}


def victorias_unet(datos: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Porcentaje de locuciones en que la U-Net supera a cada sistema (un empate no cuenta)."""
    filas = {}
    for m in METRICAS:
        t = datos[m]
        rivales = {
            "la señal ruidosa": t["Ruidoso"],
            "el mejor baseline de cada locución": t[["Compuerta (propia)", "noisereduce", "Sustracción"]].max(axis=1),
            "el autoencoder base (MaxPool)": t["AE MaxPool"],
            "el autoencoder con convoluciones con paso": t["AE con paso"],
        }
        for nombre, otro in rivales.items():
            filas.setdefault(nombre, {})[m] = 100 * (t["U-Net"] > otro).mean()
    return pd.DataFrame(filas).T[METRICAS]


def holm(p: np.ndarray) -> np.ndarray:
    """Corrección de Holm-Bonferroni."""
    n = len(p)
    orden = np.argsort(p)
    ajust = np.empty(n)
    acum = 0.0
    for rango, i in enumerate(orden):
        acum = max(acum, min(1.0, (n - rango) * p[i]))
        ajust[i] = acum
    return ajust


def main():
    datos = cargar()
    filas = []
    for a, b, tema in COMPARACIONES:
        for m in METRICAS:
            t = datos[m]
            d = t[a] - t[b]
            p = wilcoxon(t[a], t[b]).pvalue
            filas.append({"comparacion": f"{a} vs {b}", "tema": tema, "metrica": m, "n": len(d),
                          "media_A": t[a].mean(), "media_B": t[b].mean(),
                          "dif_media": d.mean(), "dif_mediana": d.median(),
                          "pct_A_gana": 100 * (d > 0).mean(), "p_valor": p})
    res = pd.DataFrame(filas)
    res["p_holm"] = holm(res["p_valor"].to_numpy())
    res["significativo"] = res["p_holm"] < 0.05
    out = M / "significancia.csv"
    res.to_csv(out, index=False, float_format="%.6g")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(res[["comparacion", "metrica", "dif_media", "dif_mediana", "pct_A_gana", "p_holm", "significativo"]]
              .round({"dif_media": 3, "dif_mediana": 3, "pct_A_gana": 1}).to_string(index=False))
    print(f"\n{len(res)} pruebas; {int(res['significativo'].sum())} significativas con Holm (alfa = 0.05).")
    print(f"Tabla completa en {out}")

    vic = victorias_unet(datos)
    out_vic = M / "victorias_unet.csv"
    vic.to_csv(out_vic, index_label="La U-Net supera a", float_format="%.1f")
    print(f"\nPorcentaje de las {len(datos['SI-SDR'])} locuciones en que gana la U-Net:")
    print(vic.round(1).to_string())
    print(f"Tabla en {out_vic}")


if __name__ == "__main__":
    main()
