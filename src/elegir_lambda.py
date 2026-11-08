"""Elige el peso λ de la pérdida combinada MSE + SI-SDR con las corridas cortas de validación.

Criterio, fijado antes de ver los resultados:
  entre los λ cuya MSE de validación no empeora más de un 10 % respecto a la U-Net
  entrenada solo con MSE en la misma época, se elige el de mayor SI-SDR de validación.
  (Si ninguno cumple, se elige el λ más pequeño.)

Así la pérdida combinada mejora la forma de onda sin sacrificar la fidelidad espectral,
y la elección no usa el conjunto de prueba.

    python src/elegir_lambda.py --runs unet_sisdr_l0.0001 unet_sisdr_l0.001 unet_sisdr_l0.01
"""
from __future__ import annotations
import argparse
import json

from utils import ROOT


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="nombres de los experimentos cortos")
    ap.add_argument("--ref", default="unet_pairs", help="U-Net entrenada solo con MSE (referencia)")
    ap.add_argument("--tolerancia", type=float, default=0.10, help="empeoramiento máximo de la MSE de validación")
    args = ap.parse_args()

    faltan = [n for n in args.runs if not (ROOT / "experiments" / n / "history.json").exists()]
    if faltan:
        raise SystemExit("Aún no terminan: " + ", ".join(faltan) + ".\n"
                         "train.py escribe history.json al final del entrenamiento; espera a que acabe "
                         "(revisa el avance con: tail -f lambda.log).")
    rows = []
    for name in args.runs:
        h = json.load(open(ROOT / "experiments" / name / "history.json"))
        ep = len(h["val"])
        rows.append({"exp": name, "epocas": ep, "val_mse": h["val_mse"][-1], "val_sisdr": h["val_sisdr"][-1],
                     "seg_por_epoca": sum(h.get("seg_por_epoca", [0])) / max(len(h.get("seg_por_epoca", [1])), 1)})
    ep = rows[0]["epocas"]
    ref = json.load(open(ROOT / "experiments" / args.ref / "history.json"))["val"][ep - 1]
    limite = ref * (1 + args.tolerancia)
    print(f"Referencia: U-Net solo MSE, MSE de validación en la época {ep} = {ref:.5f}  (límite {limite:.5f})\n")
    print(f"{'experimento':<24}{'MSE val':>10}{'SI-SDR val':>12}{'cumple':>8}{'s/época':>9}")
    for r in rows:
        r["cumple"] = r["val_mse"] <= limite
        print(f"{r['exp']:<24}{r['val_mse']:>10.5f}{r['val_sisdr']:>11.2f} {'sí' if r['cumple'] else 'no':>7}{r['seg_por_epoca']:>9.0f}")
    validos = [r for r in rows if r["cumple"]]
    elegido = max(validos, key=lambda r: r["val_sisdr"]) if validos else rows[0]
    print(f"\nλ elegido: experimento {elegido['exp']}"
          + ("" if validos else "  (ninguno cumple la tolerancia: se toma el primero de la lista, el λ más pequeño)"))


if __name__ == "__main__":
    main()
