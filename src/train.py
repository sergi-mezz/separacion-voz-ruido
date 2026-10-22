"""Entrenamiento del autoencoder.

"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from utils import SAMPLE_RATE, set_seed, ROOT, get_device
from dataset import SpeechEnhancementDataset

# Evita "Too many open files" en macOS: comparte tensores vía sistema de archivos.
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except Exception:
    pass


def _raise_fd_limit(target: int = 8192) -> None:
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_soft = min(hard, max(soft, target))
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    except Exception:
        pass


def build_model(name: str) -> nn.Module:
    if name == "ae":
        from models.autoencoder import ConvAutoencoder
        return ConvAutoencoder()
    if name == "ae_strided":
        from models.autoencoder_strided import ConvAutoencoderStrided
        return ConvAutoencoderStrided()
    raise ValueError(f"Modelo desconocido: {name}")


def get_loss(name: str):
    """Pérdida de entrenamiento: MSE (la usada en el trabajo) o L1 como alternativa."""
    return {"mse": nn.MSELoss(), "l1": nn.L1Loss()}[name]


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total, count = 0.0, 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            x = batch["noisy"].to(device)
            y = batch["clean"].to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            total += loss.item() * x.size(0)
            count += x.size(0)
    return total / max(count, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["pairs"], default="pairs")
    ap.add_argument("--clean", required=True, help="carpeta de voz limpia")
    ap.add_argument("--other", required=True, help="carpeta de ruidosos (pairs) o de ruido (mix)")
    ap.add_argument("--model", choices=["ae", "ae_strided"], default="ae")
    ap.add_argument("--loss", choices=["mse", "l1"], default="mse",
                    help="función de pérdida: mse (la usada en el trabajo) o l1")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--snr-range", type=float, nargs=2, default=[-5.0, 10.0])
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exp-name", default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    _raise_fd_limit()
    device = get_device()
    print(f"Dispositivo: {device}")

    clean_dir, other_dir, mode = args.clean, args.other, args.mode
    common = dict(mode=mode, clean_dir=clean_dir, other_dir=other_dir,
                  snr_range=tuple(args.snr_range), segment=True, log_compress=True,
                  seed=args.seed)
    # Dos vistas del mismo conjunto: en entrenamiento el segmento de 2.04 s se toma en
    # una posición distinta en cada época (aumento de datos); en validación es fijo.
    train_ds = SpeechEnhancementDataset(**common, random_crop=True)
    val_ds = SpeechEnhancementDataset(**common, random_crop=False)
    n_total = len(train_ds)
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val
    # Misma partición que random_split con la misma semilla.
    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(args.seed)).tolist()
    train_set = Subset(train_ds, perm[:n_train])
    val_set = Subset(val_ds, perm[n_train:])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    print(f"Ejemplos: {n_total}  (train {n_train} / val {n_val})  | recorte aleatorio por época")

    model = build_model(args.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Modelo: {args.model}  ({n_params/1e6:.2f} M parámetros)")
    loss_fn = get_loss(args.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    exp_name = args.exp_name or f"{args.model}_{mode}"
    exp_dir = ROOT / "experiments" / exp_name
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)

    history = {"train": [], "val": []}
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_ds.set_epoch(epoch)   # recorte aleatorio en esta época
        tr = run_epoch(model, train_loader, loss_fn, device, optimizer)
        va = run_epoch(model, val_loader, loss_fn, device, None)
        scheduler.step(va)
        history["train"].append(tr); history["val"].append(va)
        flag = ""
        if va < best_val:
            best_val = va
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch},
                       exp_dir / "checkpoints" / "best.pt")
            flag = "  <- mejor"
        print(f"Época {epoch:03d}/{args.epochs}  train {tr:.5f}  val {va:.5f}  "
              f"({time.time()-t0:.1f}s){flag}")

    torch.save({"model": model.state_dict(), "args": vars(args), "epoch": args.epochs},
               exp_dir / "checkpoints" / "last.pt")
    with open(exp_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    plt.figure(figsize=(6, 4), dpi=120)
    plt.plot(history["train"], label="entrenamiento")
    plt.plot(history["val"], label="validación")
    plt.xlabel("Época"); plt.ylabel(f"Pérdida ({args.loss.upper()})")
    plt.title(f"Curva de pérdida — {exp_name}")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    fig_path = ROOT / "results" / "figures" / f"loss_{exp_name}.png"
    plt.savefig(fig_path); plt.close()

    print(f"\nMejor val: {best_val:.5f}")
    print(f"Checkpoints en {exp_dir/'checkpoints'}")
    print(f"Curva de pérdida en {fig_path}")


if __name__ == "__main__":
    main()
