"""Pérdida combinada MSE + SI-SDR (experimento adicional).

    L = L_MSE + λ · L_SI-SDR,   con   L_MSE = MSE(X̂, X̃)   y   L_SI-SDR = −SI-SDR(ŝ, s)

El término SI-SDR entra con signo negativo porque el entrenamiento MINIMIZA la pérdida y el
SI-SDR es mejor cuanto más alto: minimizar −SI-SDR equivale a maximizar el SI-SDR. Es la
"pérdida SI-SDR" habitual en la literatura (p. ej., Conv-TasNet se entrena con −SI-SNR).

- MSE sobre la magnitud comprimida log(1 + ·), la misma de todos los modelos (Xu et al., 2015).
- SI-SDR (Le Roux et al., 2019) sobre la forma de onda: ŝ se reconstruye con la ISTFT a
  partir de la magnitud estimada y la fase ruidosa, igual que en la evaluación; s es el
  segmento de voz limpia. Se usa como pérdida, por ejemplo, en Conv-TasNet (Luo y
  Mesgarani, 2019) y se compara con otras pérdidas en Kolbæk et al. (2020).

La ISTFT está escrita solo con operaciones reales (productos de matrices y una convolución
transpuesta para el traslape-suma), para que funcione en la GPU de Apple (MPS) y sea
diferenciable. Reproduce librosa.istft (Hann periódica, n_fft = 512, salto = 128,
center=True); la equivalencia se verificó en NumPy (diferencia máxima ~1e-7).
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import N_FFT, HOP_LENGTH


class ISTFT(nn.Module):
    def __init__(self, n_fft: int = N_FFT, hop: int = HOP_LENGTH):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        K = n_fft // 2 + 1
        n = torch.arange(n_fft, dtype=torch.float64)
        k = torch.arange(K, dtype=torch.float64)
        ang = 2 * math.pi * k[:, None] * n[None, :] / n_fft
        coef = torch.full((K, 1), 2.0, dtype=torch.float64)
        coef[0] = 1.0; coef[-1] = 1.0                                # DC y Nyquist no se duplican
        self.register_buffer("cosm", (coef * torch.cos(ang) / n_fft).float())   # (K, N)
        self.register_buffer("sinm", (coef * torch.sin(ang) / n_fft).float())
        win = 0.5 - 0.5 * torch.cos(2 * math.pi * n / n_fft)        # Hann periódica
        self.register_buffer("win", win.float())
        self.register_buffer("eye", torch.eye(n_fft).unsqueeze(1))  # núcleo identidad para traslape-suma

    def _ola(self, frames):                       # (B, N, T) -> (B, (T-1)*hop + N)
        return F.conv_transpose1d(frames, self.eye, stride=self.hop).squeeze(1)

    def _envelope(self, T: int):
        """Suma de ventanas al cuadrado (normalización de la ISTFT); se calcula una vez por T."""
        if not hasattr(self, "_env_cache"):
            self._env_cache = {}
        key = (T, self.win.device)
        if key not in self._env_cache:
            with torch.no_grad():
                env = self._ola((self.win ** 2)[None, :, None].expand(1, -1, T).contiguous())
                self._env_cache[key] = torch.where(env > 1e-11, env, torch.ones_like(env))
        return self._env_cache[key]

    def forward(self, mag, phase, length: int):
        """mag, phase: (B, K, T) con K = n_fft/2 + 1. Devuelve (B, length)."""
        re, im = mag * torch.cos(phase), mag * torch.sin(phase)
        frames = torch.einsum("bkt,kn->bnt", re, self.cosm) - torch.einsum("bkt,kn->bnt", im, self.sinm)
        y = self._ola(frames * self.win[None, :, None])
        y = y / self._envelope(frames.shape[-1])
        start = self.n_fft // 2
        return y[:, start:start + length]


def si_sdr(est, ref, eps: float = 1e-8):
    """SI-SDR en dB por ejemplo del lote. est, ref: (B, L)."""
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    a = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + eps)
    target = a * ref
    noise = est - target
    return 10 * torch.log10((target.pow(2).sum(-1) + eps) / (noise.pow(2).sum(-1) + eps))


class MSESISDRLoss(nn.Module):
    """L = MSE(log-magnitud) + λ·(−SI-SDR(forma de onda)). Devuelve (pérdida, mse, si_sdr medio)."""

    def __init__(self, lam: float):
        super().__init__()
        self.lam = lam
        self.istft = ISTFT()

    def waveform(self, pred, batch):
        # pred: (B, 1, 256, T) en log(1 + ·). Se añade la banda de Nyquist de la señal ruidosa,
        # como en la evaluación, y se combina con la fase ruidosa completa (257 bandas).
        mag = torch.relu(torch.expm1(pred.squeeze(1)))
        mag = torch.cat([mag, batch["nyq"]], dim=1)                 # (B, 257, T)
        return self.istft(mag, batch["phase_full"], batch["wave_clean"].shape[-1])

    def forward(self, pred, target, batch):
        mse = F.mse_loss(pred, target)
        sdr = si_sdr(self.waveform(pred, batch), batch["wave_clean"]).mean()
        return mse + self.lam * (-sdr), mse.detach(), sdr.detach()   # −SI-SDR: pérdida SI-SDR
