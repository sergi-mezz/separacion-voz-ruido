"""Autoencoder convolucional base.

Estructura:
  Encoder:    Conv2D + ReLU + MaxPool  (submuestreo por max-pooling)
  Bottleneck: representación comprimida
  Decoder:    UpSampling + Conv2D + ReLU
  Salida:     espectrograma de magnitud estimado (1 canal)
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn

    class ConvAutoencoder(nn.Module):
        def __init__(self, base_channels: int = 32):
            super().__init__()
            c = base_channels
            # --- Encoder: Conv2D + ReLU + MaxPool (x3) ---
            self.enc1 = nn.Sequential(
                nn.Conv2d(1, c, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2))      # -> c  x 128 x 128
            self.enc2 = nn.Sequential(
                nn.Conv2d(c, 2 * c, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2))  # -> 2c x 64 x 64
            self.enc3 = nn.Sequential(
                nn.Conv2d(2 * c, 4 * c, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2))  # -> 4c x 32 x 32 (cuello)
            # --- Decoder: UpSampling + Conv2D + ReLU (x3) ---
            self.dec1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(4 * c, 2 * c, 3, padding=1), nn.ReLU(inplace=True))               # -> 2c x 64 x 64
            self.dec2 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(2 * c, c, 3, padding=1), nn.ReLU(inplace=True))                   # -> c  x 128 x 128
            self.dec3 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(c, 1, 3, padding=1))                                              # -> 1  x 256 x 256

        def forward(self, x):
            x = self.enc1(x); x = self.enc2(x); x = self.enc3(x)
            x = self.dec1(x); x = self.dec2(x); x = self.dec3(x)
            return x

except ImportError:
    ConvAutoencoder = None
