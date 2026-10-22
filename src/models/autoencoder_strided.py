"""Autoencoder convolucional con convoluciones con paso (variante para ablación).

A diferencia del modelo base (Conv + ReLU + MaxPool / UpSample), aquí el
submuestreo lo hace la propia convolución con paso (stride 2) y el sobremuestreo
una convolución transpuesta (ConvTranspose2d). Se conserva para comparar ambas
estrategias de submuestreo en el reporte del autoencoder.

  Encoder:  Conv2D stride 2 + ReLU  (x3)   1 -> 32 -> 64 -> 128,  256 -> 32
  Decoder:  ConvTranspose2D stride 2 + ReLU (x3)  128 -> 64 -> 32 -> 1,  32 -> 256
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn

    class ConvAutoencoderStrided(nn.Module):
        def __init__(self, base_channels: int = 32):
            super().__init__()
            c = base_channels
            # --- Encoder: Conv2D(stride 2) + ReLU (x3) ---
            self.encoder = nn.Sequential(
                nn.Conv2d(1, c, 3, stride=2, padding=1), nn.ReLU(inplace=True),       # -> c  x 128 x 128
                nn.Conv2d(c, 2 * c, 3, stride=2, padding=1), nn.ReLU(inplace=True),    # -> 2c x 64  x 64
                nn.Conv2d(2 * c, 4 * c, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # -> 4c x 32 x 32 (cuello)
            )
            # --- Decoder: ConvTranspose2D(stride 2) + ReLU (x3) ---
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(4 * c, 2 * c, 3, stride=2, padding=1, output_padding=1), nn.ReLU(inplace=True),  # -> 2c x 64 x 64
                nn.ConvTranspose2d(2 * c, c, 3, stride=2, padding=1, output_padding=1), nn.ReLU(inplace=True),      # -> c  x 128 x 128
                nn.ConvTranspose2d(c, 1, 3, stride=2, padding=1, output_padding=1),                                # -> 1  x 256 x 256
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

except ImportError:
    ConvAutoencoderStrided = None
