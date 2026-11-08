"""U-Net convolucional con conexiones de salto.

  Encoder:    3 niveles de (Conv 3×3 + BN + ReLU) ×2 + MaxPool 2×2      1 -> 32 -> 64 -> 128
  Cuello:     (Conv 3×3 + BN + ReLU) ×2                                  128 -> 256, 32 × 32
  Decoder:    ConvTranspose 2×2 (paso 2) + concatenación con el nivel
              del encoder (conexión de salto) + (Conv 3×3 + BN + ReLU) ×2
  Salida:     Conv 1×1 -> magnitud estimada (sin activación, como el autoencoder)

UNetNoSkip es la misma red sin las conexiones de salto (ablación): permite separar
el efecto de los saltos del de la mayor capacidad de la U-Net frente al autoencoder.
Parámetros (base_channels = 32): UNet 1 927 841; UNetNoSkip 1 734 305.
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn

    def _block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    class UNet(nn.Module):
        def __init__(self, base_channels: int = 32, skip: bool = True):
            super().__init__()
            c = base_channels
            self.skip = skip
            k = 2 if skip else 1          # con saltos, el decoder recibe el doble de canales
            self.enc1 = _block(1, c)
            self.enc2 = _block(c, 2 * c)
            self.enc3 = _block(2 * c, 4 * c)
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = _block(4 * c, 8 * c)
            self.up3 = nn.ConvTranspose2d(8 * c, 4 * c, 2, stride=2)
            self.dec3 = _block(4 * c * k, 4 * c)
            self.up2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, stride=2)
            self.dec2 = _block(2 * c * k, 2 * c)
            self.up1 = nn.ConvTranspose2d(2 * c, c, 2, stride=2)
            self.dec1 = _block(c * k, c)
            self.out = nn.Conv2d(c, 1, 1)

        def _join(self, up, enc):
            return torch.cat([up, enc], dim=1) if self.skip else up

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            b = self.bottleneck(self.pool(e3))
            d3 = self.dec3(self._join(self.up3(b), e3))
            d2 = self.dec2(self._join(self.up2(d3), e2))
            d1 = self.dec1(self._join(self.up1(d2), e1))
            return self.out(d1)

    class UNetNoSkip(UNet):
        """La misma U-Net sin conexiones de salto (ablación)."""
        def __init__(self, base_channels: int = 32):
            super().__init__(base_channels, skip=False)

except ImportError:
    UNet = None
    UNetNoSkip = None
