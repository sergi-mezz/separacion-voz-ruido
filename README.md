# Separación de voz y ruido mediante autoencoders convolucionales

Trabajo Terminal (ESCOM-IPN): realce de voz en señales monofónicas contaminadas con
ruido ambiental. Se comparan, con el mismo protocolo, tres enfoques:

- **Baselines clásicos**: compuerta espectral propia, compuerta de `noisereduce` y
  sustracción espectral, con sus parámetros ajustados sobre datos de entrenamiento.
- **Autoencoder convolucional base** (max-pooling, 0.19 M parámetros) y su variante con
  convoluciones con paso (ablación del submuestreo).
- **U-Net con conexiones de salto** (1.93 M parámetros, modelo principal) y su variante sin
  saltos (ablación), más un experimento con pérdida combinada MSE + SI-SDR.

Evaluación con SNR, SI-SDR, PESQ (banda ancha, P.862.2) y STOI sobre los 824 pares de
prueba de VoiceBank-DEMAND, con pruebas de Wilcoxon + Holm, oráculo de fase y una prueba de
generalización de −5 a 10 dB con ruido visto y no visto.

## Resultados principales (824 pares de prueba)

| Sistema                         | SNR (dB) | SI-SDR (dB) | PESQ | STOI  |
|---------------------------------|---------:|------------:|-----:|------:|
| Ruidoso (entrada)               |     8.45 |        8.45 | 1.97 | 0.921 |
| Compuerta espectral (propia)    |    14.74 |       14.64 | 2.33 | 0.913 |
| Compuerta (noisereduce)         |     8.70 |       11.33 | 2.18 | 0.921 |
| Sustracción espectral           |    13.67 |       13.61 | 2.36 | 0.917 |
| Autoencoder MaxPool (base)      |    13.21 |       13.15 | 2.40 | 0.916 |
| Autoencoder con paso            |    16.36 |       16.31 | 2.50 | 0.922 |
| U-Net sin saltos                |    16.32 |       16.39 | 2.69 | 0.938 |
| **U-Net (MSE)**                 |**18.17** |   **18.09** |**2.66**|**0.943**|
| U-Net (MSE + SI-SDR)            |    18.48 |       18.78 | 2.82 | 0.942 |
| Oráculo de fase (referencia)    |    20.13 |       20.08 | 3.95 | 0.989 |

## Entorno

Probado con Python 3.9, PyTorch 2.8.0, librosa 0.11.0, soundfile 0.13.1, NumPy 2.0.2,
SciPy 1.13.1, pesq 0.0.4, pystoi 0.4.1 y noisereduce 3.0.3.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

El código elige el dispositivo automáticamente: CUDA (NVIDIA) → MPS (Apple Silicon) → CPU.
Si alguna operación falla en MPS: `export PYTORCH_ENABLE_MPS_FALLBACK=1`.

## Datos

VoiceBank-DEMAND, versión de 28 hablantes (Edinburgh DataShare, doi:10.7488/ds/2117).
Los datos no se versionan; colócalos así:

```
data/raw/voicebank/
├── clean_trainset/   noisy_trainset/   (11 572 pares)
├── clean_testset/    noisy_testset/    (824 pares)
└── logfiles/         log_testset.txt   (tipo de ruido y SNR de cada archivo)
```

Los pares se emparejan por nombre de archivo y todo se remuestrea a 16 kHz, mono, al cargar.

## Estructura

```
main.py                     evaluación en el conjunto de prueba (baselines + modelo)
src/
├── utils.py                carga/guardado de audio, semilla, dispositivo, constantes
├── preprocessing.py        STFT/ISTFT, magnitud y fase, compresión log(1 + x)
├── dataset.py              emparejamiento de pares, segmentos de 2.04 s, recorte aleatorio
├── mix_audio.py            mezcla voz + ruido a una SNR objetivo
├── baseline.py             compuerta espectral (propia y noisereduce), sustracción espectral
├── ajuste_baselines.py     ajuste del cuantil (sustracción) y de n_std (compuerta propia)
├── ajuste_noisereduce.py   ajuste del modo y de prop_decrease de noisereduce
├── ajuste_noisereduce_amplio.py  verificación: búsqueda amplia en noisereduce
├── ajuste_cuantiles.py     exploración del cuantil de estimación del ruido
├── models/
│   ├── autoencoder.py           autoencoder base (max-pooling)
│   ├── autoencoder_strided.py   variante con convoluciones con paso
│   └── unet.py                  U-Net (con y sin conexiones de salto)
├── losses.py               pérdida combinada MSE + SI-SDR con ISTFT diferenciable
├── train.py                entrenamiento (recorte aleatorio por época, validación, checkpoints;
│                           --log-grad-norm guarda la norma del gradiente)
├── norma_gradiente.py      norma del gradiente al inicio y al final (¿actúa el recorte?)
├── elegir_lambda.py        selección de λ de la pérdida combinada con validación
├── metrics.py              SNR, SI-SDR, PESQ y STOI
├── significancia.py        Wilcoxon pareada + corrección de Holm; % de locuciones que gana la U-Net
├── oraculo_fase.py         magnitud limpia + fase ruidosa (referencia)
└── generalizacion.py       ruido real remezclado de −5 a 10 dB (ruido visto y no visto)
experiments/<exp>/          checkpoints/best.pt, last.pt e history.json
results/
├── metrics/                CSV de métricas (promedios, por locución, pruebas)
├── figures/                figuras
└── audio_examples/<tag>/   ejemplos de audio por modelo
```

## Reproducir los experimentos

### Pruebas de módulos (sobre archivos reales)

```bash
python src/preprocessing.py   # ISTFT(STFT(x)) recupera x
python src/dataset.py
python src/baseline.py
python src/metrics.py
```

Los baselines usan los parámetros ya ajustados sobre 200 pares de entrenamiento
(cuantil 0.40 en la sustracción, n_std = 17 en la compuerta propia y prop_decrease = 0.4
en noisereduce). Los comandos están en «Ajuste de los baselines».

> Los comandos usan arreglos (`TRAIN=(...)` y `"${TRAIN[@]}"`), que funcionan igual en
> zsh (la terminal de macOS) y en bash; con `TRAIN="..."` zsh pasa todo como un solo argumento.

### Ajuste de los baselines (200 pares de entrenamiento, criterio SI-SDR)

```bash
TUNE=(--clean data/raw/voicebank/clean_trainset --noisy data/raw/voicebank/noisy_trainset --n 200)
python src/ajuste_baselines.py "${TUNE[@]}"                       # cuantil 0.05–0.60 y n_std 1–5
python src/ajuste_baselines.py "${TUNE[@]}" --cuantiles 0.4 --n-std 5 6 8 10 12 15 20 --tag nstd_alto
python src/ajuste_baselines.py "${TUNE[@]}" --cuantiles 0.35 0.45 --n-std 16 17 18 25 --tag fino
python src/ajuste_noisereduce.py "${TUNE[@]}"                     # prop_decrease 0.5–1.0, ambos modos
python src/ajuste_noisereduce.py "${TUNE[@]}" --props 0.1 0.2 0.3 0.4 --tag bajo
python src/ajuste_noisereduce_amplio.py "${TUNE[@]}"              # verificación: 230 configuraciones
python src/ajuste_cuantiles.py "${TUNE[@]}"                       # exploración del cuantil
python src/ajuste_cuantiles.py "${TUNE[@]}" --cuantiles 0.4 0.5 0.6 0.7 0.8 --tag alto
```

### Entrenamiento (80 épocas, lote 16, Adam 1e-3, semilla 42)

```bash
TRAIN=(--mode pairs --clean data/raw/voicebank/clean_trainset --other data/raw/voicebank/noisy_trainset
       --epochs 80 --batch-size 16 --num-workers 4)
python src/train.py "${TRAIN[@]}" --model ae          --loss mse --exp-name ae_pairs
python src/train.py "${TRAIN[@]}" --model ae_strided  --loss mse --exp-name ae_strided
python src/train.py "${TRAIN[@]}" --model unet        --loss mse --exp-name unet_pairs
python src/train.py "${TRAIN[@]}" --model unet_noskip --loss mse --exp-name unet_noskip
```

Tiempos aproximados en un M4 Pro (MPS): ~1 min/época el autoencoder, ~4 min/época la U-Net.

### Pérdida combinada MSE + SI-SDR

```bash
# corridas cortas para elegir λ (solo validación)
for L in 0.0001 0.001 0.01; do
  python src/train.py "${TRAIN[@]}" --model unet --loss mse_sisdr --lambda-sisdr $L --epochs 10 --exp-name unet_sisdr_l$L
done
python src/elegir_lambda.py --runs unet_sisdr_l0.0001 unet_sisdr_l0.001 unet_sisdr_l0.01
# entrenamiento completo con el λ elegido (1e-3)
python src/train.py "${TRAIN[@]}" --model unet --loss mse_sisdr --lambda-sisdr 0.001 --exp-name unet_mse_sisdr
```

(`--epochs 10` después de `"${TRAIN[@]}"` sustituye a las 80 épocas.)

### Evaluación en los 824 pares de prueba

```bash
TEST=(--clean data/raw/voicebank/clean_testset --noisy data/raw/voicebank/noisy_testset)
python main.py "${TEST[@]}" --checkpoint experiments/ae_pairs/checkpoints/best.pt       --model ae          --tag ae_mse
python main.py "${TEST[@]}" --checkpoint experiments/ae_strided/checkpoints/best.pt     --model ae_strided  --tag ae_strided_mse
python main.py "${TEST[@]}" --checkpoint experiments/unet_pairs/checkpoints/best.pt     --model unet        --tag unet_mse
python main.py "${TEST[@]}" --checkpoint experiments/unet_noskip/checkpoints/best.pt    --model unet_noskip --tag unet_noskip_mse
python main.py "${TEST[@]}" --checkpoint experiments/unet_mse_sisdr/checkpoints/best.pt --model unet        --tag unet_mse_sisdr
```

Cada corrida escribe `results/metrics/evaluacion_test_<tag>.csv`,
`results/metrics/metricas_por_locucion_<tag>.csv` y los audios de las 3 primeras
locuciones en `results/audio_examples/<tag>/`.

> **Cuidado:** `--limit N` evalúa solo N pares. Úsalo siempre con un `--tag` nuevo
> (por ejemplo `--tag ae_audio --limit 3` para generar solo audios), para no
> sobrescribir las métricas completas.

### Significancia, oráculo de fase y generalización

```bash
python src/significancia.py          # también guarda el % de locuciones en que gana la U-Net
python src/oraculo_fase.py "${TEST[@]}"
python src/generalizacion.py "${TEST[@]}" --checkpoint experiments/unet_pairs/checkpoints/best.pt --model unet \
    --seen-clean data/raw/voicebank/clean_trainset --seen-noisy data/raw/voicebank/noisy_trainset
```

### Norma del gradiente (¿actúa el recorte en 5.0?)

```bash
GN=(--mode pairs --clean data/raw/voicebank/clean_trainset --other data/raw/voicebank/noisy_trainset
    --batch-size 16 --num-workers 4 --epochs 5 --log-grad-norm)
python src/train.py "${GN[@]}" --model ae          --loss mse --exp-name gn_ae
python src/train.py "${GN[@]}" --model ae_strided  --loss mse --exp-name gn_ae_strided
python src/train.py "${GN[@]}" --model unet        --loss mse --exp-name gn_unet
python src/train.py "${GN[@]}" --model unet_noskip --loss mse --exp-name gn_unet_noskip
python src/train.py "${GN[@]}" --model unet --loss mse_sisdr --lambda-sisdr 0.001 --exp-name gn_unet_mse_sisdr
python src/norma_gradiente.py        # además mide la norma al final con cada last.pt
```

Las corridas cortas repiten, con la misma semilla, las 5 primeras épocas del entrenamiento
original; el script lo comprueba comparando las pérdidas.

## Salidas principales

| Archivo | Contenido |
|---|---|
| `results/metrics/evaluacion_test_<tag>.csv` | métricas promedio de cada sistema |
| `results/metrics/metricas_por_locucion_<tag>.csv` | métricas de cada locución y sistema |
| `results/metrics/significancia.csv` | 33 pruebas de Wilcoxon con corrección de Holm |
| `results/metrics/victorias_unet.csv` | % de locuciones en que la U-Net supera a cada sistema |
| `results/metrics/generalizacion.csv` | métricas por nivel de ruido y condición (visto / no visto) |
| `results/metrics/norma_gradiente.csv` | norma del gradiente al inicio y al final, y pasos recortados |
| `results/metrics/ajuste_*.csv` | rejillas de ajuste de los baselines (200 pares de entrenamiento) |
| `results/metrics/oraculo_fase_por_locucion.csv` | oráculo de fase y señal ruidosa por locución |
| `results/audio_examples/<tag>/` | `NN_clean`, `NN_noisy`, baselines y `NN_modelo` (00 = p232_001) |

## Convenciones

- Audio mono a 16 kHz, muestras en [−1, 1]; STFT con n_fft = 512 (32 ms), hop = 128
  (8 ms), ventana Hann; se usan 256 bandas (se omite Nyquist) y segmentos de 256
  fotogramas (2.04 s); la magnitud se comprime con log(1 + x).
- Los modelos estiman la magnitud limpia; la señal se reconstruye con la fase ruidosa.
- Semilla fija 42 (`utils.set_seed`): un valor arbitrario; lo importante es que esté
  fijada. En GPU puede haber pequeñas variaciones numéricas entre ejecuciones.
- La validación es el 10 % de las locuciones de entrenamiento; el conjunto de prueba
  no se usa para ninguna decisión (ajustes, λ, épocas, elección de modelo).

## Pendiente

- Scripts que generan las figuras de la tesis.
- `src/realzar.py`: realzar un WAV cualquiera con el modelo elegido (demo y manual de uso).
- Carpeta de audios demostrativos con ejemplos seleccionados (casos difíciles, voz de
  fondo, −5 dB, autoencoder frente a U-Net).
