# Custom Wake Word for Home Assistant Voice PE

## Quick Reference

- **GitHub model repo**: https://github.com/zenoran/hey-nova-voice-pe-model
- **Model manifest URL**: `https://raw.githubusercontent.com/zenoran/hey-nova-voice-pe-model/master/hey_nova.json`
- **Unraid IP**: 10.0.0.99 (SSH: `ssh -i ~/.ssh/id_rsa root@10.0.0.99`)
- **ESPHome container**: `esphome` (on Unraid Docker)
- **Voice PE devices**:
  - `voice-pe-minimal` — IP 10.0.0.132, config `/config/voice-pe-minimal.yaml`
  - `home-assistant-voice` — IP 10.0.0.68, config `/config/home-assistant-voice.yaml`
- **Training machine**: Local dev box with RTX 5080 (GPU incompatible with TF — use CPU)
- **microWakeWord repo**: `/home/nick/dev/microWakeWord/` (kahrendt/microWakeWord clone)

---

## Architecture

```
Piper TTS (generate samples)
        ↓
microWakeWord training pipeline (CPU, ~20k steps)
        ↓
hey_nova.tflite (61KB quantized streaming MixedNet)
        ↓
Push to GitHub (zenoran/hey-nova-voice-pe-model)
        ↓
ESPHome compile on Unraid (embeds model in firmware)
        ↓
OTA flash to Voice PE device(s)
```

**CRITICAL**: The model is embedded in firmware at compile time. Each device needs its own compile because firmware includes device-specific config (name, WiFi, API keys). You cannot share firmware.bin between devices with different configs.

**CRITICAL**: openWakeWord (ONNX) models are NOT compatible with Voice PE on-device detection. Only microWakeWord (TFLite streaming) models work with ESPHome's `micro_wake_word` component.

---

## Training a New Wake Word (End to End)

### Step 1: Generate TTS Samples

```bash
cd /home/nick/dev/microWakeWord
source .venv/bin/activate

# Generate samples with Piper TTS
# Requires piper-sample-generator/ with en_US-libritts_r-medium.pt model
python3 -c "
import subprocess, random
for i in range(2000):
    speed = random.uniform(0.8, 1.2)
    noise = random.uniform(0.5, 1.0)
    subprocess.run([
        'python3', '-m', 'piper_sample_generator',
        '--model', 'piper-sample-generator/en_US-libritts_r-medium.pt',
        '--text', 'hey nova',
        '--output-dir', 'generated_samples',
        '--max-samples', '1',
        '--length-scale', str(speed),
        '--noise-scale', str(noise),
    ])
"
```

Target: 2000+ samples in `generated_samples/`. All 16kHz mono WAV.

### Step 2: Download Augmentation Data

```bash
# MIT Room Impulse Responses (for reverb augmentation)
python3 -c "
import datasets, scipy, numpy as np, os
os.makedirs('mit_rirs', exist_ok=True)
rir_dataset = datasets.load_dataset('davidscripka/MIT_environmental_impulse_responses', split='train', streaming=True)
for row in rir_dataset:
    name = row['audio']['path'].split('/')[-1]
    scipy.io.wavfile.write(f'mit_rirs/{name}', 16000, (row['audio']['array'] * 32767).astype(np.int16))
"

# FMA music (for background noise) — download fma_xs.zip, extract, convert to 16kHz
# See train_hey_nova.py for full download/conversion code
```

### Step 3: Generate Augmented Spectrogram Features

Uses microWakeWord's `Clips`, `Augmentation`, `SpectrogramGeneration`, and `RaggedMmap` classes. See `train_hey_nova.py` for the full pipeline. Generates 32000 training, 2000 validation, 200 test spectrogram features.

### Step 4: Download Negative Datasets

```bash
# From HuggingFace: kahrendt/microwakeword
# dinner_party.zip, dinner_party_eval.zip, no_speech.zip, speech.zip
# Total ~3GB, extracted to negative_datasets/
```

### Step 5: Train

```bash
export CUDA_VISIBLE_DEVICES=""  # Force CPU — RTX 5080 (compute 12.0) incompatible with TF 2.20

python3 -m microwakeword.model_train_eval \
    --training_config=training_parameters.yaml \
    --train 1 \
    --restore_checkpoint 1 \
    --test_tf_nonstreaming 0 \
    --test_tflite_nonstreaming 0 \
    --test_tflite_nonstreaming_quantized 0 \
    --test_tflite_streaming 0 \
    --test_tflite_streaming_quantized 1 \
    --use_weights best_weights \
    mixednet \
    --pointwise_filters "64,64,64,64" \
    --repeat_in_block "1, 1, 1, 1" \
    --mixconv_kernel_sizes "[5], [7,11], [9,15], [23]" \
    --residual_connection "0,0,0,0" \
    --first_conv_filters 32 \
    --first_conv_kernel_size 5 \
    --stride 3
```

Training config (`training_parameters.yaml`):
```yaml
batch_size: 128
clip_duration_ms: 1500
training_steps: [20000]
learning_rates: [0.001]
positive_class_weight: [1]
negative_class_weight: [20]
eval_step_interval: 500
target_false_positives_per_hour: 0.5
window_step_ms: 10
```

Output: `trained_models/wakeword/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite`

### Step 6: Deploy

```bash
# Copy model to GitHub repo
cp trained_models/wakeword/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite \
   /home/nick/dev/hey-nova-voice-pe-model/hey_nova.tflite

# Push to GitHub
cd /home/nick/dev/hey-nova-voice-pe-model
git add hey_nova.tflite && git commit -m "Update model" && git push
```

---

## Compiling and Flashing a Voice PE Device

### Prerequisites
- SSH access: `ssh -i ~/.ssh/id_rsa root@10.0.0.99`
- ESPHome container running on Unraid

### Compile + Flash (one device)

```bash
# 1. Generate C++ from YAML (this fetches the latest model from GitHub)
ssh root@10.0.0.99 "docker exec esphome esphome compile /config/DEVICE_NAME.yaml --only-generate"

# 2. Apply required patches
ssh root@10.0.0.99 "docker exec esphome bash -c '
# Fix esp_http_client exclusion (regenerated every time --only-generate runs)
sed -i \"s/esp_http_client;//g\" /config/.esphome/build/DEVICE_NAME/platformio.ini

# Fix timer API for ESPHome 2026.2+ (only needed for packages-based configs)
# The cached package YAML is at /config/.esphome/packages/321a0194/home-assistant-voice.yaml
# These patches persist across regenerations since the package file is cached:
#   timers.begin()->second  →  timers.front()
#   iterable_timer.second.  →  iterable_timer.
#   iterable_timer.second;  →  iterable_timer;
'"

# 3. Compile (single-threaded to avoid GCC segfaults in Docker)
ssh root@10.0.0.99 "docker exec esphome bash -c '
cd /config/.esphome/build/DEVICE_NAME && platformio run -j 1
'"

# 4. OTA Flash
ssh root@10.0.0.99 "docker exec esphome bash -c '
mkdir -p /config/.esphome/build/DEVICE_NAME/.pioenvs/DEVICE_NAME
cp /config/.esphome/build/DEVICE_NAME/.pio/build/DEVICE_NAME/firmware.bin \
   /config/.esphome/build/DEVICE_NAME/.pioenvs/DEVICE_NAME/firmware.bin
esphome upload /config/DEVICE_NAME.yaml --device DEVICE_IP
'"
```

Replace `DEVICE_NAME` and `DEVICE_IP`:
- Device 1: `voice-pe-minimal`, `10.0.0.132`
- Device 2: `home-assistant-voice`, `10.0.0.68`

### ESPHome Config — Adding a Custom Wake Word

For configs using `packages:` import (like `voice-pe-minimal.yaml`):
```yaml
substitutions:
  name: voice-pe-minimal
  friendly_name: Nova

packages:
  Nabu Casa.Home Assistant Voice PE: github://esphome/home-assistant-voice-pe/home-assistant-voice.yaml@26.2.1

esphome:
  name: ${name}
  friendly_name: ${friendly_name}

wifi:
  ssid: !secret wifi_ssid
  password: !secret wifi_password

api:
  encryption:
    key: !secret api_encryption_key

micro_wake_word:
  models:
    - model: https://raw.githubusercontent.com/zenoran/hey-nova-voice-pe-model/master/hey_nova.json
      id: hey_nova
```

For standalone configs (like `home-assistant-voice.yaml`), add the model entry to the existing `micro_wake_word.models` list.

### Model Manifest (`hey_nova.json`)

```json
{
  "type": "micro",
  "wake_word": "hey nova",
  "author": "nick",
  "website": "https://www.home-assistant.io/voice-pe/",
  "model": "https://raw.githubusercontent.com/zenoran/hey-nova-voice-pe-model/master/hey_nova.tflite",
  "trained_languages": ["en"],
  "version": 2,
  "micro": {
    "probability_cutoff": 0.5,
    "sliding_window_size": 5,
    "feature_step_size": 10,
    "tensor_arena_size": 70000,
    "minimum_esphome_version": "2024.7"
  }
}
```

Tuning:
- **probability_cutoff**: Lower = more sensitive (more false positives). Higher = less sensitive. 0.5 is default.
- **sliding_window_size**: Number of consecutive positive predictions needed. Higher = fewer false positives but slower response.
- **tensor_arena_size**: Memory allocation on ESP32. Increase to 90000 if model fails to load.

---

## Patches Required for microWakeWord

These fix NumPy 2.x / TensorFlow 2.20 compatibility:

### `microwakeword/train.py`
1. `.numpy()` calls on values that are already numpy arrays — wrap with `isinstance` check
2. `np.trapz` → `np.trapezoid` (removed in NumPy 2.0)

### `microwakeword/test.py`
1. Same `np.trapz` → `np.trapezoid` fix
2. Same `.numpy()` safety check on model predictions

---

## ESPHome Compilation Patches

These must be applied every time `--only-generate` is run:

1. **`platformio.ini`**: Remove `esp_http_client` from EXCLUDE_COMPONENTS
   ```bash
   sed -i "s/esp_http_client;//g" /config/.esphome/build/DEVICE_NAME/platformio.ini
   ```

2. **Timer API** (ESPHome 2026.2+ with Voice PE packages): Timers changed from `std::map` to `std::vector<Timer>`
   ```bash
   YAML=/config/.esphome/packages/321a0194/home-assistant-voice.yaml
   sed -i "s/timers.begin()->second/timers.front()/g" $YAML
   sed -i "s/iterable_timer\.second\./iterable_timer./g" $YAML
   sed -i "s/iterable_timer\.second;/iterable_timer;/g" $YAML
   ```
   Note: This patch persists in the cached package file. Only needs to be applied once unless the package cache is cleared.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| GPU crash (CUDA_ERROR_INVALID_PTX) | `export CUDA_VISIBLE_DEVICES=""` — RTX 5080 compute 12.0 not supported by TF 2.20 |
| `np.trapz` AttributeError | Replace with `np.trapezoid` in train.py and test.py |
| `.numpy()` AttributeError | Values are already numpy arrays in TF 2.20, add isinstance check |
| `esp_http_client.h` not found | Remove `esp_http_client;` from platformio.ini EXCLUDE_COMPONENTS |
| Timer `.second` compile error | Apply timer API patch to cached Voice PE package YAML |
| GCC segfault during compile | Use `platformio run -j 1` (single-threaded) |
| firmware.bin path mismatch | Copy from `.pio/build/` to `.pioenvs/` before upload |
| Wake word not detecting | If model was trained with openWakeWord — retrain with microWakeWord. If properly trained, lower probability_cutoff |
| Model won't load on device | Increase tensor_arena_size in hey_nova.json |
| SSH to Unraid | Use RSA key: `ssh -i ~/.ssh/id_rsa root@10.0.0.99` (ed25519 not authorized) |

---

## File Locations

| What | Where |
|------|-------|
| microWakeWord training repo | `/home/nick/dev/microWakeWord/` |
| Training venv | `/home/nick/dev/microWakeWord/.venv` |
| Training script | `/home/nick/dev/microWakeWord/train_hey_nova.py` |
| Generated TTS samples | `/home/nick/dev/microWakeWord/generated_samples/` (2000 WAVs) |
| MIT RIRs | `/home/nick/dev/microWakeWord/mit_rirs/` (270 files) |
| FMA background noise | `/home/nick/dev/microWakeWord/fma_16k/` (210 files) |
| Negative datasets | `/home/nick/dev/microWakeWord/negative_datasets/` (~3GB) |
| Trained model output | `/home/nick/dev/microWakeWord/trained_models/wakeword/tflite_stream_state_internal_quant/` |
| GitHub model repo (local) | `/home/nick/dev/hey-nova-voice-pe-model/` |
| ESPHome configs (Unraid) | `/mnt/user/appdata/esphome/` |
| ESPHome secrets (Unraid) | `/mnt/user/appdata/esphome/secrets.yaml` |
| Piper sample generator | `/home/nick/dev/microWakeWord/piper-sample-generator/` |
| Piper model | `piper-sample-generator/en_US-libritts_r-medium.pt` |
