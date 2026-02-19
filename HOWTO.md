# "Hey Nova" Custom Wake Word — Full Build & Deploy Guide

Everything done to create a custom **"hey nova"** wake word for the **Home Assistant Voice Preview Edition (Voice PE)** using **microWakeWord**, and deploy it via **ESPHome** on an **Unraid** Docker setup.

> Date completed: **2026-02-18**

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Phase 1 — Generate Synthetic Audio Clips (openWakeWord)](#phase-1--generate-synthetic-audio-clips-openwakeword)
4. [Phase 2 — Train microWakeWord Model](#phase-2--train-microwakeword-model)
5. [Phase 3 — Publish Model to GitHub](#phase-3--publish-model-to-github)
6. [Phase 4 — Deploy ESPHome on Unraid](#phase-4--deploy-esphome-on-unraid)
7. [Phase 5 — Flash Voice PE](#phase-5--flash-voice-pe)
8. [File Inventory](#file-inventory)
9. [Patches Applied to microWakeWord](#patches-applied-to-microwakeword)
10. [Troubleshooting](#troubleshooting)
11. [Re-training from Scratch](#re-training-from-scratch)

---

## Overview

```
openWakeWord (piper TTS)          microWakeWord (TF training)        GitHub + ESPHome
┌──────────────────────┐     ┌──────────────────────────────┐    ┌────────────────────┐
│ Generate ~110k WAV   │────▶│ Augment → Spectrograms →     │───▶│ hey_nova.tflite     │
│ positive + negative  │     │ Train MixConv → Quantize →   │    │ hey_nova.json       │
│ clips via Piper TTS  │     │ stream_state_internal_quant  │    │ → Voice PE OTA      │
└──────────────────────┘     └──────────────────────────────┘    └────────────────────┘
```

---

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.11 | microWakeWord requires 3.10+ |
| uv | latest | Used for venv + pip |
| TensorFlow | 2.20.0 | CPU mode (GPU had PTX issues on RTX 50-series) |
| openWakeWord | local clone | `/home/nick/dev/openWakeWord` |
| microWakeWord | OHF-Voice fork | `/home/nick/dev/micro-wake-word` |
| Unraid | — | Docker host running Home Assistant |
| Home Assistant | container | `ghcr.io/home-assistant/home-assistant:latest` on Unraid |
| ESPHome | container | `ghcr.io/esphome/esphome:latest` on Unraid |
| GitHub CLI (gh) | authenticated | Account: `zenoran` |

---

## Phase 1 — Generate Synthetic Audio Clips (openWakeWord)

This phase was done **before** this session using the openWakeWord automatic training notebook. The output was ~110k WAV clips already on disk.

### Source clips location

```
/home/nick/dev/openWakeWord/my_custom_model/my_model/
├── positive_train/    # ~1,500 "hey nova" WAV clips (Piper TTS generated)
├── positive_test/     # held-out positive clips
├── negative_train/    # ~100k+ negative clips (random speech, noise)
└── negative_test/     # held-out negative clips
```

All clips are **16 kHz mono PCM WAV**.

### How these were originally generated

Using `openwakeword/train.py` and the Piper sample generator:
- Positive: Piper TTS with phonetic spelling `hey_nova` / `hey nova`, multiple voices, noise scales
- Negative: AudioSet, FMA, LibriSpeech, Common Voice subsets

---

## Phase 2 — Train microWakeWord Model

### 2.1 Clone and set up environment

```bash
cd /home/nick/dev
git clone https://github.com/OHF-Voice/micro-wake-word.git
cd micro-wake-word

# Python 3.11 venv
uv python install 3.11
uv venv .venv311 --python 3.11
source .venv311/bin/activate

# Install microWakeWord + dependencies
uv pip install -e .
uv pip install torchcodec torch torchaudio
```

### 2.2 Patches applied to microWakeWord

Three compatibility fixes were needed for NumPy 2.x / TensorFlow 2.20:

**`microwakeword/train.py`** — two fixes:
1. `result["fp"].numpy()` → safe fallback for when TF returns plain ndarray:
   ```python
   def _to_numpy(x):
       return x.numpy() if hasattr(x, "numpy") else np.asarray(x)
   ```
   Applied to `result["fp"]`, `ambient_predictions["tp"]`, `["fp"]`, `["fn"]`.

2. `np.trapz` → `np.trapezoid` (removed in NumPy 2.0):
   ```python
   integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
   ```

**`microwakeword/test.py`** — same `np.trapz` fix in `tflite_streaming_model_roc()`.

### 2.3 Data preparation script

**`local_scripts/prepare_hey_nova_data.py`**

Reads existing clips from openWakeWord output, applies augmentation (EQ, distortion, pitch shift, color noise, gain), generates spectrogram features as RaggedMmap, and writes a training YAML config.

Key settings:
- Augmentation duration: 3.2s
- Positive clips: split 90/5/5 train/val/test, repeated 2x for training
- Negative clips: split 90/5/5, repeated 2x for training
- Ambient negatives: from `negative_test/` directory (unaugmented)

### 2.4 Training config

**`local_data/hey_nova/training_parameters_hey_nova.yaml`**:
```yaml
batch_size: 128
clip_duration_ms: 1500
training_steps: [4000]
learning_rates: [0.001]
positive_class_weight: [1]
negative_class_weight: [20]
eval_step_interval: 250
maximization_metric: average_viable_recall
minimization_metric: ambient_false_positives_per_hour
target_minimization: 1.2
window_step_ms: 10
```

Feature sets:
- Wakeword (positive): `sampling_weight: 3.0`, `truncation_strategy: truncate_start`
- Non-wake (negative): `sampling_weight: 8.0`, `truncation_strategy: random`

### 2.5 Training command

**`local_scripts/train_hey_nova.sh`**:
```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv311/bin/activate
export CUDA_VISIBLE_DEVICES=-1   # Force CPU (RTX 50-series PTX incompatible)

python local_scripts/prepare_hey_nova_data.py

python -m microwakeword.model_train_eval \
  --training_config='local_data/hey_nova/training_parameters_hey_nova.yaml' \
  --train 1 \
  --restore_checkpoint 1 \
  --test_tflite_streaming_quantized 1 \
  --use_weights best_weights \
  mixednet \
  --pointwise_filters '64,64,64,64' \
  --repeat_in_block '1,1,1,1' \
  --mixconv_kernel_sizes '[5],[7,11],[9,15],[23]' \
  --residual_connection '0,0,0,0' \
  --first_conv_filters 32 \
  --first_conv_kernel_size 5 \
  --stride 3
```

### 2.6 Training results

- Final validation: **Accuracy 0.999, Recall 0.998, Precision 0.997**
- Output model: 61 KB quantized streaming TFLite
- Model path: `local_data/hey_nova/trained_models/hey_nova/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite`

> **Note:** These metrics are on synthetic data. Real-world performance will differ — the model may need threshold tuning or retraining with more diverse samples.

---

## Phase 3 — Publish Model to GitHub

### 3.1 Create repo and push

```bash
mkdir -p /home/nick/dev/hey-nova-voice-pe-model
cp hey_nova.tflite hey_nova.json voice_pe_micro_wake_word_snippet.yaml \
   /home/nick/dev/hey-nova-voice-pe-model/

cd /home/nick/dev/hey-nova-voice-pe-model
git init && git add . && git commit -m 'Add hey nova microWakeWord model artifacts'
gh repo create zenoran/hey-nova-voice-pe-model --public --source=. --remote=origin --push
```

### 3.2 Model manifest (`hey_nova.json`)

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
    "probability_cutoff": 0.97,
    "sliding_window_size": 5,
    "feature_step_size": 10,
    "tensor_arena_size": 70000,
    "minimum_esphome_version": "2024.7"
  }
}
```

### 3.3 Published URLs

- Repo: https://github.com/zenoran/hey-nova-voice-pe-model
- JSON manifest: `https://raw.githubusercontent.com/zenoran/hey-nova-voice-pe-model/master/hey_nova.json`
- TFLite model: `https://raw.githubusercontent.com/zenoran/hey-nova-voice-pe-model/master/hey_nova.tflite`

### 3.4 Also copied to Unraid HA filesystem

```bash
scp hey_nova.json hey_nova.tflite root@unraid:/mnt/user/appdata/hass/
```

---

## Phase 4 — Deploy ESPHome on Unraid

Since HA runs as a plain Docker container on Unraid (no Add-on Store), ESPHome was deployed as a separate container.

### Launch command (already run)

```bash
ssh root@unraid 'docker run -d \
  --name esphome \
  --restart unless-stopped \
  -p 6052:6052 \
  -p 6123:6123/udp \
  -v /mnt/user/appdata/esphome:/config \
  --network bridge \
  ghcr.io/esphome/esphome:latest'
```

### Access

- ESPHome dashboard: `http://<unraid-ip>:6052`
- Config persistence: `/mnt/user/appdata/esphome` on Unraid
- Restart policy: `unless-stopped` (survives reboots)

### Connect to Home Assistant

In Home Assistant:
1. **Settings → Devices & Services → Add Integration → ESPHome**
2. Host: `<unraid-ip>`, Port: `6052`

---

## Phase 5 — Flash Voice PE

### 5.1 Adopt Voice PE in ESPHome

1. Open ESPHome dashboard at `http://<unraid-ip>:6052`
2. Voice PE should appear as a discovered device
3. Click **Adopt** → this creates an editable YAML config

### 5.2 Add custom wake word to YAML

In the Voice PE's ESPHome YAML config, add:

```yaml
micro_wake_word:
  models:
    - model: https://raw.githubusercontent.com/zenoran/hey-nova-voice-pe-model/master/hey_nova.json
      id: hey_nova_model
  vad:
    model: github://esphome/micro-wake-word-models/models/v2/vad.json@main

voice_assistant:
  on_wake_word_detected:
    then:
      - voice_assistant.start:
          wake_word: !lambda return wake_word;
```

### 5.3 Install OTA

Click **Install** in ESPHome dashboard → **Wirelessly** (OTA).

### 5.4 Troubleshooting model load

If Voice PE fails to load the model, increase `tensor_arena_size` in `hey_nova.json`:
```json
"tensor_arena_size": 90000
```
Push to GitHub and reflash.

---

## File Inventory

| File | Location | Purpose |
|------|----------|---------|
| Positive WAV clips | `/home/nick/dev/openWakeWord/my_custom_model/my_model/positive_train/` | TTS-generated "hey nova" samples |
| Negative WAV clips | `/home/nick/dev/openWakeWord/my_custom_model/my_model/negative_train/` | Background/speech negatives |
| microWakeWord repo | `/home/nick/dev/micro-wake-word/` | Training framework (with patches) |
| Data prep script | `/home/nick/dev/micro-wake-word/local_scripts/prepare_hey_nova_data.py` | Augmentation + spectrogram generation |
| Training launcher | `/home/nick/dev/micro-wake-word/local_scripts/train_hey_nova.sh` | End-to-end training script |
| Training config | `/home/nick/dev/micro-wake-word/local_data/hey_nova/training_parameters_hey_nova.yaml` | Hyperparameters |
| Trained TFLite | `/home/nick/dev/micro-wake-word/local_data/hey_nova/trained_models/hey_nova/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite` | 61 KB quantized model |
| Published model repo | `/home/nick/dev/hey-nova-voice-pe-model/` | GitHub-hosted artifacts |
| ESPHome config | `/mnt/user/appdata/esphome/` (on Unraid) | Persistent ESPHome configs |
| HA model copies | `/mnt/user/appdata/hass/hey_nova.*` (on Unraid) | Backup copies on HA host |

---

## Patches Applied to microWakeWord

These are needed until upstream merges NumPy 2.x / TF 2.20 fixes.

### `microwakeword/train.py`

```diff
+ def _to_numpy(x):
+     return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

- test_set_fp = result["fp"].numpy()
+ test_set_fp = _to_numpy(result["fp"])

- all_true_positives = ambient_predictions["tp"].numpy()
- ambient_false_positives = ambient_predictions["fp"].numpy() - test_set_fp
- all_false_negatives = ambient_predictions["fn"].numpy()
+ all_true_positives = _to_numpy(ambient_predictions["tp"])
+ ambient_false_positives = _to_numpy(ambient_predictions["fp"]) - test_set_fp
+ all_false_negatives = _to_numpy(ambient_predictions["fn"])

- average_viable_recall = np.trapz(...) / 2.0
+ integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
+ average_viable_recall = integrate(...) / 2.0
```

### `microwakeword/test.py`

```diff
- auc = np.trapz(y_coordinates, x_coordinates)
+ integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
+ auc = integrate(y_coordinates, x_coordinates)
```

---

## Re-training from Scratch

If you need to retrain (e.g., with more samples or different hyperparameters):

```bash
cd /home/nick/dev/micro-wake-word
source .venv311/bin/activate

# Edit prepare_hey_nova_data.py to adjust augmentation, limits, etc.
# Edit training_parameters_hey_nova.yaml for hyperparameters

# Remove old training artifacts
rm -rf local_data/hey_nova/trained_models/hey_nova

# Run full pipeline
./local_scripts/train_hey_nova.sh

# Copy new model to publish repo
cp local_data/hey_nova/trained_models/hey_nova/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite \
   /home/nick/dev/hey-nova-voice-pe-model/hey_nova.tflite

# Push
cd /home/nick/dev/hey-nova-voice-pe-model
git add hey_nova.tflite && git commit -m 'Retrained model' && git push

# Reflash Voice PE from ESPHome dashboard (Install → OTA)
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| TF GPU crash (CUDA_ERROR_INVALID_PTX) | Set `CUDA_VISIBLE_DEVICES=-1` to force CPU training |
| `np.trapz` AttributeError | Apply patch above (NumPy 2.x removed `trapz`) |
| `result["fp"].numpy()` AttributeError | Apply `_to_numpy()` patch (TF 2.20 returns ndarray not Tensor) |
| `torchcodec` / `torch` missing | `uv pip install torchcodec torch torchaudio` in the mWW venv |
| Voice PE won't load model | Increase `tensor_arena_size` in `hey_nova.json` to `90000` |
| ESPHome container gone after reboot | Already set `--restart unless-stopped`; if Unraid clears it, re-run the `docker run` command from Phase 4 |
