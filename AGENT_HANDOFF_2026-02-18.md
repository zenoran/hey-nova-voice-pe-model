# Agent Handoff — Voice PE Custom Wake Word Project

Date: 2026-02-18
Prepared by: GitHub Copilot (GPT-5.3-Codex)

This file is a complete transition brief for a new agent to continue from the current state.

---

## 1) User goal

Deploy a **custom on-device wake word** (`hey nova`) on **Home Assistant Voice Preview Edition (Voice PE)**.

Constraints:
- Home Assistant runs as **Docker on Unraid** (no Add-on Store)
- ESPHome must run separately as container
- User wants practical, step-by-step guidance and minimal confusion

---

## 2) High-level outcome so far

✅ Custom microWakeWord model was trained and exported to TFLite
✅ Model manifest + TFLite were published to GitHub
✅ ESPHome container was installed and running on Unraid
✅ Voice PE full official YAML and fallback minimal YAML were prepared and uploaded to ESPHome config folder
✅ `hey_nova` model inserted into full official Voice PE YAML
✅ User provided Wi-Fi secrets

⚠️ Current blocker is firmware flashing workflow stability (UI says "Connecting" while server compiles in background; long compile times and prior toolchain instability observed)

---

## 3) Repositories and important paths

### Local machine
- `openWakeWord` repo: `/home/nick/dev/openWakeWord`
- `micro-wake-word` repo: `/home/nick/dev/micro-wake-word`
- Published model repo: `/home/nick/dev/hey-nova-voice-pe-model`

### Unraid host
- Home Assistant appdata: `/mnt/user/appdata/hass`
- ESPHome appdata/config: `/mnt/user/appdata/esphome`

### Key files
- Full official Voice PE config (patched):
  - `/mnt/user/appdata/esphome/home-assistant-voice.yaml`
- Minimal fallback Voice PE config:
  - `/mnt/user/appdata/esphome/voice-pe-minimal.yaml`
- ESPHome secrets:
  - `/mnt/user/appdata/esphome/secrets.yaml`

---

## 4) Model artifacts and URLs

### GitHub repo created
- https://github.com/zenoran/hey-nova-voice-pe-model

### Published files
- JSON manifest:
  - https://raw.githubusercontent.com/zenoran/hey-nova-voice-pe-model/master/hey_nova.json
- TFLite:
  - https://raw.githubusercontent.com/zenoran/hey-nova-voice-pe-model/master/hey_nova.tflite
- Snippet:
  - https://raw.githubusercontent.com/zenoran/hey-nova-voice-pe-model/master/voice_pe_micro_wake_word_snippet.yaml
- Existing documentation:
  - `/home/nick/dev/hey-nova-voice-pe-model/HOWTO.md`

---

## 5) Training details completed

### Data source reused
From previous openWakeWord generation:
- `/home/nick/dev/openWakeWord/my_custom_model/my_model/positive_train`
- `/home/nick/dev/openWakeWord/my_custom_model/my_model/negative_train`
- plus test splits

### microWakeWord env
- repo: `/home/nick/dev/micro-wake-word`
- venv: `.venv311`
- installed dependencies include `torchcodec`, `torch`, `torchaudio`

### Scripts created
- `/home/nick/dev/micro-wake-word/local_scripts/prepare_hey_nova_data.py`
- `/home/nick/dev/micro-wake-word/local_scripts/train_hey_nova.sh`

### Model output
- `/home/nick/dev/micro-wake-word/local_data/hey_nova/trained_models/hey_nova/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite`
- Copied as:
  - `/home/nick/dev/openWakeWord/voice_pe_model/hey_nova.tflite`
  - `/home/nick/dev/hey-nova-voice-pe-model/hey_nova.tflite`

---

## 6) Source patches made (important)

In local `micro-wake-word` repo, compatibility fixes were applied:

1. `microwakeword/train.py`
- Handle metrics that may be numpy arrays (not tensors)
- Replace `np.trapz` with fallback to `np.trapezoid` when available

2. `microwakeword/test.py`
- Replace `np.trapz` with `np.trapezoid` fallback

These were required for NumPy 2.x behavior.

---

## 7) Unraid + ESPHome setup completed

### Container status actions performed
- Installed ESPHome container (initially `latest`, then switched to `dev`)
- Current image: `ghcr.io/esphome/esphome:dev`
- Port mapping:
  - `6052/tcp`
  - `6123/udp`
- Volume:
  - `/mnt/user/appdata/esphome:/config`
- Restart policy:
  - `unless-stopped`

### Dashboard
- URL used by user: `http://10.0.0.99:6052`

---

## 8) Current flashing configs

### A) Full official YAML (patched)
`/mnt/user/appdata/esphome/home-assistant-voice.yaml`

Changes made:
- Added `hey_nova` model in `micro_wake_word.models`
- Added `hey_nova` cutoffs in wake sensitivity select lambda
- Added missing secret-based wifi fields:
  - `ssid: !secret wifi_ssid`
  - `password: !secret wifi_password`
- Added API key line:
  - `key: !secret api_encryption_key`

### B) Minimal fallback YAML (created)
`/mnt/user/appdata/esphome/voice-pe-minimal.yaml`

Purpose:
- Reduce complexity to get first successful flash
- Includes required Voice PE hardware blocks + `hey_nova` micro wake word

---

## 9) Secrets status

`/mnt/user/appdata/esphome/secrets.yaml`
contains keys:
- `wifi_ssid`
- `wifi_password`
- `api_encryption_key`

A valid base64 API key was auto-generated and written when previous placeholder failed validation.

---

## 10) What failed / known issues

### Earlier failures (resolved)
- Full YAML compile failed due missing wifi fields in config (resolved)
- API key format invalid (resolved)

### Ongoing risk
- Very long first compile times on Unraid (10–30+ mins)
- Browser flasher page (`web.esphome.io`) may show "Connecting" while backend is still building
- Previously saw toolchain instability (assembler segfault) in one path; fallback to minimal config and single-job compile was used while diagnosing

---

## 11) Last observed runtime status

From logs:
- `esphome --dashboard compile --only-generate /config/voice-pe-minimal.yaml` exited `0`
- `esphome --dashboard compile /config/voice-pe-minimal.yaml` launched from user action
- Firmware binary exists at least once:
  - `/config/.esphome/build/voice-pe-minimal/.pioenvs/voice-pe-minimal/firmware.bin` (seen as 1.4M)

User is interacting with the browser flasher and reported "Connecting".

---

## 12) Exactly what the next agent should do first

1. **Check compile still running and completion status**:
```bash
ssh root@unraid 'docker logs --tail 120 esphome 2>&1 | tail -120'
ssh root@unraid 'pgrep -af "esphome --dashboard compile /config/voice-pe-minimal.yaml" || true'
ssh root@unraid 'ls -lh /mnt/user/appdata/esphome/.esphome/build/voice-pe-minimal/.pioenvs/voice-pe-minimal/firmware.bin 2>/dev/null || echo no_firmware'
```

2. If compile finished, guide user to:
- Use ESPHome dashboard card `voice-pe-minimal.yaml` -> Install
- Prefer manual download + web.esphome.io if direct handoff stalls

3. If compile still failing, capture full error with:
```bash
ssh root@unraid 'docker exec esphome esphome compile /config/voice-pe-minimal.yaml'
```
and address that exact error only.

4. After first successful flash:
- Ensure device appears in Home Assistant ESPHome integration
- Verify wakeword by saying `hey nova`

5. Optional cleanup after success:
- Remove confusing extra project card or mark one canonical config

---

## 13) Commands history highlights (important replay items)

### Check HA container
```bash
ssh root@unraid 'docker ps -a --format "table {{.Names}}\t{{.Image}}\t{{.Status}}"'
```

### Deploy ESPHome container (used)
```bash
docker run -d \
  --name esphome \
  --restart unless-stopped \
  -p 6052:6052 \
  -p 6123:6123/udp \
  -v /mnt/user/appdata/esphome:/config \
  --network bridge \
  ghcr.io/esphome/esphome:dev
```

### Upload configs
```bash
scp /tmp/home-assistant-voice.yaml root@unraid:/mnt/user/appdata/esphome/home-assistant-voice.yaml
scp /tmp/voice-pe-minimal.yaml root@unraid:/mnt/user/appdata/esphome/voice-pe-minimal.yaml
```

### Validate compile errors
```bash
ssh root@unraid 'docker exec esphome esphome compile /config/home-assistant-voice.yaml'
ssh root@unraid 'docker exec esphome esphome compile /config/voice-pe-minimal.yaml'
```

---

## 14) User communication preferences observed

- User prefers blunt/simple instructions
- User asked: "speak like im an idiot"
- Best response style: short concrete steps, no jargon, one action at a time

---

## 15) Critical do/don’t for next agent

Do:
- Keep instructions short and procedural
- Validate with server-side logs before asking user to retry
- Favor `voice-pe-minimal.yaml` as first successful flash path

Don’t:
- Send user back to managed onboarding docs now
- Introduce more project files unless required
- Ask broad/abstract questions while user is mid-install

---

## 16) Current priority

Get **one successful first flash** of Voice PE using `voice-pe-minimal.yaml`, then verify `hey nova` wake word works.
