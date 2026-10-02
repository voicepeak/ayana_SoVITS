---
language:
  - ja
library_name: gpt-sovits
tags:
  - text-to-speech
  - voice-cloning
  - japanese
  - gpt-sovits
  - v2pro
license: other
license_name: research-only
---

# ayana_SoVITS

Fine-tuned **GPT-SoVITS v2Pro** voice model for the Japanese voice of
**Otonashi Ayana**. For personal / research use only.

## Files
| file | what |
|---|---|
| `ayana-e14.ckpt` | GPT (prosody) weights — GPT-SoVITS v2Pro |
| `ayana_e8_s672.pth` | SoVITS (timbre) weights — v2Pro |

Place them in a GPT-SoVITS checkout as:
```
GPT_weights_v2Pro/ayana-e14.ckpt
SoVITS_weights_v2Pro/ayana_e8_s672.pth
```

## Usage
See the code repo: <https://github.com/voicepeak/ayana_SoVITS>

Minimal (inside a GPT-SoVITS environment):
```python
from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav
change_gpt_weights("GPT_weights_v2Pro/ayana-e14.ckpt")
change_sovits_weights("SoVITS_weights_v2Pro/ayana_e8_s672.pth")
gen = list(get_tts_wav(
    ref_wav_path="refs/ref0308.wav",
    prompt_text="私を芸能界デビューさせたいという人がいるのでしょう",
    prompt_language="日文",
    text="こんにちは、ユキト君。", text_language="日文",
    top_k=20, top_p=0.6, temperature=0.6))
sr, audio = gen[-1]
```

## Device
Runs on **GPU** (auto-detected, RTF ≈ 0.12 on an RTX 4090) or **CPU** (RTF ≈ 0.7–1.0).
Force CPU with `AYANA_DEVICE=cpu`.

## Recommended settings
- `temperature=0.6, top_p=0.6, top_k=20` (higher temperature causes premature EOS → dropped endings)
- Split long text into ≤24-char chunks and concatenate (prevents `max_sec` truncation)
- Optional 15 kHz low-pass to reduce vocoder high-frequency hiss
- A 3–10 s reference clip + its exact transcript is required; it sets the tone

## Training
- ~65 min single-speaker Japanese source → 732 clean clips → 490 curated (2–10 s)
- SoVITS 8 epochs, GPT 15 epochs, on a single RTX 4090

## Language
Japanese only. Other languages are cross-lingual and unreliable.

## Disclaimer
The character and its voice belong to the original rights holders. Do not use
for commercial purposes, impersonation, or redistribution of original assets.
Research / personal use only.
