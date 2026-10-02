# ayana_SoVITS

Japanese voice clone of **Otonashi Ayana** fine-tuned with
[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) **v2Pro**, runnable on **CPU**.

> Weights: <https://huggingface.co/Vociepeak/ayana_SoVITS>

**Japanese only.** English / Chinese are cross-lingual and unreliable (the model
was fine-tuned on Japanese only).

## Features
- Fine-tuned on ~43 min of clean audio
- Long text is split into short chunks and concatenated → no truncated endings
  (GPT-SoVITS otherwise stops at `max_sec` or on a premature EOS)
- Optional 15 kHz low-pass to remove the vocoder's high-frequency hiss
- CPU-friendly (RTF ≈ 0.7–1.0)

## Install
1. Clone GPT-SoVITS and install its deps + `pretrained_models`.
2. Put the fine-tuned weights in place:
   `GPT_weights_v2Pro/ayana-e14.ckpt`, `SoVITS_weights_v2Pro/ayana_e8_s672.pth`
3. `pip install -r requirements.txt`
4. Point the scripts at GPT-SoVITS via `AYANA_GSV_ROOT` (or place this repo next to it).

## Usage
```bash
python src/tts.py --text "こんにちは、ユキト君。" --out hello.wav
python src/tts.py --text-file script.txt --out out.wav --speed 1.05
```
Key settings: `temperature=0.6, top_p=0.6, top_k=20`, `lowpass 15kHz`,
chunked synthesis for long text.

## Disclaimer
Personal / research use only. The character and its voice belong to the original
rights holders. Do not use commercially, for impersonation, or redistribute the
original game assets.

## License
Code: MIT. Weights: research use only.
