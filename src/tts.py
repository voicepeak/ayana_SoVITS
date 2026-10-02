# -*- coding: utf-8 -*-
"""CLI: text -> wav with the fine-tuned Ayana voice.

Examples:
  python tts.py --text "こんにちは、ユキト君。" --out out.wav
  python tts.py --text-file script.txt --out out.wav --speed 1.05
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import soundfile as sf
import ayana_tts

p = argparse.ArgumentParser()
p.add_argument("--text", default=None)
p.add_argument("--text-file", default=None)
p.add_argument("--out", default="out.wav")
p.add_argument("--language", default="日文",
               choices=["日文", "中文", "英文", "多语种混合"])
p.add_argument("--speed", type=float, default=1.0)
p.add_argument("--temperature", type=float, default=0.6)
p.add_argument("--max-chars", type=int, default=24, help="chunk size, prevents truncation")
p.add_argument("--no-lowpass", action="store_true", help="disable 15kHz low-pass")
a = p.parse_args()

if a.text_file:
    text = open(a.text_file, encoding="utf-8").read()
elif a.text:
    text = a.text
else:
    text = "こんにちは、ユキト君。今日もいい天気だね。"

t0 = time.time()
sr, audio = ayana_tts.synth_long(
    text, language=a.language, speed=a.speed, temperature=a.temperature,
    max_chars=a.max_chars, cut=(16000 if a.no_lowpass else 15000), log=True)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
sf.write(a.out, audio, sr)
dur = len(audio) / sr if sr else 0
print("saved %s  %.2fs  (%.1fs wall, RTF %.2f)" %
      (a.out, dur, time.time() - t0, (time.time() - t0) / max(0.01, dur)))
