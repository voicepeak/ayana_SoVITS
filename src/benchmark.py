# -*- coding: utf-8 -*-
"""Quick benchmark: shows the compute device and per-sentence RTF.

  python benchmark.py                 # CPU/GPU auto
  AYANA_DEVICE=cpu python benchmark.py

RTF = synthesis seconds / audio seconds  (lower is faster; <1 = faster than realtime).
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import ayana_tts

SENTENCES = [
    "おはよう、ユキト君。今日もいい天気だね。",
    "意味なんてない。あるがままにあるものに、意味を与える必要なんてない。",
    "私はこの世界で、みなかみユキト君に出会えたことを心から感謝してる。",
]


def main():
    t0 = time.time()
    ayana_tts.load()
    print("model_load = %.2fs" % (time.time() - t0))
    total_a = total_s = 0.0
    for i, s in enumerate(SENTENCES, 1):
        t = time.time()
        sr, a = ayana_tts.synth(s)
        el = time.time() - t
        dur = len(a) / sr if sr else 0
        total_a += dur
        total_s += el
        print("  #%d audio=%.2fs synth=%.2fs rtf=%.3f" % (i, dur, el, el / dur if dur else 0))
    print("TOTAL audio=%.2fs synth=%.2fs  RTF=%.3f" %
          (total_a, total_s, total_s / total_a if total_a else 0))


if __name__ == "__main__":
    main()
