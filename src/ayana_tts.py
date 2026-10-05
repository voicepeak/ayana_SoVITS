# -*- coding: utf-8 -*-
"""Ayana (Otonashi Ayana) voice — local TTS helper on top of GPT-SoVITS.

Cross-platform, CPU-friendly. Long text is split into short chunks and
concatenated, so the model never truncates the tail (GPT-SoVITS stops at
max_sec or on a premature EOS).

Requires a GPT-SoVITS checkout (https://github.com/RVC-Boss/GPT-SoVITS) and the
fine-tuned weights from Hugging Face (see README).

Locate GPT-SoVITS via (in order):
  1. env  AYANA_GSV_ROOT   (or GSV_ROOT)
  2. sibling folder  ../GPT-SoVITS  (relative to this file)
  3. ./GPT-SoVITS  (cwd)

Override weights / reference via env:
  AYANA_GPT      (default GPT_weights_v2Pro/ayana-e14.ckpt)
  AYANA_SOVITS   (default SoVITS_weights_v2Pro/ayana_e8_s672.pth)
  AYANA_REF      (default <repo>/refs/ref0308.wav)
  AYANA_REF_TEXT (default the phrase shipped with ref0308.wav)
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def _find_gsv():
    env = os.environ.get("AYANA_GSV_ROOT") or os.environ.get("GSV_ROOT")
    if env and os.path.isdir(os.path.join(env, "GPT_SoVITS")):
        return os.path.abspath(env)
    cands = [
        os.path.join(REPO, "GPT-SoVITS"),
        os.path.join(REPO, "..", "GPT-SoVITS"),
        os.path.join(HERE, "GPT-SoVITS"),
        "GPT-SoVITS",
    ]
    for c in cands:
        if os.path.isdir(os.path.join(c, "GPT_SoVITS")):
            return os.path.abspath(c)
    return None


GSV_ROOT = _find_gsv()
if GSV_ROOT is None:
    raise EnvironmentError(
        "GPT-SoVITS not found. Clone it and set env AYANA_GSV_ROOT to its path.")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
# Device: GPT-SoVITS auto-uses CUDA when available. Force CPU with AYANA_DEVICE=cpu.
if os.environ.get("AYANA_DEVICE", "").strip().lower() == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, GSV_ROOT)
sys.path.insert(0, os.path.join(GSV_ROOT, "GPT_SoVITS"))
_orig_cwd = os.getcwd()
os.chdir(GSV_ROOT)  # GPT-SoVITS resolves model paths relative to its root

import numpy as np
from scipy.signal import butter, sosfiltfilt
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.inference_webui import (
    change_gpt_weights, change_sovits_weights, get_tts_wav)

i18n = I18nAuto()

GPT = os.environ.get("AYANA_GPT", "GPT_weights_v2Pro/ayana-e14.ckpt")
SOVITS = os.environ.get("AYANA_SOVITS", "SoVITS_weights_v2Pro/ayana_e8_s672.pth")
REF = os.environ.get("AYANA_REF", os.path.join(REPO, "refs", "ref0308.wav"))


def _ref_text():
    t = os.environ.get("AYANA_REF_TEXT")
    if t:
        return t
    companion = os.path.splitext(REF)[0] + ".txt"   # e.g. refs/xxx.wav -> refs/xxx.txt
    if os.path.isfile(companion):
        return open(companion, encoding="utf-8").read().strip()
    return "私を芸能界デビューさせたいという人がいるのでしょう"


REFTEXT = _ref_text()

_loaded = False


def device_info():
    """Return a short string describing the compute device in use."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda (%s)" % torch.cuda.get_device_name(0)
        return "cpu"
    except Exception:
        return "unknown"


def load():
    """Load the fine-tuned GPT + SoVITS weights (idempotent)."""
    global _loaded
    if not _loaded:
        change_gpt_weights(gpt_path=GPT)
        change_sovits_weights(sovits_path=SOVITS)
        _loaded = True
        print("[ayana] device=%s" % device_info(), flush=True)


def _to_float(a):
    a = np.asarray(a)
    if a.dtype == np.int16:
        return a.astype(np.float32) / 32768.0
    if a.dtype == np.int32:
        return a.astype(np.float32) / 2147483648.0
    a = a.astype(np.float32)
    m = float(np.abs(a).max()) if a.size else 0.0
    return a / m if m > 1.5 else a


def _lowpass(x, sr, cut):
    ny = sr / 2.0
    if cut >= ny or x.size == 0:
        return x.astype(np.float32)
    sos = butter(6, cut / ny, btype="low", output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def synth(text, language="日文", speed=1.0, temperature=0.6,
          top_k=20, top_p=0.6, cut=15000):
    """One short sentence -> (sr, float32 audio)."""
    load()
    gen = list(get_tts_wav(
        ref_wav_path=REF, prompt_text=REFTEXT, prompt_language=i18n("日文"),
        text=text, text_language=i18n(language),
        top_k=top_k, top_p=top_p, temperature=temperature, speed=speed))
    sr, a = gen[-1]
    return sr, _lowpass(_to_float(a), sr, cut)


def split_chunks(text, max_chars=24):
    text = re.sub(r"\s+", "", text)
    parts = re.split(r"(?<=[。！？!?；;：:\n])", text)
    parts = [p for p in parts if p.strip()]
    out = []
    for p in parts:
        if len(p) <= max_chars:
            out.append(p)
            continue
        buf = ""
        for s in re.split(r"(?<=[，,、])", p):
            if len(buf) + len(s) <= max_chars:
                buf += s
            else:
                if buf:
                    out.append(buf)
                    buf = ""
                while len(s) > max_chars:
                    out.append(s[:max_chars])
                    s = s[max_chars:]
                buf = s
        if buf:
            out.append(buf)
    return out


def synth_long(text, language="日文", speed=1.0, temperature=0.6,
               max_chars=24, cut=15000, gap=0.06, log=False):
    """Long text -> (sr, audio), synthesized chunk-by-chunk (no truncation)."""
    load()
    pieces = []
    sr = 32000
    for c in split_chunks(text, max_chars):
        try:
            sr, a = synth(c, language=language, speed=speed,
                          temperature=temperature, cut=cut)
            pieces.append(a)
            if log:
                print("  chunk ok:", c, flush=True)
        except Exception as e:  # keep going; a single bad chunk shouldn't abort
            if log:
                print("  chunk FAIL:", c, e, flush=True)
    if not pieces:
        return sr, np.zeros(0, np.float32)
    sil = np.zeros(int(sr * gap), np.float32)
    merged = []
    for k, a in enumerate(pieces):
        if k:
            merged.append(sil)
        merged.append(a)
    return sr, np.concatenate(merged).astype(np.float32)
