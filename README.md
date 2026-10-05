# ayana_SoVITS

用 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) **v2Pro** 微调的「音無彩名（おとなし彩名 / Otonashi Ayana）」日语语音克隆，支持 **本地 CPU** 推理。

> 微调权重托管在 Hugging Face：<https://huggingface.co/Vociepeak/ayana_SoVITS>

**语言范围：仅日语可靠。** 中文 / 英文属跨语言合成，模型未见过，发音与断句不稳定，不推荐。

---

## 特性
- 少量语料（约 43 分钟）微调即可获得较贴的音色
- 推理脚本针对**长文本**做了**分块合成再拼接**，避免 GPT-SoVITS 的 `max_sec` / 提前 EOS 导致的**结尾吞字**
- 内置 **15kHz 低通**后处理，压掉声码器引入的高频"电音"
- 纯 CPU 可跑（RTF ≈ 0.7–1.0），无需显卡

## 目录
```
ayana_SoVITS/
├─ src/
│  ├─ ayana_tts.py      # 可复用模块：synth() / synth_long()
│  └─ tts.py            # 命令行：文本 -> wav（长文自动分块）
├─ refs/
│  ├─ ref0308.wav       # 推理参考音频（决定语气）
│  └─ ref0308.txt       # 参考音频对应的文本
├─ requirements.txt
└─ README.md
```

## 安装

1) **准备 GPT-SoVITS 与底模**（本仓库不含引擎）
```bash
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS
# 按其 README 安装依赖 + 下载 pretrained_models（hubert / v2Pro 等）
```

2) **下载微调权重**放到 GPT-SoVITS 根目录
```
GPT-SoVITS/GPT_weights_v2Pro/ayana-e14.ckpt
GPT-SoVITS/SoVITS_weights_v2Pro/ayana_e8_s672.pth
```
（从 Hugging Face 下载，见下）

3) **安装依赖（先选 torch：GPU 还是 CPU）**
```bash
# GPU（推荐，如 RTX 4090/30 系）——先装 CUDA 版 torch
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
# 纯 CPU 则用：
# pip install torch==2.5.1 torchaudio==2.5.1
pip install -r requirements.txt
```
> ⚠️ Windows 上直接 `pip install torch`（走 PyPI）拿到的是 **CPU 版**；要用显卡必须按上面的 CUDA index 安装。

4) **告诉脚本 GPT-SoVITS 在哪**（三选一）
```bash
# Linux/macOS
export AYANA_GSV_ROOT=/path/to/GPT-SoVITS
# Windows PowerShell
$env:AYANA_GSV_ROOT = "D:\path\to\GPT-SoVITS"
```
或把本仓库放到 GPT-SoVITS 同级目录（脚本会自动找 `../GPT-SoVITS`）。

## 用法
```bash
python src/tts.py --text "こんにちは、ユキト君。今日もいい天気だね。" --out hello.wav
python src/tts.py --text-file script.txt --out out.wav --speed 1.05
```
参数：`--language 日文`、`--speed`、`--temperature 0.6`、`--max-chars 24`、`--no-lowpass`。

作为库调用：
```python
import ayana_tts
sr, audio = ayana_tts.synth_long("長い文章……", language="日文")
```

## 从 Hugging Face 下载权重
```bash
pip install -U huggingface_hub
huggingface-cli download Vociepeak/ayana_SoVITS --local-dir hf_ayana
# 把 .ckpt/.pth 分别放进 GPT_weights_v2Pro/ 与 SoVITS_weights_v2Pro/
```

## 关键设置
- **参考音频** `refs/ref0308.wav`：语气、音色锚点由它决定，换它即换语气。
- `temperature=0.6, top_p=0.6, top_k=20`（GPT-SoVITS 默认；调高会随机提前结束、吞掉结尾字）。
- `lowpass 15kHz`：去除高频电音；若嫌闷可 `--no-lowpass`。
- 长文本请用 `tts.py` / `synth_long()`：分块合成再拼接，防止截断。

## 设备与性能（GPU / CPU）
GPT-SoVITS **自动检测设备**：有 N 卡就用 GPU（fp16），否则 CPU——脚本无需改动。

| 设备 | RTF（合成秒 / 音频秒） | 5s 语音耗时 |
|---|---|---|
| **RTX 4090** | **≈ 0.12** | ~0.6s |
| CPU（Ultra 7 265K，20 核） | ≈ 0.7–1.0 | ~3.5–5s |

- 查看当前设备与实测：`python src/benchmark.py`
- **强制 CPU**：设置环境变量 `AYANA_DEVICE=cpu`（或 `CUDA_VISIBLE_DEVICES=""`）
- 实时对话 → 用 GPU；离线 / 批量 → CPU 亦可

```bash
python src/benchmark.py
# [ayana] device=cuda (NVIDIA GeForce RTX 4090)
#   #1 audio=3.86s synth=0.47s rtf=0.121 ...
```

## 训练数据（简述）
- 源素材：约 65 分钟日语单人语音
- 处理：Whisper large-v3 **词级时间戳** → 按词/句边界切片、去掉句内长静音 → 得 732 条干净切片 → 精选 490 条（句长 2–10s、语速正常）
- 切片首尾留余量（0.12s/0.16s），避免切掉尾音
- 微调：GPT-SoVITS v2Pro，SoVITS 8 epoch + GPT 20 epoch

## 致谢
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)（RVC-Boss）
- 文本前端、声码器等见 GPT-SoVITS 上游依赖

## ⚠️ 免责声明
本项目仅用于**个人学习与研究**。角色「音無彩名（おとなし彩名）」及其声优的音色、原游戏素材均**版权归原作者/发行方所有**。请勿用于**商业用途、冒充他人或任何违法用途**；请勿公开传播原始游戏素材。若权利人提出异议，将立即删除相关内容。

## License
代码以 MIT 许可发布；模型权重仅供研究，权属与使用限制见上。
