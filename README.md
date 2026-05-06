# Audio Profanity Filter

实时音频违禁词过滤器 —— **HuBERT 声学特征 + CNN**，不依赖语音识别 (ASR)。

## 原理

```
麦克风 → 延迟线 → 输出 (虚拟声卡)
              │
              ├── 分析窗口 (0.5s) → HuBERT (768维) → CNN → 概率
              │                                              │
              └── 检测到违禁词 → 替换为自定义音频 ────────────┘
```

1. 音频流入延迟线（1 秒缓冲）
2. 0.5 秒窗口、0.1 秒步长滑动，送 **HuBERT** 提取 768 维声学特征
3. 1D CNN 逐窗口二分类，连续 3 窗阳性触发替换
4. 延迟线对应位置覆盖为鸟叫/哔声等自定义音频

**不转文字，直接分析声学特征**，跨语种通用。

## 安装

```bash
pip install torch transformers sounddevice soundfile pydub numpy
```

首次运行需要下载 HuBERT 模型（`facebook/hubert-base-ls960`，~360MB），运行库中_download_hubert.py即自动下载（release版无需下载）。

## 快速开始

### 启动
```bash
start.bat
# 或 python -u gui.py
```

### 录样本 & 训练
```bash
python tools.py
```
选 1 录样本，选 2 训练。全程命令行交互。

### 配置 `config.json`
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `window_sec` | 分析窗口长度 | 0.5 |
| `hop_sec` | 滑动步长 | 0.1 |
| `min_positive_windows` | 连续阳性窗口数 | 3 |
| `output_delay_sec` | 输出延迟 | 1.0 |
| `threshold_override` | 手动阈值 | 0.7 |
| `replacement_file` | 替换音频 | `replacements/bird.wav` |
| `min_energy` | 能量门限 | 0.003 |

## 替换音频

自定义音频放 `replacements/` 目录，支持 WAV/FLAC/MP3，自动重采样 16kHz 单声道。

## 声卡设置

- **输入**: 直通麦克风（不要用带降噪的，如 NVIDIA Broadcast）
- **输出**: [VB-Audio Virtual Cable](https://vb-audio.com/Cable/)
- 在 Discord/游戏/直播里把麦克风选为 `CABLE Output`

## 模型

```
音频 → HuBERT (7层卷积 + 12层 Transformer) → 768维
     → Conv1d(768→256→128→64→1) → Sigmoid
```

## License

MIT
