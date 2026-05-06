"""HuBERT 特征提取 — 本地模型"""
import numpy as np
import torch
from pathlib import Path
from transformers import HubertModel, Wav2Vec2FeatureExtractor

MODEL_DIR = Path(__file__).parent / "hubert_model_files"
_processor = None
_model = None

def _load():
    global _processor, _model
    if _model is None:
        if not (MODEL_DIR / "pytorch_model.bin").exists():
            raise FileNotFoundError(
                f"HuBERT model not found at {MODEL_DIR}\n"
                f"Please run: python _download_hubert.py"
            )
        _processor = Wav2Vec2FeatureExtractor.from_pretrained(str(MODEL_DIR))
        _model = HubertModel.from_pretrained(str(MODEL_DIR))
        if torch.cuda.is_available():
            _model = _model.cuda()
        _model.eval()

def extract(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """
    提取 HuBERT 768 维特征
    audio: float32 mono
    返回: (frames, 768)
    """
    _load()
    inputs = _processor(audio, sampling_rate=sample_rate, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = _model(**inputs, output_hidden_states=True)
        feats = outputs.hidden_states[9].squeeze(0).cpu().numpy()
    return feats.astype(np.float32)
