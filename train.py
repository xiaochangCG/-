"""训练违禁词检测模型 - HuBERT 特征"""
import sys, json, wave
import numpy as np
from pathlib import Path

SR = 16000
BASE = Path(__file__).parent
SAMPLES_DIR = BASE / 'samples'
PROFANITY_DIR = SAMPLES_DIR / 'profanity'
NORMAL_DIR = SAMPLES_DIR / 'normal'

def load_wav(p):
    with wave.open(str(p), 'rb') as wf:
        return np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768

def get_slices(audio, window_sr=8000, hop_sr=1600):
    """从完整音频中提取滑动窗口的 HuBERT 特征"""
    from hubert import extract
    slices = []
    for i in range(0, max(1, len(audio) - window_sr), hop_sr):
        seg = audio[i:i + window_sr]
        if len(seg) < window_sr // 2:
            break
        feats = extract(seg)
        if feats.shape[0] >= 3:
            slices.append(feats)
    if not slices and len(audio) >= window_sr // 2:
        feats = extract(audio[:window_sr])
        if feats.shape[0] >= 3:
            slices.append(feats)
    return slices

def main():
    cfg = json.load(open(BASE / 'config.json', encoding='utf-8')).get('train', {})
    epochs = cfg.get('epochs', 100)
    lr = cfg.get('lr', 0.001)

    for i, arg in enumerate(sys.argv):
        if arg == '--epochs' and i + 1 < len(sys.argv):
            epochs = int(sys.argv[i + 1])
        if arg == '--lr' and i + 1 < len(sys.argv):
            lr = float(sys.argv[i + 1])

    from classifier_torch_hubert import HuBERTClassifier

    print('=== Train (HuBERT) ===\n')

    # 加载样本
    all_pos = []
    for d in PROFANITY_DIR.iterdir():
        if not d.is_dir():
            continue
        fts = []
        for f in sorted(d.glob('*.wav')):
            try:
                audio = load_wav(f)
                for feat in get_slices(audio):
                    fts.append(feat)
            except Exception as e:
                print(f'  skip {f.name}: {e}')
        all_pos.extend(fts)
        print(f'  "{d.name}": {len(fts)} slices')

    all_neg = []
    if NORMAL_DIR.exists():
        fts = []
        for f in sorted(NORMAL_DIR.glob('*.wav')):
            try:
                audio = load_wav(f)
                for feat in get_slices(audio):
                    fts.append(feat)
            except Exception as e:
                print(f'  skip {f.name}: {e}')
        all_neg.extend(fts)
        print(f'  normal: {len(fts)} slices')

    # 平衡样本
    if len(all_neg) < len(all_pos) * 0.5:
        n_gen = max(int(len(all_pos) * 0.8 - len(all_neg)), 5)
        print(f'  generating {n_gen} synthetic negatives...')
        for _ in range(n_gen):
            all_neg.append(np.random.randn(np.random.randint(10, 50), 768) * 0.3)

    X = all_pos + all_neg
    y = np.concatenate([np.ones(len(all_pos)), np.zeros(len(all_neg))]).astype(np.float32)
    print(f'\nTotal: {len(X)} (pos={len(all_pos)} neg={len(all_neg)})')

    # 训练
    model = HuBERTClassifier()
    print(f'Training {epochs} epochs...\n')
    model.fit(X, y, epochs=epochs, lr=lr, batch_size=16, val_split=0.2)

    # 评估
    all_probs = np.array([float(model.predict_proba(f)) for f in X])
    preds = (all_probs > model.threshold).astype(int)
    acc = np.mean(preds == y)
    tp = np.sum((preds == 1) & (y == 1))
    fp = np.sum((preds == 1) & (y == 0))
    fn = np.sum((preds == 0) & (y == 1))
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)

    model_path = BASE / 'model_hubert.pth'
    model.save(str(model_path))

    print(f'\n=== Results ===')
    print(f'Accuracy:   {acc:.2%}')
    print(f'Precision:  {prec:.2%}')
    print(f'Recall:     {rec:.2%}')
    print(f'F1:         {f1:.4f}')
    print(f'Threshold:  {model.threshold:.2f}')
    print(f'Saved: {model_path}')

if __name__ == '__main__':
    main()
