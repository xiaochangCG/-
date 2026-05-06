"""预提取 HuBERT 特征 — 训练前运行一次"""
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import wave
from pathlib import Path

SR = 16000
BASE = Path(__file__).parent
SAMPLES_DIR = BASE / 'samples'
PROFANITY_DIR = SAMPLES_DIR / 'profanity'
NORMAL_DIR = SAMPLES_DIR / 'normal'

def load_wav(path):
    with wave.open(str(path), 'rb') as wf:
        audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        return audio.astype(np.float32) / 32768.0

def main():
    from hubert import extract
    print('Loading HuBERT model...')
    # Warm up
    _ = extract(np.zeros(16000, dtype=np.float32))
    print('Ready.\n')

    total = 0
    for label_dir, label_name in [(PROFANITY_DIR, 'profanity'), (NORMAL_DIR, 'normal')]:
        print(f'=== {label_name} ===')
        for wav in sorted(label_dir.rglob('*.wav')):
            npy = wav.with_suffix('.hubert.npy')
            if npy.exists():
                continue
            try:
                audio = load_wav(wav)
                feats = extract(audio)
                npy.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(npy), feats)
                total += 1
                if total % 10 == 0:
                    print(f'  {total} done...', flush=True)
            except Exception as e:
                print(f'  ERROR {wav.name}: {e}')

    print(f'\nTotal extracted: {total}')

if __name__ == '__main__':
    main()
