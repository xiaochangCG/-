# -*- coding: utf-8 -*-
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import sounddevice as sd
import numpy as np
import time
import wave
from pathlib import Path
from collections import deque

SR = 16000
SAMPLES_DIR = Path(__file__).parent / 'samples'
PROFANITY_DIR = SAMPLES_DIR / 'profanity'
NORMAL_DIR = SAMPLES_DIR / 'normal'


def save_wav(audio, path, sr=16000):
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())


def auto_record(target_dir, max_samples=50, label=''):
    """自动连续录制模式"""
    target_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(target_dir.glob('*.wav')))
    count = existing
    stop_flag = [False]

    print(f'\n=== 自动录制 ({label}已{existing}个) ===')
    print(f'最多{max_samples}个')
    print('不停重复念词，间隔0.5秒以上')
    print('按回车开始，再按回车停止\n')

    input('按回车开始...')
    print('监听中...', flush=True)

    buffer = deque()
    recording = False
    rec_start = 0
    silence_count = 0
    threshold = 0.02
    silence_s = int(0.5 * SR)
    min_speech_s = int(0.2 * SR)
    max_rec_s = int(2.5 * SR)

    def callback(indata, frames, time_info, status):
        nonlocal recording, rec_start, silence_count, count
        if status:
            return
        audio = indata.flatten()
        for s in audio:
            buffer.append(s)
            if abs(s) > threshold:
                silence_count = 0
                if not recording:
                    recording = True
                    rec_start = len(buffer) - 1
            elif recording:
                silence_count += 1
                if silence_count >= silence_s:
                    start = max(0, rec_start - int(0.1 * SR))
                    seg_len = len(buffer) - silence_count - start
                    if seg_len >= min_speech_s:
                        seg = np.array(list(buffer))[start:len(buffer) - silence_count]
                        if len(seg) > 2 * SR:
                            seg = seg[:2 * SR]
                        count += 1
                        save_wav(seg, target_dir / f'{count:04d}.wav')
                        print(f'\r  已保存 {count} 个', end='', flush=True)
                        if count >= max_samples:
                            stop_flag[0] = True
                    recording = False
            if recording and len(buffer) - rec_start > max_rec_s:
                recording = False
            while len(buffer) > SR * 3:
                buffer.popleft()
                if rec_start > 0:
                    rec_start -= 1

    stream = sd.InputStream(device=1, channels=1, samplerate=SR, blocksize=256, callback=callback, dtype='float32')
    stream.start()
    print("按回车停止(录制满自动停止)...", flush=True)
    try:
        while not stop_flag[0]:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop(); stream.close()

    new = len(list(target_dir.glob('*.wav')))
    print(f'\n完成！总计 {new} 个 (新增 {new - existing} 个)')


def record_normal_auto(max_samples=30):
    stop_flag = [False]
    """自动录制正常语音"""
    NORMAL_DIR.mkdir(parents=True, exist_ok=True)
    existing = len(list(NORMAL_DIR.glob('*.wav')))
    count = existing

    print(f'\n=== 自动录制正常语音 (已{existing}个) ===')
    print('自然说话，别带违禁词')
    input('按回车开始...')
    print('监听中...', flush=True)

    buffer = deque()
    recording = False
    rec_start = 0
    silence_count = 0
    threshold = 0.015
    silence_s = int(0.6 * SR)
    min_speech_s = int(0.3 * SR)

    def callback(indata, frames, time_info, status):
        nonlocal recording, rec_start, silence_count, count
        if status:
            return
        audio = indata.flatten()
        for s in audio:
            buffer.append(s)
            if abs(s) > threshold:
                silence_count = 0
                if not recording:
                    recording = True
                    rec_start = len(buffer) - 1
            elif recording:
                silence_count += 1
                if silence_count >= silence_s:
                    start = max(0, rec_start - int(0.1 * SR))
                    seg_len = len(buffer) - silence_count - start
                    if seg_len >= min_speech_s:
                        seg = np.array(list(buffer))[start:len(buffer) - silence_count]
                        if len(seg) > 3 * SR:
                            seg = seg[:3 * SR]
                        count += 1
                        save_wav(seg, NORMAL_DIR / f'{count:04d}.wav')
                        print(f'\r  正常语音: {count} 个', end='', flush=True)
                        if count >= max_samples:
                            stop_flag[0] = True
                    recording = False
            while len(buffer) > SR * 4:
                buffer.popleft()
                if rec_start > 0:
                    rec_start -= 1

    stream = sd.InputStream(device=1, channels=1, samplerate=SR, blocksize=256, callback=callback, dtype='float32')
    stream.start()
    print("按回车停止(录制满自动停止)...", flush=True)
    try:
        while not stop_flag[0]:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop(); stream.close()

    print(f'\n完成！总计 {len(list(NORMAL_DIR.glob("*.wav")))} 个')


def main():
    print('=== 违禁词样本录制工具 ===\n')
    while True:
        print('1. 录违禁词到统一文件夹 (推荐)')
        print('2. 录违禁词到单独文件夹')
        print('3. 录正常语音')
        print('4. 查看样本')
        print('5. 退出')
        choice = input('选择: ').strip()

        if choice == '1':
            n = int(input('最多录几个 (默认50): ').strip() or '50')
            all_dir = PROFANITY_DIR / 'all'
            auto_record(all_dir, max_samples=n, label='违禁词')
        elif choice == '2':
            word = input('违禁词: ').strip()
            if word:
                auto_record(PROFANITY_DIR / word, label=word)
        elif choice == '3':
            record_normal_auto()
        elif choice == '4':
            print()
            total_p = 0
            for d in PROFANITY_DIR.iterdir():
                if d.is_dir():
                    n = len(list(d.glob('*.wav')))
                    total_p += n
                    print(f'  "{d.name}": {n} 个')
            print(f'  违禁词总计: {total_p}')
            nn = len(list(NORMAL_DIR.glob('*.wav')))
            print(f'  正常语音: {nn} 个')
        elif choice == '5':
            break


if __name__ == '__main__':
    main()
