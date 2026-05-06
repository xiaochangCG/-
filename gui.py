# -*- coding: utf-8 -*-
"""
音频过滤器 GUI 启动器 - PyTorch CNN
"""
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import json
import queue
import time
from pathlib import Path

import sounddevice as sd
import numpy as np

SR = 16000
BLOCK = 512
BASE = Path(__file__).parent
CONFIG_PATH = BASE / 'config.json'
MODEL_PATH = BASE / 'model_hubert.pth'


class FilterApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Audio Profanity Filter')
        self.root.geometry('640x520')
        self.root.resizable(True, True)

        self.running = False
        self.log_queue = queue.Queue()
        self.devices = {}

        self._build_ui()
        self._refresh_devices()
        self._load_config()
        self._process_log()

    def _build_ui(self):
        title = ttk.Label(self.root, text='Audio Profanity Filter', font=('Microsoft YaHei', 14, 'bold'))
        title.pack(pady=(15, 5))

        status_frame = ttk.Frame(self.root)
        status_frame.pack(pady=5)
        self.status_canvas = tk.Canvas(status_frame, width=20, height=20, highlightthickness=0)
        self.status_canvas.pack(side=tk.LEFT, padx=(0, 8))
        self.status_dot = self.status_canvas.create_oval(2, 2, 18, 18, fill='#888')
        self.status_label = ttk.Label(status_frame, text='Ready', font=('Microsoft YaHei', 11))
        self.status_label.pack(side=tk.LEFT)

        dev_frame = ttk.LabelFrame(self.root, text='Devices', padding=10)
        dev_frame.pack(fill=tk.X, padx=15, pady=10)
        ttk.Label(dev_frame, text='Input (Mic):').grid(row=0, column=0, sticky=tk.W, pady=2)
        self.input_var = tk.StringVar()
        self.input_combo = ttk.Combobox(dev_frame, textvariable=self.input_var, state='readonly', width=50)
        self.input_combo.grid(row=0, column=1, padx=(10, 0), pady=2)
        ttk.Label(dev_frame, text='Output (VB-Cable):').grid(row=1, column=0, sticky=tk.W, pady=2)
        self.output_var = tk.StringVar()
        self.output_combo = ttk.Combobox(dev_frame, textvariable=self.output_var, state='readonly', width=50)
        self.output_combo.grid(row=1, column=1, padx=(10, 0), pady=2)
        ttk.Button(dev_frame, text='Refresh', command=self._refresh_devices).grid(row=2, column=0, columnspan=2, pady=(8, 0))

        ctrl_frame = ttk.Frame(self.root)
        ctrl_frame.pack(pady=10)
        self.start_btn = ttk.Button(ctrl_frame, text='Start', command=self._start, width=15)
        self.start_btn.pack(side=tk.LEFT, padx=10)
        self.stop_btn = ttk.Button(ctrl_frame, text='Stop', command=self._stop, width=15, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=10)

        log_frame = ttk.LabelFrame(self.root, text='Log', padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        self.log_area = scrolledtext.ScrolledText(log_frame, height=10, font=('Consolas', 9), wrap=tk.WORD, state=tk.DISABLED)
        self.log_area.pack(fill=tk.BOTH, expand=True)

    def _refresh_devices(self):
        self.devices.clear()
        devs = sd.query_devices()
        in_list, out_list = [], []
        for i, d in enumerate(devs):
            name = d['name']
            label = f'[{i}] {name}'
            self.devices[label] = i
            if d['max_input_channels'] > 0:
                in_list.append(label)
            if d['max_output_channels'] > 0:
                out_list.append(label)
        self.input_combo['values'] = in_list
        self.output_combo['values'] = out_list
        for label in in_list:
            if 'mchose' in label.lower():
                self.input_var.set(label); break
        for label in out_list:
            if 'cable' in label.lower() and 'input' in label.lower():
                self.output_var.set(label); break

    def _load_config(self):
        try:
            cfg = json.load(open(CONFIG_PATH, encoding='utf-8'))
            for label, idx in self.devices.items():
                if idx == cfg.get('input_device'):
                    self.input_var.set(label)
                if idx == cfg.get('output_device'):
                    self.output_var.set(label)
        except Exception:
            pass

    def _save_config(self):
        try:
            in_dev = self.devices.get(self.input_var.get(), 1)
            out_dev = self.devices.get(self.output_var.get(), 10)
            cfg = json.load(open(CONFIG_PATH, encoding='utf-8'))
            cfg['input_device'] = in_dev
            cfg['output_device'] = out_dev
            json.dump(cfg, open(CONFIG_PATH, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
            return in_dev, out_dev
        except Exception as e:
            self._log(f'Config save failed: {e}')
            return 1, 10

    def _log(self, msg):
        self.log_queue.put(msg)

    def _process_log(self):
        while not self.log_queue.empty():
            try:
                msg = self.log_queue.get_nowait()
                self.log_area.configure(state=tk.NORMAL)
                self.log_area.insert(tk.END, msg + '\n')
                self.log_area.see(tk.END)
                self.log_area.configure(state=tk.DISABLED)
            except queue.Empty:
                break
        self.root.after(100, self._process_log)

    def _set_status(self, status, color):
        self.status_label.config(text=status)
        self.status_canvas.itemconfig(self.status_dot, fill=color)

    def _start(self):
        if self.running:
            return
        if not MODEL_PATH.exists():
            messagebox.showwarning('No Model', 'Train a model first:\n\npython train.py')
            return

        in_dev, out_dev = self._save_config()
        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self._set_status('Starting...', '#FFA500')
        self._log(f'Input: [{in_dev}] {sd.query_devices(in_dev)["name"]}')
        self._log(f'Output: [{out_dev}] {sd.query_devices(out_dev)["name"]}')
        self.running = True
        threading.Thread(target=self._run_filter, args=(in_dev, out_dev), daemon=True).start()

    def _run_filter(self, in_dev, out_dev):
        try:
            cfg = json.load(open(CONFIG_PATH, encoding='utf-8'))
            feature_type = cfg.get('feature_type', 'hubert')
            model_path = BASE / cfg.get('model_path', 'model_hubert.pth')

            from hubert import extract as hubert_extract
            from classifier_torch_hubert import HuBERTClassifier
            model = HuBERTClassifier.load(str(model_path))
            # 允许手动覆盖阈值
            override = cfg.get('threshold_override', None)
            if override is not None:
                old_t = model.threshold
                model.threshold = float(override)
                self._log(f'Threshold override: {old_t:.2f} -> {override}')
            self._log(f'Model loaded (threshold={model.threshold:.2f})')

            window_sec = cfg.get('window_sec', 0.5)
            hop_sec = cfg.get('hop_sec', 0.1)
            min_pos = cfg.get('min_positive_windows', 3)
            delay_sec = cfg.get('output_delay_sec', 1.5)
            min_energy = cfg.get('min_energy', 0.003)
            delay_samples = int(delay_sec * SR)

            audio_q = queue.Queue(maxsize=200)

            def mic_cb(indata, frames, ti, status):
                if status and self.running:
                    self._log(f'Input status: {status}')
                try:
                    audio_q.put_nowait(indata.copy())
                except queue.Full:
                    pass

            in_stream = sd.InputStream(device=in_dev, channels=1, samplerate=SR, blocksize=BLOCK, callback=mic_cb, dtype='float32')
            out_stream = sd.OutputStream(device=out_dev, channels=1, samplerate=SR, blocksize=BLOCK, dtype='float32')
            in_stream.start()
            out_stream.start()

            self._set_status('Running', '#00CC00')
            self._log(f'Window: {window_sec}s, Hop: {hop_sec}s, Min streak: {min_pos}')

            from collections import deque
            delay_line = deque()
            analysis_buf = deque()
            sample_counter = 0  # 全局采样计数器
            profanity_segments = []
            positive_streak = 0
            streak_start = 0
            w_samples = int(window_sec * SR)
            h_samples = int(hop_sec * SR)
            detect_cd = 0

            # 加载自定义替换音频
            custom_rep_audio = None
            rtype = cfg.get('replacement_type', 'bird')
            rfile = cfg.get('replacement_file', '')
            if rfile:
                rpath = BASE / rfile
                if rpath.exists():
                    try:
                        from pydub import AudioSegment
                        seg = AudioSegment.from_file(str(rpath))
                        seg = seg.set_frame_rate(SR).set_channels(1).set_sample_width(2)
                        samples = np.array(seg.get_array_of_samples(), dtype=np.float32) / 32768.0
                        custom_rep_audio = samples
                        dur = len(custom_rep_audio) / SR
                        self._log(f'Loaded replacement: {rpath.name} ({dur:.1f}s, mono {SR}Hz)')
                    except Exception as e:
                        self._log(f'Failed: {e}, trying wave...')
                        try:
                            import wave
                            with wave.open(str(rpath), 'rb') as wf:
                                src_sr = wf.getframerate()
                                nf = wf.getnframes()
                                raw = np.frombuffer(wf.readframes(nf), dtype=np.int16)
                                audio_f32 = raw.astype(np.float32) / 32768.0
                            if src_sr != SR:
                                from scipy import signal
                                n_out = int(len(audio_f32) * SR / src_sr)
                                audio_f32 = signal.resample(audio_f32, n_out).astype(np.float32)
                            custom_rep_audio = audio_f32
                            self._log(f'Loaded via wave: ({len(audio_f32)/SR:.1f}s)')
                        except Exception as e2:
                            self._log(f'All methods failed: {e2}')

            def replacement(dur_s):
                # 最小播放时长，防止鸟叫被压缩
                min_dur = cfg.get('replacement_min_dur', 0.5)
                actual_dur = max(dur_s, min_dur)
                n = int(actual_dur * SR)

                if custom_rep_audio is not None:
                    src = custom_rep_audio
                    if n <= len(src):
                        return src[:n].copy()
                    else:
                        repeats = n // len(src) + 1
                        return np.tile(src, repeats)[:n].copy()
                # 默认哔声 (电视台风格: 1000Hz 纯音)
                t = np.linspace(0, dur_s, int(SR * dur_s), endpoint=False)
                audio = np.sin(2 * np.pi * 1000 * t)
                env = np.ones_like(t)
                fade = min(int(0.005 * SR), len(audio) // 2)
                if fade > 0:
                    env[:fade] = np.linspace(0, 1, fade)
                    env[-fade:] = np.linspace(1, 0, fade)
                return (audio * env * 0.5).astype(np.float32)

            while self.running:
                try:
                    chunk = audio_q.get(timeout=0.1)
                except queue.Empty:
                    try:
                        out_stream.write(np.zeros((BLOCK, 1), dtype=np.float32))
                    except Exception:
                        pass
                    continue

                chunk_1d = chunk.flatten()

                # 加入分析和延迟缓冲区
                for s in chunk_1d:
                    analysis_buf.append(s)
                    delay_line.append(s)
                sample_counter += len(chunk_1d)

                while len(analysis_buf) >= w_samples + 3 * h_samples:
                    acc = np.array(list(analysis_buf), dtype=np.float32)

                    # HuBERT: 每个窗口单独提取特征
                    probs = []
                    hp_samples = int(hop_sec * SR)
                    for i in range(0, len(acc) - w_samples + 1, hp_samples):
                        seg = acc[i:i + w_samples]
                        if seg.std() < min_energy:
                            probs.append(0.0)
                        else:
                            try:
                                feats = hubert_extract(seg)
                                probs.append(float(model.predict_proba(feats)))
                            except Exception:
                                probs.append(0.0)
                    probs = np.array(probs)

                    if detect_cd <= 0 and len(probs) > 0:
                        self._log(f'Prob: [{probs.min():.3f}-{probs.max():.3f}] mean={probs.mean():.3f} e={acc.std():.4f}')
                        # 如果全部是1.0，保存一段音频用于诊断
                        if (probs == 1.0).all() and acc.std() > 0.001:
                            try:
                                import wave
                                dump = (acc * 32767).clip(-32768,32767).astype(np.int16)
                                dpath = BASE / f'_dump_{int(time.time())}.wav'
                                with wave.open(str(dpath), 'wb') as wf:
                                    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
                                    wf.writeframes(dump.tobytes())
                                self._log(f'Dumped audio: {dpath.name}')
                            except:
                                pass
                        detect_cd = 30

                    for i, prob in enumerate(probs):
                        # 全局坐标 = 当前捕捉总数 - 缓冲区剩余 + 窗口偏移
                        win_t = sample_counter - len(analysis_buf) + i * h_samples
                        seg_end_t = win_t + w_samples
                        if prob > model.threshold:
                            positive_streak += 1
                            if positive_streak == 1:
                                streak_start = win_t
                        else:
                            if positive_streak >= min_pos:
                                # 预生成完整替换音频
                                seg_dur = (seg_end_t - streak_start) / SR
                                full_rep = replacement(max(seg_dur, 0.5))
                                profanity_segments.append((streak_start, seg_end_t, full_rep))
                                if detect_cd <= 0:
                                    self._log(f'Profanity [{streak_start/SR:.1f}s-{seg_end_t/SR:.1f}s]')
                                    detect_cd = 30
                            positive_streak = 0

                    pop_n = min(h_samples, len(analysis_buf))
                    for _ in range(pop_n):
                        analysis_buf.popleft()

                if detect_cd > 0:
                    detect_cd -= 1

                # Output (delayed)
                if len(delay_line) > delay_samples + BLOCK:
                    # 输出位于 delay_samples 前的音频
                    out_audio = np.array([delay_line.popleft() for _ in range(BLOCK)], dtype=np.float32)
                    # 当前输出的全局起始位置 = 当前已捕获 - 延迟队列长度 - 已输出块大小
                    out_start = sample_counter - len(delay_line) - BLOCK

                    i = 0
                    while i < len(profanity_segments):
                        abs_s, abs_e, rep_full = profanity_segments[i]
                        rel_s = abs_s - out_start
                        rel_e = abs_e - out_start
                        if rel_e <= 0:
                            profanity_segments.pop(i)
                            continue
                        if rel_s < BLOCK and rel_e > 0:
                            rs = max(0, int(rel_s))
                            re = min(BLOCK, int(rel_e))
                            if re > rs:
                                # 从预生成缓冲中取对应偏移
                                rep_offset = max(0, int(-rel_s))
                                take = re - rs
                                rl = min(take, len(rep_full) - rep_offset)
                                if rl > 0 and rs + rl <= BLOCK:
                                    out_audio[rs:rs + rl] = rep_full[rep_offset:rep_offset + rl]
                        if rel_e >= BLOCK:
                            i += 1
                        else:
                            profanity_segments.pop(i)

                    try:
                        out_stream.write(out_audio.reshape(-1, 1))
                    except Exception:
                        pass
                else:
                    try:
                        out_stream.write(np.zeros((BLOCK, 1), dtype=np.float32))
                    except Exception:
                        pass

            in_stream.stop(); in_stream.close()
            out_stream.stop(); out_stream.close()
        except Exception as e:
            self._log(f'Error: {e}')
            import traceback
            self._log(traceback.format_exc())
        finally:
            if 'hubert' in dir():
                try:
                    hubert.close()
                    self._log('HuBERT server stopped')
                except Exception:
                    pass

        if self.running:
            self.running = False
            self.root.after(0, self._on_stopped)

    def _stop(self):
        self.running = False
        self._set_status('Stopping...', '#FFA500')
        self._log('Stopping...')
        self.root.after(500, self._on_stopped)

    def _on_stopped(self):
        self._set_status('Stopped', '#888')
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self._log('Stopped')


def main():
    root = tk.Tk()
    FilterApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
