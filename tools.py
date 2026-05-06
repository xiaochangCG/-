"""训练 & 录制工具"""
import sys, os, json, subprocess
from pathlib import Path

BASE = Path(__file__).parent

def main():
    os.chdir(str(BASE))
    cfg = json.load(open('config.json', 'r', encoding='utf-8'))
    tcfg = cfg.get('train', {})

    while True:
        print()
        print('=' * 40)
        print('  Audio Filter 工具')
        print('=' * 40)
        print()
        print('[1] 录制样本')
        print('[2] 训练模型')
        print('[3] 提取 HuBERT 特征')
        print('[4] 查看样本')
        print('[5] 退出')
        print()

        choice = input('选择: ').strip()

        if choice == '1':
            subprocess.run([sys.executable, '-u', 'record.py'])

        elif choice == '2':
            eps = input(f'训练轮数 (默认100): ').strip()
            lr = input(f'学习率 (默认0.001): ').strip()
            cmd = [sys.executable, '-u', 'train.py']
            if eps:
                cmd.extend(['--epochs', eps])
            if lr:
                cmd.extend(['--lr', lr])
            subprocess.run(cmd)

        elif choice == '3':
            subprocess.run([sys.executable, '-u', 'extract_hubert_features.py'])

        elif choice == '4':
            profanity = BASE / 'samples' / 'profanity'
            normal = BASE / 'samples' / 'normal'
            print()
            total_p = 0
            for d in profanity.iterdir():
                if d.is_dir():
                    n = sum(1 for _ in d.glob('*.wav'))
                    total_p += n
                    print(f'  "{d.name}": {n} 个')
            print(f'  违禁词总计: {total_p}')
            nn = sum(1 for _ in normal.glob('*.wav')) if normal.exists() else 0
            print(f'  正常语音: {nn} 个')

        elif choice == '5':
            break
        else:
            print('无效选择')

if __name__ == '__main__':
    main()
