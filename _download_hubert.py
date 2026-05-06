"""一次性下载 HuBERT 模型到本地"""
import ssl, urllib.request, json
from pathlib import Path

LOCAL = Path(__file__).parent / "hubert_model_files"
FILES = ["config.json", "preprocessor_config.json", "pytorch_model.bin"]

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

LOCAL.mkdir(exist_ok=True)
for f in FILES:
    path = LOCAL / f
    if path.exists():
        print(f"skip {f} (已存在)")
        continue
    url = f"https://huggingface.co/facebook/hubert-base-ls960/resolve/main/{f}"
    print(f"downloading {f}...", end=" ", flush=True)
    try:
        with urllib.request.urlopen(url, context=ctx, timeout=120) as r:
            path.write_bytes(r.read())
        print("ok")
    except Exception as e:
        print(f"FAIL: {e}")

print(f"\nDone. Model at: {LOCAL}")
