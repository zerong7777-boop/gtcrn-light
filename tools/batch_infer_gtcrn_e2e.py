#!/usr/bin/env python3
import argparse, os, glob
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
import os, sys                                  # CHANGED
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # CHANGED: 把 SEtrain 根目录加到 sys.path

# 端到端 GTCRN：models/__init__.py 里应当暴露 GTCRN
from models import GTCRN   # CHANGED


'''

# 到 SEtrain 目录（保证能 import models）
cd ~/PycharmProjects/RZ/danzi/audio/SEtrain

# 选一个 checkpoint（把 best.pt/ckpt 改成你的实际文件名）
CKPT="/home/jgzn/PycharmProjects/RZ/danzi/audio/SEtrain/exp_gtcrn_vbd_2025-11-10-00h47m/checkpoints/best_model_076.tar"

# 输出目录（会自动创建）
OUT="/home/jgzn/PycharmProjects/RZ/danzi/audio/SEtrain/exp_gtcrn_vbd_2025-11-10-00h47m/enh_test"

python tools/batch_infer_gtcrn_e2e.py \
  --ckpt "$CKPT" \
  --in_dir  ../data/test/noisy \
  --out_dir "$OUT" \
  --sr 16000 --device cuda    # 没GPU就改成 --device cpu


'''

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="训练得到的权重路径（.pt/.ckpt）")
    ap.add_argument("--in_dir", required=True, help="测试集 noisy 目录")
    ap.add_argument("--out_dir", required=True, help="增强结果输出目录")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pattern", default="*.wav")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 模型配置需与训练一致
    model = GTCRN(n_fft=512, hop_len=256, win_len=512).to(args.device).eval()
    state = torch.load(args.ckpt, map_location="cpu")
    sd = state.get("model", state)   # 兼容不同保存格式
    model.load_state_dict(sd, strict=False)

    files = sorted(glob.glob(os.path.join(args.in_dir, args.pattern)))
    assert files, f"未在 {args.in_dir} 找到 {args.pattern}"

    for f in tqdm(files, desc="infer"):
        x, fs = sf.read(f, dtype="float32")
        if fs != args.sr:
            raise RuntimeError(f"{f} 采样率={fs}，应为 {args.sr}，请先统一重采样。")
        if x.ndim == 2: x = x.mean(axis=1)

        with torch.no_grad():
            y = model(torch.from_numpy(x).to(args.device).unsqueeze(0)).squeeze(0).float().cpu().numpy()

        sf.write(os.path.join(args.out_dir, os.path.basename(f)), y.astype(np.float32), args.sr)

if __name__ == "__main__":
    main()
