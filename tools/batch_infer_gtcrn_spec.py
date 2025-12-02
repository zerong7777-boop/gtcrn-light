#!/usr/bin/env python3
import argparse, os, glob, torch, soundfile as sf
from tqdm import tqdm

import os, sys                                  # CHANGED
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # CHANGED: 把 SEtrain 根目录加到 sys.path

from models import GTCRN  # 频谱版

'''
python tools/batch_infer_gtcrn_spec.py \
  --ckpt runs/gtcrn_spec_vbd/**/checkpoints/best.pt \
  --in_dir  ../data/test/noisy \
  --out_dir ../SEtrain/outs/gtcrn_spec_test \
  --sr 16000 --n_fft 512 --hop 256 --win 512 --device cuda

'''

def stft_ri(x, n_fft, hop, win, window):
    S = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win, window=window,
                   center=True, return_complex=True)
    return torch.view_as_real(S).to(torch.float32)

def istft_from_ri(ri, n_fft, hop, win, window, length):
    # ri: (F,T,2) 或 (1,F,T,2)
    if ri.dim() == 4:
        ri = ri[0]  # (F,T,2)
    ri = ri.to(torch.float32)
    real = ri[..., 0].contiguous()
    imag = ri[..., 1].contiguous()
    S = torch.complex(real, imag)  # (F,T), complex64

    y = torch.istft(
        S, n_fft=n_fft, hop_length=hop, win_length=win, window=window,
        center=True, length=length
    )
    return y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--n_fft", type=int, default=512)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--win", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    window = torch.hann_window(args.win).pow(0.5).to(args.device)

    model = GTCRN().to(args.device).eval()   # 频谱核心网络若无入参，直接实例化
    state = torch.load(args.ckpt, map_location="cpu"); sd = state.get("model", state)
    model.load_state_dict(sd, strict=False)

    wavs = sorted(glob.glob(os.path.join(args.in_dir, "*.wav")))
    assert wavs, f"no wavs in {args.in_dir}"
    for w in tqdm(wavs, desc="infer-spec"):
        x, fs = sf.read(w, dtype="float32"); assert fs == args.sr
        if x.ndim == 2: x = x.mean(axis=1)
        x_t = torch.from_numpy(x).to(args.device)

        with torch.no_grad():
            n_spec = stft_ri(x_t, args.n_fft, args.hop, args.win, window)[None]   # (1,F,T,2)
            y_spec = model(n_spec)[0]                                            # (F,T,2)
            y = istft_from_ri(y_spec, args.n_fft, args.hop, args.win, window, length=len(x_t))

        sf.write(os.path.join(args.out_dir, os.path.basename(w)), y.float().cpu().numpy(), args.sr)

if __name__ == "__main__":
    main()
