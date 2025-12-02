#!/usr/bin/env python3
import argparse, os, numpy as np, soundfile as sf, math
from tqdm import tqdm

# pip install pesq pystoi
from pesq import pesq
from pystoi import stoi
'''

python tools/eval_metrics.py --pairs ../SEtrain/outs/test_enh_vs_clean.scp --sr 16000 --mode wb --estoi


'''
def si_snr(ref, est, eps=1e-8):
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    ref_energy = np.sum(ref ** 2) + eps
    proj = np.sum(ref * est) * ref / ref_energy
    noise = est - proj
    ratio = (np.sum(proj ** 2) + eps) / (np.sum(noise ** 2) + eps)
    return 10 * np.log10(ratio + eps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="pairs.scp 路径（每行：enh clean）")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--mode", default="wb", choices=["wb","nb"], help="PESQ 模式：wb=16k")
    ap.add_argument("--estoi", action="store_true", help="额外计算 extended STOI")
    args = ap.parse_args()

    lines = [l.strip() for l in open(args.pairs) if l.strip()]
    pesq_scores, stoi_scores, estoi_scores, sisnr_scores = [], [], [], []

    for i, line in enumerate(tqdm(lines, desc="eval"), 1):
        enh, clean = line.split()[:2]
        y, fs1 = sf.read(enh, dtype="float32")
        x, fs2 = sf.read(clean, dtype="float32")
        assert fs1 == fs2 == args.sr, f"采样率不为 {args.sr}: {enh} / {clean}"

        if y.ndim == 2: y = y.mean(axis=1)
        if x.ndim == 2: x = x.mean(axis=1)

        L = min(len(x), len(y))
        x = x[:L]; y = y[:L]

        try:
            pesq_scores.append(pesq(args.sr, x, y, args.mode))
        except Exception:
            # PESQ 失败（极个别 NaN/异常），记为 nan
            pesq_scores.append(np.nan)

        try:
            stoi_scores.append(stoi(x, y, args.sr, extended=False))
            if args.estoi:
                estoi_scores.append(stoi(x, y, args.sr, extended=True))
        except Exception:
            stoi_scores.append(np.nan)
            if args.estoi: estoi_scores.append(np.nan)

        sisnr_scores.append(si_snr(x, y))

    def _mean(xs):
        xs = np.asarray(xs, dtype=float)
        return float(np.nanmean(xs)) if xs.size else float("nan")

    print("\n=== Metrics (mean over pairs) ===")
    print(f"PESQ ({args.mode}): {_mean(pesq_scores):.4f}")
    print(f"STOI:             {_mean(stoi_scores):.4f}")
    if args.estoi:
        print(f"ESTOI:            {_mean(estoi_scores):.4f}")
    print(f"SI-SNR (dB):      {_mean(sisnr_scores):.4f}")


if __name__ == "__main__":
    main()
