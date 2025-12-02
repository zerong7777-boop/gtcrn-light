#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_se.py
语音增强评测一体化脚本：
- 输入：pairs.scp（每行：<enhanced.wav> <clean.wav>）或 --enh_dir + --clean_dir（按文件名匹配）
- 指标：PESQ（wb/nb）、STOI、可选 ESTOI、SI-SNR
- 并行：--num_workers N
- 输出：终端汇总 + CSV 明细（--out_csv）

依赖：
  pip install soundfile pesq pystoi tqdm
（Linux 下 PESQ 更稳；全部音频建议统一为 16 kHz 单声道）

用法示例（与你目录一致）：
  # 方式一：已有 pairs.scp
  python evaluate_se.py --pairs ../SEtrain/outs/test_enh_vs_clean.scp --sr 16000 --mode wb --estoi --out_csv ../SEtrain/outs/metrics_vbd.csv

  # 方式二：直接给目录，让脚本匹配同名文件
  python evaluate_se.py --enh_dir ../SEtrain/outs/gtcrn_e2e_test --clean_dir ../data/test/clean --sr 16000 --mode wb --out_csv ../SEtrain/outs/metrics_vbd.csv


cd ~/PycharmProjects/RZ/danzi/audio/tools

# 方式一：用 pairs.scp
python evaluate_se.py \
  --pairs ../SEtrain/outs/test_enh_vs_clean.scp \
  --sr 16000 --mode wb --estoi \
  --num_workers 4 \
  --out_csv ../SEtrain/outs/metrics_vbd.csv

# 方式二：直接给目录
python evaluate_se.py \
  --enh_dir ../SEtrain/outs/gtcrn_e2e_test \
  --clean_dir ../data/test/clean \
  --sr 16000 --mode wb \
  --num_workers 4 \
  --out_csv ../SEtrain/outs/metrics_vbd.csv

"""

from __future__ import annotations
import argparse, os, sys, glob, csv, math
from typing import List, Tuple, Dict, Optional
import numpy as np
import soundfile as sf
from tqdm import tqdm

# 可选依赖检查
try:
    from pesq import pesq
    _HAS_PESQ = True
except Exception:
    _HAS_PESQ = False

try:
    from pystoi import stoi
    _HAS_STOI = True
except Exception:
    _HAS_STOI = False

def _maybe_resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    # 优先 librosa，再退 scipy.signal.resample_poly
    try:
        import librosa
        return librosa.resample(x, orig_sr=sr_in, target_sr=sr_out, res_type="kaiser_best")
    except Exception:
        try:
            from scipy.signal import resample_poly
            g = math.gcd(sr_in, sr_out)
            up = sr_out // g
            down = sr_in // g
            return resample_poly(x, up, down)
        except Exception as e:
            raise RuntimeError(f"需要 librosa 或 scipy 才能从 {sr_in}Hz 重采到 {sr_out}Hz。") from e

def si_snr(ref: np.ndarray, est: np.ndarray, eps: float = 1e-8) -> float:
    """Scale-Invariant SNR (dB)"""
    ref = ref.astype(np.float64)
    est = est.astype(np.float64)
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    ref_energy = np.sum(ref ** 2) + eps
    proj = (np.sum(ref * est) / ref_energy) * ref
    noise = est - proj
    ratio = (np.sum(proj ** 2) + eps) / (np.sum(noise ** 2) + eps)
    return 10.0 * np.log10(ratio + eps)

def load_pair(enh_path: str, clean_path: str, sr: int, auto_resample: bool) -> Tuple[np.ndarray, np.ndarray]:
    y, fs1 = sf.read(enh_path, dtype="float32")
    x, fs2 = sf.read(clean_path, dtype="float32")
    if y.ndim == 2: y = y.mean(axis=1)
    if x.ndim == 2: x = x.mean(axis=1)
    if fs1 != sr or fs2 != sr:
        if not auto_resample:
            raise RuntimeError(f"采样率不匹配（{fs1}/{fs2} vs {sr}）。可加 --auto_resample 或提前统一重采样。")
        y = _maybe_resample(y, fs1, sr)
        x = _maybe_resample(x, fs2, sr)
    # 长度对齐：截到最短
    L = min(len(x), len(y))
    if L <= 0:
        raise RuntimeError("空音频或长度为 0")
    x = x[:L]
    y = y[:L]
    # 防极值：限制幅度 [-1, 1]
    x = np.clip(x, -1.0, 1.0)
    y = np.clip(y, -1.0, 1.0)
    return y, x  # (enh, clean)

def eval_one(enh_path: str, clean_path: str, sr: int, pesq_mode: str, do_estoi: bool, auto_resample: bool) -> Dict[str, float]:
    enh, clean = load_pair(enh_path, clean_path, sr, auto_resample)

    res: Dict[str, float] = {}
    # PESQ
    if _HAS_PESQ:
        try:
            res["pesq"] = float(pesq(sr, clean, enh, pesq_mode))
        except Exception:
            res["pesq"] = float("nan")
    else:
        res["pesq"] = float("nan")

    # STOI / ESTOI
    if _HAS_STOI:
        try:
            res["stoi"] = float(stoi(clean, enh, sr, extended=False))
        except Exception:
            res["stoi"] = float("nan")
        if do_estoi:
            try:
                res["estoi"] = float(stoi(clean, enh, sr, extended=True))
            except Exception:
                res["estoi"] = float("nan")
    else:
        res["stoi"] = float("nan")
        if do_estoi:
            res["estoi"] = float("nan")

    # SI-SNR
    try:
        res["si_snr"] = float(si_snr(clean, enh))
    except Exception:
        res["si_snr"] = float("nan")

    return res

def build_pairs_from_dirs(enh_dir: str, clean_dir: str) -> List[Tuple[str, str]]:
    enhs = sorted(glob.glob(os.path.join(enh_dir, "*.wav")))
    pairs = []
    miss = []
    for e in enhs:
        bn = os.path.basename(e)
        c = os.path.join(clean_dir, bn)
        if os.path.exists(c):
            pairs.append((e, c))
        else:
            miss.append(bn)
    if miss:
        print(f"[WARN] {len(miss)} 个增强文件在 clean_dir 下未找到同名参考（只展示前 10 个）：", file=sys.stderr)
        for m in miss[:10]:
            print("   -", m, file=sys.stderr)
    if not pairs:
        raise RuntimeError("没有匹配到任何 (enh, clean) 对。")
    return pairs

def read_pairs_scp(pairs_path: str) -> List[Tuple[str, str]]:
    pairs = []
    with open(pairs_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"{pairs_path}:{ln} 行格式错误：`{line}`")
            e, c = parts[0], parts[1]
            if not (os.path.isfile(e) and os.path.isfile(c)):
                raise FileNotFoundError(f"{pairs_path}:{ln} 文件不存在：{e} 或 {c}")
            pairs.append((e, c))
    if not pairs:
        raise RuntimeError("pairs.scp 为空。")
    return pairs

def main():
    ap = argparse.ArgumentParser(description="语音增强评测（PESQ/STOI/ESTOI/SI-SNR）")
    g_in = ap.add_mutually_exclusive_group(required=True)
    g_in.add_argument("--pairs", help="pairs.scp 路径，每行：<enhanced.wav> <clean.wav>")
    g_in.add_argument("--enh_dir", help="增强 wav 目录（与 --clean_dir 配合）")
    ap.add_argument("--clean_dir", help="参考 clean wav 目录（与 --enh_dir 配合）")

    ap.add_argument("--sr", type=int, default=16000, help="采样率（PESQ wb=16k）")
    ap.add_argument("--mode", default="wb", choices=["wb","nb"], help="PESQ 模式：wb(16k) 或 nb(8k)")
    ap.add_argument("--estoi", action="store_true", help="额外计算 ESTOI")
    ap.add_argument("--auto_resample", action="store_true", help="如采样率不匹配则自动重采样（需 librosa 或 scipy）")

    ap.add_argument("--num_workers", type=int, default=0, help="并行进程数，0=单进程")
    ap.add_argument("--out_csv", default="", help="保存逐文件结果到 CSV")
    args = ap.parse_args()

    # 构造 (enh, clean) 对
    if args.pairs:
        pairs = read_pairs_scp(args.pairs)
    else:
        if not args.clean_dir:
            ap.error("--enh_dir 需要配合 --clean_dir 使用")
        pairs = build_pairs_from_dirs(args.enh_dir, args.clean_dir)

    # 指标计算
    per_file: List[Dict[str, float]] = []
    from concurrent.futures import ProcessPoolExecutor, as_completed

    def _worker(p):
        e, c = p
        res = eval_one(e, c, args.sr, args.mode, args.estoi, args.auto_resample)
        return {"file": os.path.basename(e), "enh": e, "clean": c, **res}

    if args.num_workers and args.num_workers > 0:
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futures = [ex.submit(_worker, p) for p in pairs]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="eval(parallel)"):
                per_file.append(fut.result())
    else:
        for p in tqdm(pairs, desc="eval"):
            per_file.append(_worker(p))

    # 汇总
    def _mean(key):
        xs = [d.get(key, float("nan")) for d in per_file]
        xs = np.asarray(xs, dtype=float)
        return float(np.nanmean(xs)) if xs.size else float("nan")

    print("\n=== Summary (mean over {} pairs) ===".format(len(per_file)))
    print("PESQ ({}): {:.4f}".format(args.mode, _mean("pesq")))
    print("STOI     : {:.4f}".format(_mean("stoi")))
    if args.estoi:
        print("ESTOI    : {:.4f}".format(_mean("estoi")))
    print("SI-SNR dB: {:.4f}".format(_mean("si_snr")))

    # 导出 CSV
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        keys = ["file", "pesq", "stoi"] + (["estoi"] if args.estoi else []) + ["si_snr", "enh", "clean"]
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for d in per_file:
                row = {k: d.get(k, "") for k in keys}
                w.writerow(row)
        print(f"\nCSV 写入：{args.out_csv}")

if __name__ == "__main__":
    main()
