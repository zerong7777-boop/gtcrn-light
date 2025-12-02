#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resample_to_16k.py
将数据集中所有 wav 统一为 16 kHz 单声道（就地覆盖，安全：先写 tmp 再原子替换）。
优先使用 ffmpeg；若无则回退到 Python( libosa / scipy.signal.resample_poly )。

用法示例：
  python resample_to_16k.py --root ~/PycharmProjects/RZ/danzi/audio/data
  # 仅查看将会处理哪些文件（不实际改动）：
  python resample_to_16k.py --root ... --dry-run
"""
import argparse, os, sys, glob, shutil, subprocess
import numpy as np
import soundfile as sf

def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

def resample_ffmpeg(inp: str, outp: str):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-i", inp, "-ac", "1", "-ar", "16000", outp]
    subprocess.run(cmd, check=True)

def resample_python(inp: str, outp: str):
    data, sr = sf.read(inp, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)  # 多声道转单声道
    if sr == 16000:
        y = mono
    else:
        # 优先 librosa；若不可用再试 scipy
        try:
            import librosa
            y = librosa.resample(mono, orig_sr=sr, target_sr=16000, res_type="kaiser_best")
        except Exception:
            try:
                from scipy.signal import resample_poly
                import math
                g = math.gcd(sr, 16000)
                up = 16000 // g
                down = sr // g
                y = resample_poly(mono, up, down)
            except Exception as e:
                raise RuntimeError(
                    f"需要 ffmpeg 或 librosa 或 scipy 之一来从 {sr}Hz 重采到 16000Hz"
                ) from e
    sf.write(outp, y.astype(np.float32), 16000)

def process_one(path: str, use_ff: bool, dry: bool=False) -> str:
    info = sf.info(path)
    if info.samplerate == 16000 and info.channels == 1:
        return "skip"
    if dry:
        return f"would_convert({info.samplerate}Hz,{info.channels}ch)"
    tmp = path + ".tmp16k.wav"
    if use_ff:
        resample_ffmpeg(path, tmp)
    else:
        resample_python(path, tmp)
    os.replace(tmp, path)
    return "converted"

def scan(root: str, subdirs: list[str]) -> list[str]:
    files = []
    for sd in subdirs:
        d = os.path.join(root, sd)
        files.extend(sorted(glob.glob(os.path.join(d, "*.wav"))))
    return files

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/PycharmProjects/RZ/danzi/audio/data"),
                    help="数据根目录，包含 train/clean, train/noisy, test/clean, test/noisy")
    ap.add_argument("--subdirs", nargs="+",
                    default=["train/clean","train/noisy","test/clean","test/noisy"],
                    help="要处理的相对子目录列表")
    ap.add_argument("--no-ffmpeg", action="store_true", help="禁用 ffmpeg（强制走 Python）")
    ap.add_argument("--dry-run", action="store_true", help="演练模式，仅打印将要转换的文件")
    args = ap.parse_args()

    files = scan(args.root, args.subdirs)
    if not files:
        print("未找到 wav 文件，请检查 --root 与目录结构。", file=sys.stderr)
        sys.exit(1)

    use_ff = (not args.no_ffmpeg) and has_ffmpeg()
    print(f"共发现 {len(files)} 个 wav；后端：{'ffmpeg' if use_ff else 'python(librosa/scipy)'}；dry={args.dry_run}")

    n_skip = n_conv = n_err = 0
    for i, f in enumerate(files, 1):
        try:
            status = process_one(f, use_ff, args.dry_run)
            if status == "skip":
                n_skip += 1
            elif status.startswith("would_convert"):
                print(f"[DRY] {f} -> {status}")
            else:
                n_conv += 1
            if i % 200 == 0:
                print(f"进度 {i}/{len(files)} | 转换 {n_conv} | 跳过 {n_skip}")
        except Exception as e:
            n_err += 1
            print(f"[ERR] {f}: {e}", file=sys.stderr)

    # 终检：仍非 16k 单声道的样本
    bad = []
    for f in files:
        try:
            info = sf.info(f)
            if not (info.samplerate == 16000 and info.channels == 1):
                bad.append(f)
        except Exception:
            bad.append(f)

    print("\n完成。统计：")
    print(f"  转换: {n_conv}")
    print(f"  跳过: {n_skip}")
    print(f"  错误: {n_err}")
    if bad:
        out_list = os.path.join(args.root, "non_16k_mono.txt")
        with open(out_list, "w") as fw:
            fw.write("\n".join(bad))
        print(f"  警告：仍有 {len(bad)} 个文件不是 16k 单声道，已写入 {out_list}")

if __name__ == "__main__":
    main()
