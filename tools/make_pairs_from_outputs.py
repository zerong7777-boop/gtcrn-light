#!/usr/bin/env python3
import argparse, os, glob
'''

python tools/make_pairs_from_outputs.py \
  --enh_dir  /home/jgzn/PycharmProjects/RZ/danzi/audio/SEtrain/exp_gtcrn_vbd_2025-11-10-00h47m/enh_test \
  --clean_dir ../data/test/clean \
  --out_scp ../SEtrain/outs/test_enh_vs_clean.scp

'''
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enh_dir", required=True, help="批量推理输出目录")
    ap.add_argument("--clean_dir", required=True, help="test/clean 目录")
    ap.add_argument("--out_scp", required=True, help="写出的 pairs.scp 路径")
    args = ap.parse_args()

    enhs = sorted(glob.glob(os.path.join(args.enh_dir, "*.wav")))
    assert enhs, f"未在 {args.enh_dir} 找到增强文件"

    pairs = []
    miss = []
    for e in enhs:
        bn = os.path.basename(e)
        c = os.path.join(args.clean_dir, bn)
        if os.path.exists(c):
            pairs.append(f"{e} {c}\n")
        else:
            miss.append(bn)

    os.makedirs(os.path.dirname(args.out_scp), exist_ok=True)
    with open(args.out_scp, "w") as f:
        f.writelines(pairs)

    print(f"写入 {args.out_scp}，条目 {len(pairs)}；缺失 {len(miss)}")
    if miss:
        print("缺失清单（增强有而 clean 无）：")
        for m in miss[:20]:
            print("  ", m)
        if len(miss) > 20:
            print("  ...")

if __name__ == "__main__":
    main()
