#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
打印 GTCRN / GTCRN-Light(v1/v2/v3) 的 Para.(M) 与 MACs(G/s)

- mode=spec：加载“频谱版”核心网络（不含 stft/istft），用 1 秒输入统计 MACs（G/s）
- mode=e2e ：加载“端到端版”（含 stft/istft），仅统计参数量（常规 FLOPs 工具不支持 stft）
"""
import os, argparse, importlib.util, inspect
import torch

def clever_format(x, unit=1.0, decimals=3):
    return f"{float(x)/unit:.{decimals}f}"

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def _import_class(file_path, class_name):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到模型文件：{file_path}")
    spec = importlib.util.spec_from_file_location("xmod_stats", file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    try:
        return getattr(mod, class_name)
    except AttributeError as e:
        raise RuntimeError(f"在 {file_path} 中找不到类 {class_name}") from e

def _build_model(cls, **kwargs):
    sig = inspect.signature(cls.__init__)
    valid = {k: v for k, v in kwargs.items() if k in sig.parameters}
    try:
        return cls(**valid)
    except TypeError:
        return cls()

def _profile_macs(model, dummy):
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        return macs
    except Exception as e:
        raise RuntimeError(
            "统计 MACs 需要安装 thop：`pip install thop`（推荐）。"
            f"\n原始错误：{repr(e)}"
        )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["spec", "e2e"], default="spec")

    # —— 选择架构 —— #
    ap.add_argument("--arch",
        choices=["gtcrn", "gtrcn_light", "gtrcn_light_v2", "gtrcn_light_v3"],
        default="gtcrn")

    # 文件路径（均可被 --spec-file 覆盖）
    ap.add_argument("--gtcrn-file",   default="models/gtcrn.py", help="类名：GTCRN")
    ap.add_argument("--light-file",   default="models/gtrcn_light.py", help="类名：GTCRN_light（v1）")
    ap.add_argument("--lightv2-file", default="models/gtrcn_light_v2.py", help="类名：GTCRN_light_v2")
    ap.add_argument("--lightv3-file", default="models/gtrcn_light_v3.py", help="类名：GTCRN_light（v3）")
    ap.add_argument("--e2e-file",     default="models/gtcrn_end2end.py", help="类名：GTCRN")

    # 兼容旧习惯（若提供且存在则覆盖上面的文件路径）
    ap.add_argument("--spec-file", default="", help="兼容旧参数：若提供且存在，将覆盖相应架构的默认文件")

    # 轻量版可调
    ap.add_argument("--width-mult", type=float, default=1.0)
    ap.add_argument("--two-rnn",    action="store_true")
    ap.add_argument("--rnn-bidir",  action="store_true", default=True)

    # 输入形状设定
    ap.add_argument("--sr",   type=int, default=16000)
    ap.add_argument("--n_fft",type=int, default=512)
    ap.add_argument("--hop",  type=int, default=256)
    ap.add_argument("--win",  type=int, default=512)

    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    if args.mode == "spec":
        # 1) 选择文件与类名
        if args.arch == "gtcrn":
            file_path = args.gtcrn_file
            class_name = "GTCRN"
            ctor_kwargs = dict(n_fft=args.n_fft, hop_len=args.hop, win_len=args.win)
            arch_label = "GTCRN (Spec Core)"
            extra = {}
        elif args.arch == "gtrcn_light":
            # v1：类名仍叫 GTCRN_light
            file_path = args.light_file
            class_name = "GTCRN_light"
            ctor_kwargs = dict(width_mult=args.width_mult, use_two_dpgrnn=args.two_rnn)
            arch_label = "GTCRN-Light (Spec Core)"
            extra = {"width_mult": args.width_mult, "use_two_dpgrnn": bool(args.two_rnn)}
        elif args.arch == "gtrcn_light_v2":
            file_path = args.lightv2_file
            class_name = "GTCRN_light_v2"
            ctor_kwargs = dict(width_mult=args.width_mult, use_two_dpgrnn=args.two_rnn,
                               rnn_bidirectional=args.rnn_bidir)
            arch_label = "GTCRN-Light v2 (Spec Core)"
            extra = {"width_mult": args.width_mult, "use_two_dpgrnn": bool(args.two_rnn),
                     "rnn_bidirectional": bool(args.rnn_bidir)}
        else:  # gtrcn_light_v3
            file_path = args.lightv3_file
            # 我们的 v3 文件中类名仍为 GTCRN_light（说明文档已注明）
            class_name = "GTCRN_light"
            ctor_kwargs = dict(width_mult=args.width_mult, use_two_dpgrnn=args.two_rnn,
                               rnn_bidirectional=args.rnn_bidir)
            arch_label = "GTCRN-Light v3 (Spec Core)"
            extra = {"width_mult": args.width_mult, "use_two_dpgrnn": bool(args.two_rnn),
                     "rnn_bidirectional": bool(args.rnn_bidir)}

        # 兼容旧参数：若传了 --spec-file 并存在，则覆盖 file_path
        if args.spec_file and os.path.exists(args.spec_file):
            file_path = args.spec_file

        # 2) 加载类并构建模型
        GT = _import_class(file_path, class_name)
        model = _build_model(GT, **ctor_kwargs).to(device).eval()

        # 3) 形状 & 统计
        F = args.n_fft // 2 + 1
        T = (args.sr // args.hop) + 1
        dummy = torch.randn(1, F, T, 2, device=device)
        params_m = count_params(model) / 1e6
        macs_g = _profile_macs(model, dummy) / 1e9

        # 4) 明确打印到底加载了哪个“文件/类”
        print(f"=== {arch_label} ===")
        if extra:
            print("Config  : " + ", ".join(f"{k}={v}" for k, v in extra.items()))
        print(f"Loaded  : file='{file_path}', class='{class_name}'")
        print(f"Input(1s): F={F}, T={T}  (n_fft={args.n_fft}, hop={args.hop}, win={args.win})")
        print(f"Para. (M):  {clever_format(params_m)}")
        print(f"MACs (G/s): {clever_format(macs_g)}  # 核心网络，不含 STFT/ISTFT")

    else:
        file_path = args.e2e_file
        class_name = "GTCRN"
        if args.spec_file and os.path.exists(args.spec_file):
            file_path = args.spec_file
        GT = _import_class(file_path, class_name)
        model = _build_model(GT, n_fft=args.n_fft, hop_len=args.hop, win_len=args.win).to(device).eval()
        params_m = count_params(model) / 1e6

        print("=== GTCRN (End-to-End) ===")
        print(f"Loaded  : file='{file_path}', class='{class_name}'")
        print(f"Para. (M):  {clever_format(params_m)}")
        print("MACs (G/s): N/A  （端到端包含 STFT/ISTFT；按论文口径仅统计频谱核心网络）")

if __name__ == "__main__":
    main()
