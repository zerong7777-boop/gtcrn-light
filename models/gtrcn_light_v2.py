# models/gtrcn_light_v2.py
# -*- coding: utf-8 -*-
"""
GTCRN-Light v2 (Spec-domain)
目标：在不“改爆”结构的前提下进一步降低 MACs，同时维持与原 GTCRN-Light 相同的 I/O 接口：
    forward(spec_ri: (B,F,T,2)) -> (B,F,T,2)

关键微调（与 v1 相比）：
1) 频轴一次性下采样/2（Encoder）+ 对称上采样/2（Decoder）
   - 编码早期以 stride=(2,1) 将 F: 257→129，后续主干计算都在较小频宽上进行；
   - 解码末端用 ConvTranspose2d(kernel=(3,1), stride=(2,1), padding=(1,0)) 精确恢复到原 F（奇数 257）。
   → 在保持 UNet 跳连与 CRM 框架下，显著降低卷积与 RNN 的计算量（F×T 主导）。

2) GRU 放置在最底尺度的特征图上（F/2, T/4）
   - 与 v1 相比，GRU 的时空分辨率更低；计算成本约降至原来的 1/8 左右；
   - 默认使用单层双向 GRU（可配置），兼顾时序一致性与感知指标。

3) 其它设计保持不变：CRM（tanh 限幅）+ 轻量 DW-Separable Conv + 跳连与时间一致性上采样。

用法：
    from models.gtrcn_light_v2 import GTCRN_light_v2 as Model
    y = Model(width_mult=1.0, use_two_dpgrnn=False)(x)   # x: (B,F,T,2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# 基础模块
# -------------------------
class DWSeparableConv2d(nn.Module):
    """Depthwise + Pointwise 的轻量卷积块"""
    def __init__(self, in_ch, out_ch, k=(3,3), s=(1,1), p=(1,1)):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.PReLU(out_ch)
    def forward(self, x):
        x = self.dw(x); x = self.pw(x); x = self.bn(x); x = self.act(x)
        return x


class ResBlock(nn.Module):
    """轻量残差块：两次 DW-Separable"""
    def __init__(self, ch):
        super().__init__()
        self.conv1 = DWSeparableConv2d(ch, ch, k=(3,3), s=(1,1), p=(1,1))
        self.conv2 = DWSeparableConv2d(ch, ch, k=(3,3), s=(1,1), p=(1,1))
    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class TemporalGate(nn.Module):
    """时间轴门控（1x1 Conv + Sigmoid）"""
    def __init__(self, ch):
        super().__init__()
        self.proj = nn.Conv2d(ch, ch, kernel_size=1, bias=True)
    def forward(self, x):
        g = torch.sigmoid(self.proj(x))
        return x * g


# -------------------------
# 频率展开 + 时间向 GRU
# -------------------------
class GRNN(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=True):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1,
                          batch_first=True, bidirectional=bidirectional)
        out_ch = hidden_size * (2 if bidirectional else 1)
        self.proj = nn.Linear(out_ch, input_size)
    def forward(self, x):
        # x: (B,C,F,T) -> (B*F, T, C)
        B,C,F,T = x.shape
        h = x.permute(0,2,3,1).contiguous().view(B*F, T, C)
        y,_ = self.gru(h)
        y = self.proj(y)                                 # (B*F, T, C)
        y = y.view(B, F, T, C).permute(0,3,1,2).contiguous()
        return y


# -------------------------
# Encoder / Decoder
# v2: 在 stem 之后加入一次“频轴/2”下采样；解码末尾对称“频轴×2”上采样
#     时间轴仍是两次 /2（T/4）与对称反卷积还原（保持与 v1 一致的时间对齐策略）
# -------------------------
class Encoder_v2(nn.Module):
    def __init__(self, base_c: int):
        super().__init__()
        c = base_c
        self.stem = DWSeparableConv2d(2, c, k=(3,3), s=(1,1), p=(1,1))       # (B,2,F,T)->(B,c,F,T)
        # 频轴下采样 /2：stride=(2,1) 仅压 F
        self.downF = DWSeparableConv2d(c, c, k=(3,3), s=(2,1), p=(1,1))      # (B,c,F/2,T)
        # 时间轴两次 /2：stride=(1,2) 仅压 T
        self.downT1 = nn.Sequential(
            DWSeparableConv2d(c, c, k=(3,3), s=(1,2), p=(1,1)),              # (B,c,F/2,T/2)
            ResBlock(c)
        )
        self.downT2 = nn.Sequential(
            DWSeparableConv2d(c, c, k=(3,3), s=(1,2), p=(1,1)),              # (B,c,F/2,T/4)
            ResBlock(c)
        )
    def forward(self, x):
        e0 = self.stem(x)                 # (B,c,F,T)     — 最浅层（用作最终跳连）
        eF = self.downF(e0)               # (B,c,F/2,T)   — 频轴一半
        e1 = self.downT1(eF)              # (B,c,F/2,T/2)
        e2 = self.downT2(e1)              # (B,c,F/2,T/4)
        return e0, e1, e2                 # （注意：e0 是全 F；e1/e2 是 F/2）


class Decoder_v2(nn.Module):
    """
    时间反卷积参数维持与 v1 一致：
      kernel=(1,4), stride=(1,2), padding=(0,1)  → 精确 T/2 上采样
    频轴恢复使用：
      ConvTranspose2d(kernel=(3,1), stride=(2,1), padding=(1,0)) → 可从 129 精确到 257（奇数）
    """
    def __init__(self, base_c: int):
        super().__init__()
        c = base_c

        # T: /4 -> /2
        self.upT1 = nn.Sequential(
            nn.ConvTranspose2d(c, c, kernel_size=(1,4), stride=(1,2),
                               padding=(0,1), bias=False),
            nn.BatchNorm2d(c), nn.PReLU(c)
        )
        self.refine1 = DWSeparableConv2d(c*2, c, k=(3,3), s=(1,1), p=(1,1))  # 与 e1 (F/2,T/2) 融合

        # T: /2 -> /1
        self.upT2 = nn.Sequential(
            nn.ConvTranspose2d(c, c, kernel_size=(1,4), stride=(1,2),
                               padding=(0,1), bias=False),
            nn.BatchNorm2d(c), nn.PReLU(c)
        )
        # 频轴恢复到原 F（由 F/2 → F）
        self.upF  = nn.Sequential(
            nn.ConvTranspose2d(c, c, kernel_size=(3,1), stride=(2,1),
                               padding=(1,0), bias=False),
            nn.BatchNorm2d(c), nn.PReLU(c)
        )

        # 与 e0 (F,T) 融合并细化
        self.refine2 = DWSeparableConv2d(c*2, c, k=(3,3), s=(1,1), p=(1,1))

        # 输出 2 通道的 CRM（RI 掩膜），后续 tanh
        self.out_head = nn.Conv2d(c, 2, kernel_size=1, bias=True)

    def forward(self, e0, e1, h_bottleneck):
        # h_bottleneck: (B,c,F/2,T/4) — 经过底部 RNN 的特征
        x = self.upT1(h_bottleneck)                       # (B,c,F/2,T/2)
        # T 对齐拼接（防止偶数/奇数引起的 1 帧差异）
        if x.size(-1) != e1.size(-1):
            T = min(x.size(-1), e1.size(-1))
            x  = x[..., :T]
            e1 = e1[..., :T]
        x = torch.cat([x, e1], dim=1)                     # (B,2c,F/2,T/2)
        x = self.refine1(x)                               # (B,c,F/2,T/2)

        x = self.upT2(x)                                  # (B,c,F/2,T)
        # 频轴恢复到原 F
        x = self.upF(x)                                   # (B,c,F,T)

        # 与最浅层 e0 (B,c,F,T) 融合
        # 若 F/T 有 1 像素差，做裁剪对齐
        if x.size(-1) != e0.size(-1):
            T = min(x.size(-1), e0.size(-1))
            x  = x[..., :T]
            e0 = e0[..., :T]
        if x.size(-2) != e0.size(-2):
            F = min(x.size(-2), e0.size(-2))
            x  = x[:, :, :F]
            e0 = e0[:, :, :F]
        x = torch.cat([x, e0], dim=1)                     # (B,2c,F,T)
        x = self.refine2(x)                               # (B,c,F,T)

        m = self.out_head(x)                              # (B,2,F,T)  (CRM)
        return m


# -------------------------
# 顶层：GTCRN_light_v2（Spec）
# -------------------------
class GTCRN_light_v2(nn.Module):
    def __init__(self,
                 width_mult: float = 1.0,
                 use_two_dpgrnn: bool = False,         # v2 默认单层 GRU，进一步减 MACs
                 rnn_bidirectional: bool = True):
        super().__init__()
        c = max(12, int(round(16 * float(width_mult))))
        rnn_h = max(16, int(round(32 * float(width_mult))))

        self.enc = Encoder_v2(base_c=c)

        # 底部：轻量时序记忆（在 F/2, T/4 上运行）
        self.gate1 = TemporalGate(c)
        self.rnn1  = GRNN(input_size=c, hidden_size=rnn_h,
                          bidirectional=bool(rnn_bidirectional))

        self.use_two = bool(use_two_dpgrnn)
        if self.use_two:
            # 可选第二层（默认关闭，若追求感知指标可打开）
            self.gate2 = TemporalGate(c)
            # 第二层可设为单向，进一步降算力
            self.rnn2  = GRNN(input_size=c, hidden_size=rnn_h//1,
                              bidirectional=False)

        self.dec = Decoder_v2(base_c=c)

    @staticmethod
    def _to_nchw(spec_ri: torch.Tensor) -> torch.Tensor:
        # (B,F,T,2) -> (B,2,F,T)
        return spec_ri.permute(0,3,1,2).contiguous()

    @staticmethod
    def _to_bft2(x: torch.Tensor) -> torch.Tensor:
        # (B,2,F,T) -> (B,F,T,2)
        return x.permute(0,2,3,1).contiguous()

    def forward(self, spec_ri: torch.Tensor) -> torch.Tensor:
        """
        spec_ri: (B,F,T,2) -> enh_ri: (B,F,T,2)
        """
        assert spec_ri.dim()==4 and spec_ri.size(-1)==2, "Input must be (B,F,T,2) RI-spec"
        x0 = self._to_nchw(spec_ri)                      # (B,2,F,T)

        # 编码：返回 e0(F,T), e1(F/2,T/2), e2(F/2,T/4)
        e0, e1, e2 = self.enc(x0)                        # (B,c,*,*)
        # 底部时序记忆（更低分辨率，显著降 MACs）
        h = self.gate1(e2)
        h = self.rnn1(h)
        if self.use_two:
            h = self.gate2(h)
            h = self.rnn2(h)

        # 解码得到 CRM（B,2,F,T）
        m = self.dec(e0, e1, h)
        m = torch.tanh(m)                                # 限幅 [-1,1]

        # 应用复数掩膜：Ŝ = M ⊙ S
        enh = m * x0                                     # (B,2,F,T)
        return self._to_bft2(enh)                        # (B,F,T,2)


# 便捷自测
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=2)
    p.add_argument("--F", type=int, default=257)
    p.add_argument("--T", type=int, default=63)
    p.add_argument("--width-mult", type=float, default=1.0)
    p.add_argument("--two-rnn", action="store_true")
    p.add_argument("--uni", action="store_true", help="use unidirectional GRU at bottom")
    args = p.parse_args()

    m = GTCRN_light_v2(width_mult=args.width_mult,
                       use_two_dpgrnn=args.two_rnn,
                       rnn_bidirectional=(not args.uni))
    x = torch.randn(args.B, args.F, args.T, 2)
    y = m(x)
    print("Input :", tuple(x.shape))
    print("Output:", tuple(y.shape))
    if y.shape != x.shape:
        print(f"[WARN] shape mismatch: ΔF={y.size(1)-x.size(1)}, ΔT={y.size(2)-x.size(2)}")
