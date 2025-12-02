# models/gtrcn_light.py
# -*- coding: utf-8 -*-
"""
GTCRN-Light (Spec-domain) — CRM + UNet + Time-Consistent Upsampling
保持与原 gtcrn 接口一致：
    forward(spec_ri: (B,F,T,2)) -> (B,F,T,2)

设计要点：
1) Complex Ratio Mask (CRM)：输出掩膜经 tanh 限幅，Ŝ = tanh(M) ⊙ S，稳健提升 PESQ/STOI。
2) 仅在时间轴下/上采样；(1,2) 下采样两次 + ConvTranspose(1,4, stride=2, pad=1) 两次上采样，
   精确还原 T，F 全程不变（=257），消除形状对齐的损失。
3) U-Net 跳连：编码多尺度信息传递至解码端，轻量网络也能保细节。
4) 轻量 RNN 记忆：频率展开、时间向 GRU（可 1/2 层），在小参数下维持时序一致性。
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# 基础模块
# -------------------------
class DWSeparableConv2d(nn.Module):
    """Depthwise + Pointwise，轻量卷积块"""
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
    """轻量残差：DW-Separable + 1x1 调整"""
    def __init__(self, ch):
        super().__init__()
        self.conv1 = DWSeparableConv2d(ch, ch, k=(3,3), s=(1,1), p=(1,1))
        self.conv2 = DWSeparableConv2d(ch, ch, k=(3,3), s=(1,1), p=(1,1))
    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class TemporalGate(nn.Module):
    """时间轴轻量门控（1x1 Conv + Sigmoid）"""
    def __init__(self, ch):
        super().__init__()
        self.proj = nn.Conv2d(ch, ch, kernel_size=1, bias=True)
    def forward(self, x):
        g = torch.sigmoid(self.proj(x))
        return x * g


# -------------------------
# RNN：频率展开，时间向 GRU
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
# 仅在时间轴 stride=2 下采样两次；频轴 stride=1，F 保持不变
# -------------------------
class Encoder(nn.Module):
    def __init__(self, base_c: int):
        super().__init__()
        c = base_c
        self.stem = DWSeparableConv2d(2, c, k=(3,3), s=(1,1), p=(1,1))   # (B,2,F,T)->(B,c,F,T)
        # 下采样块：时间轴 /2，两次后 T/4
        self.down1 = nn.Sequential(
            DWSeparableConv2d(c,   c, k=(3,3), s=(1,2), p=(1,1)),  # T/2
            ResBlock(c)
        )
        self.down2 = nn.Sequential(
            DWSeparableConv2d(c,   c, k=(3,3), s=(1,2), p=(1,1)),  # T/4
            ResBlock(c)
        )
    def forward(self, x):
        e0 = self.stem(x)        # (B,c,F,T)
        e1 = self.down1(e0)      # (B,c,F,T/2)
        e2 = self.down2(e1)      # (B,c,F,T/4)
        return e0, e1, e2


class Decoder(nn.Module):
    """
    反卷积参数固定：
      kernel=(1,4), stride=(1,2), padding=(0,1), output_padding=(0,0)
    可精确还原时间长度：T/4 -> T/2 -> T
    """
    def __init__(self, base_c: int):
        super().__init__()
        c = base_c
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(c, c, kernel_size=(1,4), stride=(1,2),
                               padding=(0,1), bias=False),
            nn.BatchNorm2d(c), nn.PReLU(c)
        )  # T/4 -> T/2
        self.refine1 = DWSeparableConv2d(c*2, c, k=(3,3), s=(1,1), p=(1,1))  # + 跳连 e1

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(c, c, kernel_size=(1,4), stride=(1,2),
                               padding=(0,1), bias=False),
            nn.BatchNorm2d(c), nn.PReLU(c)
        )  # T/2 -> T
        self.refine2 = DWSeparableConv2d(c*2, c, k=(3,3), s=(1,1), p=(1,1))  # + 跳连 e0

        # 输出 2 通道的 CRM（对 RI 的 mask），后续 tanh
        self.out_head = nn.Conv2d(c, 2, kernel_size=1, bias=True)

    def forward(self, e0, e1, e2):
        x = self.up1(e2)                                 # (B,c,F,T/2)
        # 对齐拼接（理论上 T/2、F 相同；如遇奇偶差，可做裁/填）
        if x.size(-1) != e1.size(-1):                    # 时间对齐
            T = min(x.size(-1), e1.size(-1))
            x  = x[..., :T]
            e1 = e1[..., :T]
        x = torch.cat([x, e1], dim=1)                    # (B,2c,F,T/2)
        x = self.refine1(x)                              # -> (B,c,F,T/2)

        x = self.up2(x)                                  # (B,c,F,T)
        if x.size(-1) != e0.size(-1):
            T = min(x.size(-1), e0.size(-1))
            x  = x[..., :T]
            e0 = e0[..., :T]
        x = torch.cat([x, e0], dim=1)                    # (B,2c,F,T)
        x = self.refine2(x)                              # -> (B,c,F,T)

        m = self.out_head(x)                             # (B,2,F,T)  (CRM)
        return m


# -------------------------
# 顶层：GTCRN_light（Spec）
# -------------------------
class GTCRN_light(nn.Module):
    def __init__(self,
                 width_mult: float = 1.0,
                 use_two_dpgrnn: bool = True):
        super().__init__()
        c = max(12, int(round(16 * float(width_mult))))
        rnn_h = max(16, int(round(32 * float(width_mult))))

        self.enc = Encoder(base_c=c)
        self.gate1 = TemporalGate(c)
        self.rnn1  = GRNN(input_size=c, hidden_size=rnn_h, bidirectional=True)

        self.use_two = bool(use_two_dpgrnn)
        if self.use_two:
            self.gate2 = TemporalGate(c)
            self.rnn2  = GRNN(input_size=c, hidden_size=rnn_h//1, bidirectional=False)

        self.dec = Decoder(base_c=c)

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
        assert spec_ri.dim()==4 and spec_ri.size(-1)==2
        x0 = self._to_nchw(spec_ri)                   # (B,2,F,T)

        # 编码
        e0, e1, e2 = self.enc(x0)                    # (B,c,F,T),(B,c,F,T/2),(B,c,F,T/4)

        # 轻量时序记忆
        h = self.gate1(e2)
        h = self.rnn1(h)
        if self.use_two:
            h = self.gate2(h)
            h = self.rnn2(h)

        # 解码得到 CRM
        m = self.dec(e0, e1, h)                      # (B,2,F,T)
        m = torch.tanh(m)                            # 限幅到 [-1,1]

        # 应用复数掩膜：Ŝ = M ⊙ S
        enh = m * x0                                 # (B,2,F,T)
        return self._to_bft2(enh)                    # (B,F,T,2)


# 便捷 sanity check
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=2)
    p.add_argument("--F", type=int, default=257)
    p.add_argument("--T", type=int, default=63)
    p.add_argument("--width-mult", type=float, default=1.0)
    p.add_argument("--two-rnn", action="store_true")
    args = p.parse_args()

    m = GTCRN_light(width_mult=args.width_mult, use_two_dpgrnn=args.two_rnn)
    x = torch.randn(args.B, args.F, args.T, 2)
    y = m(x)
    print("in :", tuple(x.shape))
    print("out:", tuple(y.shape))
