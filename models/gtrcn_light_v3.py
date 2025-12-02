# models/gtrcn_light_v3.py
# -*- coding: utf-8 -*-
"""
GTCRN-Light v3 — a lighter GTRCN that keeps the original pipeline:
ERB → SFE → Encoder(freq down ×2) → DPGRNN(intra→inter) → Decoder → ERB^-1 → CRM (RI).
Lightweight tricks: TRALite, DW-Separable, bottlenecked DPGRNN, ERB buffers.
"""

from typing import List
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# ERB (fixed filters as buffers; not counted as trainable params)
# -------------------------
class ERB(nn.Module):
    def __init__(self, erb_subband_1: int = 65, erb_subband_2: int = 64,
                 nfft: int = 512, high_lim: int = 8000, fs: int = 16000):
        super().__init__()
        self.erb_subband_1 = erb_subband_1
        self.erb_subband_2 = erb_subband_2
        self.nfreqs = nfft // 2 + 1

        # build triangular filter bank (64 bands on high part)
        W = self._erb_filters(erb_subband_1, erb_subband_2, nfft, high_lim, fs)  # [erb2, nfreqs - erb1]
        # register buffers
        self.register_buffer("W_bm", W.t())          # (erb2, F_high)
        self.register_buffer("W_bs", W)      # (F_high, erb2)

    @staticmethod
    def _hz2erb(freq_hz): return 21.4 * np.log10(0.00437 * freq_hz + 1)
    @staticmethod
    def _erb2hz(erb_f):   return (10 ** (erb_f / 21.4) - 1) / 0.00437

    def _erb_filters(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        low_lim = erb_subband_1 / nfft * fs
        erb_low  = self._hz2erb(low_lim); erb_high = self._hz2erb(high_lim)
        erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
        bins = np.round(self._erb2hz(erb_points)/fs*nfft).astype(np.int32)
        fb = np.zeros([erb_subband_2, nfft // 2 + 1], dtype=np.float32)

        fb[0,   bins[0]:bins[1]] = (bins[1] - np.arange(bins[0], bins[1]) + 1e-12) / (bins[1] - bins[0] + 1e-12)
        for i in range(erb_subband_2 - 2):
            fb[i+1, bins[i]:bins[i+1]]     = (np.arange(bins[i], bins[i+1]) - bins[i] + 1e-12) / (bins[i+1]-bins[i] + 1e-12)
            fb[i+1, bins[i+1]:bins[i+2]]   = (bins[i+2] - np.arange(bins[i+1], bins[i+2]) + 1e-12) / (bins[i+2]-bins[i+1] + 1e-12)
        fb[-1,  bins[-2]:bins[-1]+1] = 1 - fb[-2, bins[-2]:bins[-1]+1]
        fb = fb[:, erb_subband_1:]  # keep high part only
        return torch.from_numpy(np.abs(fb))  # (erb2, F_high)

    def bm(self, x):  # x: (B,C,T,F)
        x_low = x[..., :self.erb_subband_1]
        x_high = x[..., self.erb_subband_1:]  # (B,C,T,F_high=192)
        x_high_erb = torch.matmul(x_high, self.W_bm)  # @ (192,64) -> (B,C,T,64)
        return torch.cat([x_low, x_high_erb], dim=-1)  # -> (B,C,T,129)

    def bs(self, x_erb):  # x_erb: (B,C,T,129)
        low = x_erb[..., :self.erb_subband_1]
        high = x_erb[..., self.erb_subband_1:]  # (B,C,T,64)
        high_lin = torch.matmul(high, self.W_bs)  # @ (64,192) -> (B,C,T,192)
        return torch.cat([low, high_lin], dim=-1)  # -> (B,C,T,257)


# -------------------------
# TRALite: temporal depthwise gating (no RNN; tiny params)
# -------------------------
class TRALite(nn.Module):
    def __init__(self, channels: int, k: int = 3):
        super().__init__()
        self.dw = nn.Conv1d(channels, channels, kernel_size=k, padding=k//2, groups=channels, bias=True)
        self.pw = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x):  # x: (B,C,T,F)
        e = x.pow(2).mean(dim=3)                 # (B,C,T)  energy over freq
        g = self.pw(self.dw(e))                  # (B,C,T)
        g = self.act(g).unsqueeze(-1)            # (B,C,T,1)
        return x * g

# -------------------------
# DW-Separable building blocks
# -------------------------
class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=(1,3), s=(1,1), p=(0,1)):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=k, stride=s, padding=p, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.PReLU()
    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))

class DSDeconv(nn.Module):
    def __init__(self, in_ch, out_ch, k=(1,3), s=(1,2), p=(0,1), op=(0,0)):  # ← 这里从 (0,1) 改成 (0,0)
        super().__init__()
        # depthwise transposed conv
        self.dw = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=k, stride=s, padding=p, output_padding=op, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.PReLU()
    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))

class GTConvLite(nn.Module):
    """
    A lighter GT-Conv: DSConv (dilated) + TRALite + residual shuffle-like fuse
    """
    def __init__(self, ch: int, dilation: int):
        super().__init__()
        self.ds = nn.Conv2d(ch, ch, kernel_size=(3,3), padding=(dilation,1), dilation=(dilation,1),
                            groups=ch, bias=False)        # depthwise dilated
        self.pw = nn.Conv2d(ch, ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.act= nn.PReLU()
        self.tra = TRALite(ch)

    def forward(self, x):  # (B,ch,T,F)
        h = self.act(self.bn(self.pw(self.ds(x))))
        h = self.tra(h)
        return h + x

# -------------------------
# SFE (light): depthwise 1×3 on freq to gather local subband context
# -------------------------
class SFE_Lite(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=(1,3), padding=(0,1), groups=in_ch, bias=False)
    def forward(self, x):  # (B,3,T,F)
        return self.dw(x)  # keep 3ch, but with subband mixing

# -------------------------
# DPGRNN (bottlenecked intra→inter)
# -------------------------
class DPGRNN_Bottleneck(nn.Module):
    def __init__(self, c: int, r: int, bidir_intra: bool = True):
        super().__init__()
        r = max(8, r)  # safety
        self.c = c; self.r = r
        # proj
        self.pre  = nn.Linear(c, r, bias=False)
        self.post = nn.Linear((2 if bidir_intra else 1) * r, c, bias=False)  # intra输出回C
        self.post2= nn.Linear(r, c, bias=False)                               # inter输出回C

        # RNNs
        self.intra = nn.GRU(r, r, num_layers=1, batch_first=True, bidirectional=bidir_intra)
        self.inter = nn.GRU(r, r, num_layers=1, batch_first=True, bidirectional=False)

        # norm（只按通道 C，避免与 F 尺寸绑定）
        self.ln1 = nn.LayerNorm(c, eps=1e-8)
        self.ln2 = nn.LayerNorm(c, eps=1e-8)

    def forward(self, x):  # x: (B,C,T,F)
        B, C, T, F = x.shape

        # -------- intra (per time step, sequence along F) --------
        h = x.permute(0, 2, 3, 1).reshape(B * T, F, C)  # (B*T, F, C)
        h = self.pre(h)  # (B*T, F, r)
        h, _ = self.intra(h)  # (B*T, F, r*(1+bidir))
        h = self.post(h).reshape(B, T, F, C).permute(0, 3, 1, 2)  # (B,C,T,F)

        # 残差后先把 C 放到最后一维，再做 LN(C)
        y = (x + h).permute(0, 2, 3, 1)  # (B,T,F,C)
        y = self.ln1(y)  # 归一化通道 C
        y = y.permute(0, 3, 1, 2)  # 回到 (B,C,T,F)

        # -------- inter (per freq bin, sequence along T) --------
        z = y.permute(0, 3, 2, 1).reshape(B * F, T, C)  # (B*F, T, C)
        z = self.pre(z)  # (B*F, T, r)
        z, _ = self.inter(z)  # (B*F, T, r)
        z = self.post2(z).reshape(B, F, T, C).permute(0, 3, 2, 1)  # (B,C,T,F)

        out = (y + z).permute(0, 2, 3, 1)  # (B,T,F,C)
        out = self.ln2(out)  # 再按通道做 LN
        out = out.permute(0, 3, 1, 2)  # 回到 (B,C,T,F)
        return out


# -------------------------
# Encoder / Decoder (freq down ×2 / up ×2, with DW-Separable)
# -------------------------
class Encoder(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            # stem: 3→ch, down F /2
            DSConv(3, ch, k=(1,3), s=(1,2), p=(0,1)),
            # second down: keep ch, down F /2
            DSConv(ch, ch, k=(1,3), s=(1,2), p=(0,1)),
            # GTConv stacks (dilated)
            GTConvLite(ch, dilation=1),
            GTConvLite(ch, dilation=2),
            GTConvLite(ch, dilation=5),
        ])
    def forward(self, x):
        outs = []
        for b in self.blocks:
            x = b(x); outs.append(x)
        return x, outs

class Decoder(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            GTConvLite(ch, dilation=5),
            GTConvLite(ch, dilation=2),
            GTConvLite(ch, dilation=1),
            DSDeconv(ch, ch, k=(1,3), s=(1,2), p=(0,1), op=(0,0)),  # 33 → 65
            DSDeconv(ch, 2,  k=(1,3), s=(1,2), p=(0,1), op=(0,0)),  # 65 → 129
        ])
    def forward(self, x, enc_outs: List[torch.Tensor]):
        L = len(self.blocks)
        for i in range(L):
            # 对称相加（与 GTRCN 解码一致思想）：逐层加回 encoder 的对应输出
            x = self.blocks[i](x + enc_outs[L-1-i])
        return x

# -------------------------
# CRM head
# -------------------------
class ApplyMask(nn.Module):
    def forward(self, mask, spec):  # both (B,2,T,F)
        s_r = spec[:,0] * mask[:,0] - spec[:,1] * mask[:,1]
        s_i = spec[:,1] * mask[:,0] + spec[:,0] * mask[:,1]
        return torch.stack([s_r, s_i], dim=1)

# -------------------------
# Top-level: GTCRN_light (v3)
# -------------------------
class GTCRN_light_v3(nn.Module):
    """
    Args (kept for your CLI):
      - width_mult: scales channel width (base C=12)
      - use_two_dpgrnn: stack two DPGRNN blocks (like original "×2")
      - rnn_bidirectional: intra path bidirectional or not
    """
    def __init__(self, width_mult: float = 1.0, use_two_dpgrnn: bool = False, rnn_bidirectional: bool = True):
        super().__init__()
        base_c = 12
        ch = max(8, int(round(base_c * float(width_mult))))  # lighter than 16
        self.dp_width = 33  # ERB 129 after /2 /2 along F → 33

        self.erb = ERB(65, 64)
        self.sfe = SFE_Lite(in_ch=3)

        self.enc = Encoder(ch)
        # bottleneck dim r（瓶颈，极大降低 RNN 参数；参考 v1/v2 “瓶颈 GRU”思路）
        r = max(8, int(math.ceil(ch * 0.75)))
        self.dp1 = DPGRNN_Bottleneck(c=ch, r=r, bidir_intra=rnn_bidirectional)
        self.use_two = use_two_dpgrnn
        if self.use_two:
            self.dp2 = DPGRNN_Bottleneck(c=ch, r=r, bidir_intra=rnn_bidirectional)

        self.dec  = Decoder(ch)
        self.mask = ApplyMask()

    def forward(self, spec):  # spec: (B,F,T,2)
        # prepare 3-channel input (|S|, Re, Im)
        s_r = spec[...,0].permute(0,2,1)    # (B,T,F)
        s_i = spec[...,1].permute(0,2,1)
        s_m = torch.sqrt(s_r*s_r + s_i*s_i + 1e-12)
        x3  = torch.stack([s_m, s_r, s_i], dim=1)     # (B,3,T,F)

        x3 = self.erb.bm(x3)                         # (B,3,T,129)
        x3 = self.sfe(x3)                             # SFE-lite (B,3,T,129)

        z, enc_outs = self.enc(x3)                    # (B,ch,T,33)
        z = self.dp1(z)
        if self.use_two:
            z = self.dp2(z)

        m_erb = self.dec(z, enc_outs)                 # (B,2,T,129)
        m_lin = self.erb.bs(m_erb)                    # (B,2,T,F)

        spec2 = spec.permute(0,3,2,1)                 # (B,2,T,F)
        out   = self.mask(m_lin, spec2).permute(0,3,2,1)  # (B,F,T,2)
        return out
