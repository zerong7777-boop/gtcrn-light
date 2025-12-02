# SEtrain/losses/spec_ri_mag.py
import torch
import torch.nn as nn

class SpecRIMAGLoss(nn.Module):
    """
    L = lamda_ri * MSE([real,imag]) + lamda_mag * MSE(|S|^c)
    - pred/target: (B, F, T, 2)
    """
    def __init__(self,
                 n_fft: int = 512,
                 hop_len: int = 256,
                 win_len: int = 512,
                 compress_factor: float = 0.3,
                 eps: float = 1e-12,
                 lamda_ri: float = 30.0,
                 lamda_mag: float = 70.0):
        super().__init__()
        self.cf = float(compress_factor)
        self.eps = float(eps)
        self.lri = float(lamda_ri)
        self.lmag = float(lamda_mag)
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        # RI MSE
        loss_ri = self.mse(pred, target)
        # Mag MSE (可压缩幅度)
        def mag_pow(x):
            # x: (B,F,T,2)
            real, imag = x.unbind(-1)
            mag = torch.sqrt(real*real + imag*imag + self.eps)
            return torch.pow(mag + self.eps, self.cf) if self.cf != 1.0 else mag
        loss_mag = self.mse(mag_pow(pred), mag_pow(target))
        return self.lri * loss_ri + self.lmag * loss_mag
