# SEtrain/dataloader_spec.py
import os, random, glob
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset as TorchDataset

class PairSpecDataset(TorchDataset):
    """
    频谱版数据集：scp 每行 "<noisy.wav> <clean.wav>"
    输出：noisy_spec, clean_spec 形状 (F, T, 2) ，float32
    """
    def __init__(self,
                 scp_path: str,
                 fs: int = 16000,
                 length_in_seconds: float = 4.0,
                 n_fft: int = 512,
                 hop_len: int = 256,
                 win_len: int = 512,
                 random_start_point: bool = True,
                 num_data_per_epoch: int | None = None,
                 train: bool = True):
        super().__init__()
        assert os.path.isfile(scp_path), f"SCP not found: {scp_path}"
        self.fs = int(fs)
        self.L = int(round(length_in_seconds * self.fs))
        self.n_fft, self.hop_len, self.win_len = int(n_fft), int(hop_len), int(win_len)
        self.random_start_point = bool(random_start_point) and train
        self.num_data_per_epoch = int(num_data_per_epoch) if (num_data_per_epoch and train) else None
        self.train = bool(train)

        self.pairs = []
        with open(scp_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line: continue
                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(f"Line {i+1} invalid: {line}")
                n, c = parts[0], parts[1]
                if not (os.path.isfile(n) and os.path.isfile(c)):
                    raise FileNotFoundError(f"Missing file in line {i+1}: {n} | {c}")
                self.pairs.append((n, c))

        # 每轮采样子集
        self.indices_epoch = list(range(len(self.pairs)))
        if self.train and self.num_data_per_epoch:
            self.sample_data_per_epoch()

        # 预先建窗（PyTorch STFT 期望 1D window）
        self.window = torch.hann_window(self.win_len).pow(0.5)

    def sample_data_per_epoch(self):
        if (not self.train) or (self.num_data_per_epoch is None):
            self.indices_epoch = list(range(len(self.pairs)))
        else:
            N = len(self.pairs)
            k = min(self.num_data_per_epoch, N)
            self.indices_epoch = random.sample(range(N), k)

    def __len__(self):
        return len(self.indices_epoch)

    def _load_wav_mono(self, path: str):
        x, fs = sf.read(path, dtype="float32", always_2d=False)
        if fs != self.fs:
            raise ValueError(f"Sample rate mismatch: {fs} vs {self.fs} ({path})")
        if x.ndim == 2: x = x.mean(axis=1)
        return x

    def _cut_or_pad(self, x: np.ndarray, start: int):
        end = start + self.L
        seg = x[start:end]
        if len(seg) < self.L:
            seg = np.pad(seg, (0, self.L - len(seg)), mode="constant")
        return seg

    def _stft_ri(self, x_1d: np.ndarray):
        # (F, T, 2) 复谱（实虚对）
        x = torch.from_numpy(x_1d)
        S = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_len, win_length=self.win_len,
                       window=self.window, center=True, return_complex=True)
        ri = torch.view_as_real(S).to(torch.float32)  # (F, T, 2)
        return ri

    def __getitem__(self, idx: int):
        i = self.indices_epoch[idx]
        noisy_p, clean_p = self.pairs[i]
        noisy = self._load_wav_mono(noisy_p)
        clean = self._load_wav_mono(clean_p)

        max_start = max(0, min(len(noisy), len(clean)) - self.L)
        start = (random.randint(0, max_start) if (self.random_start_point and max_start > 0) else 0)

        n_seg = self._cut_or_pad(noisy, start)
        c_seg = self._cut_or_pad(clean, start)

        n_spec = self._stft_ri(n_seg)
        c_spec = self._stft_ri(c_seg)
        return n_spec, c_spec

    @staticmethod
    def collate_fn(batch):
        ns, cs = zip(*batch)
        return torch.stack(ns, 0), torch.stack(cs, 0)
