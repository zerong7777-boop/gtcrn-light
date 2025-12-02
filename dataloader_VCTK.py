# SEtrain/dataloader.py
import os
import random
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset as TorchDataset

class PairWavDataset(TorchDataset):
    """
    通用配对波形数据集（适配 VoiceBank-DEMAND 等），基于 scp 列表：
      每行：<noisy_path> <clean_path>

    返回：
      noisy:  (L,)  float32
      clean:  (L,)  float32
    """
    def __init__(
        self,
        scp_path: str,
        fs: int = 16000,
        length_in_seconds: float = 4.0,
        random_start_point: bool = True,
        num_data_per_epoch: int | None = None,
        train: bool = True,
    ):
        super().__init__()

        assert os.path.isfile(scp_path), f"SCP not found: {scp_path}"
        self.fs = int(fs)
        self.L = int(round(length_in_seconds * self.fs))  # 目标样本点数
        self.random_start_point = bool(random_start_point) and train
        self.num_data_per_epoch = int(num_data_per_epoch) if (num_data_per_epoch and train) else None
        self.train = bool(train)

        # 读取 scp 列表
        self.pairs: list[tuple[str, str]] = []
        with open(scp_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(f"Line {i+1} in {scp_path} is invalid: `{line}`")
                noisy, clean = parts[0], parts[1]
                if (not os.path.isfile(noisy)) or (not os.path.isfile(clean)):
                    # 允许软链接、绝对/相对路径混用，最终都要能读到
                    raise FileNotFoundError(f"Missing file in line {i+1}: {noisy} or {clean}")
                self.pairs.append((noisy, clean))

        # 训练集可选：每个 epoch 随机抽样一个子集
        self.indices_epoch = list(range(len(self.pairs)))
        if self.train and self.num_data_per_epoch:
            self.sample_data_per_epoch()

        # 方便 DataLoader 取用的采样率（给写样例用）
        self.samplerate = self.fs

    # 兼容你现有的 train.py：若存在该方法，会在每个 epoch 前调用
    def sample_data_per_epoch(self):
        if (not self.train) or (self.num_data_per_epoch is None):
            self.indices_epoch = list(range(len(self.pairs)))
        else:
            N = len(self.pairs)
            k = min(self.num_data_per_epoch, N)
            self.indices_epoch = random.sample(range(N), k)

    def __len__(self):
        return len(self.indices_epoch)

    def __getitem__(self, idx: int):
        # 将 epoch 子集索引映射回原始 pairs
        i = self.indices_epoch[idx]
        noisy_path, clean_path = self.pairs[i]

        # 读取音频；若为多声道，取均值转单声道
        noisy, fs_n = sf.read(noisy_path, dtype="float32", always_2d=False)
        clean, fs_c = sf.read(clean_path, dtype="float32", always_2d=False)
        if fs_n != self.fs or fs_c != self.fs:
            raise ValueError(f"Sample rate mismatch: {fs_n}/{fs_c} vs {self.fs} ({noisy_path})")
        if noisy.ndim == 2:
            noisy = noisy.mean(axis=1)
        if clean.ndim == 2:
            clean = clean.mean(axis=1)

        # 统一长度：随机/固定起点裁切；不足补零
        n_len = len(noisy)
        c_len = len(clean)
        if self.random_start_point:
            # 找一个合法起点（保证剩余长度 >= L）；若原始长度不足 L，则起点=0
            max_start = max(0, min(n_len, c_len) - self.L)
            start = random.randint(0, max_start) if max_start > 0 else 0
        else:
            start = 0

        end = start + self.L
        noisy_seg = noisy[start:end]
        clean_seg = clean[start:end]

        # 右侧零填充到 L
        if len(noisy_seg) < self.L:
            noisy_seg = np.pad(noisy_seg, (0, self.L - len(noisy_seg)), mode="constant")
        if len(clean_seg) < self.L:
            clean_seg = np.pad(clean_seg, (0, self.L - len(clean_seg)), mode="constant")

        # 转张量
        noisy_t = torch.from_numpy(noisy_seg.astype(np.float32))
        clean_t = torch.from_numpy(clean_seg.astype(np.float32))
        return noisy_t, clean_t

    # 可选的 collate_fn（这里固定长度，默认堆叠即可；保留接口兼容 train.py 的 hasattr 判断）
    @staticmethod
    def collate_fn(batch):
        noisys, cleans = zip(*batch)
        return torch.stack(noisys, dim=0), torch.stack(cleans, dim=0)
