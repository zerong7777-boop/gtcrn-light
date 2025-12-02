#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch
import random
import shutil
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
from glob import glob
from pesq import pesq
from joblib import Parallel, delayed
import soundfile as sf
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from distributed_utils import reduce_value

# 频谱版模型（输入/输出均为复谱 (B,F,T,2)）
from models.gtrcn_light_v3 import GTCRN_light_v3 as Model

# 频谱损失（RI + Mag）
from losses.spec_ri_mag import SpecRIMAGLoss as Loss

# 频谱数据集：返回 (noisy_spec, clean_spec) 皆为 (F,T,2)
from dataloader_spec import PairSpecDataset as Dataset

# 学习率计划（线性预热 + 余弦退火）
from scheduler import LinearWarmupCosineAnnealingLR as WarmupLR


# =========================
# Debug helpers
# =========================
def count_params(m: torch.nn.Module):
    total = sum(p.numel() for p in m.parameters())
    train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, train

def _p0(msg, rank):
    """print only on rank-0"""
    if rank == 0:
        print(msg, flush=True)


# -------------------------
# 复现可控：固定随机种子
# -------------------------
seed = 43
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True  # 如需更高复现性可打开（性能略降）


# -------------------------
# 工具函数：频谱 ↔ 波形
# -------------------------
def ri_batch_to_wav(batch_ri, n_fft, hop, win_len, samplerate, seg_seconds, device):
    """
    将 (B,F,T,2) 的 [real,imag] 复谱批量还原为 (B,L) 波形。
    L = samplerate * seg_seconds
    """
    real = batch_ri[..., 0].to(torch.float32).contiguous()
    imag = batch_ri[..., 1].to(torch.float32).contiguous()
    S = torch.complex(real, imag)  # (B,F,T), complex64

    win = torch.hann_window(win_len).pow(0.5).to(device)
    L = int(round(float(samplerate) * float(seg_seconds)))
    y = torch.istft(
        S, n_fft=n_fft, hop_length=hop, win_length=win_len,
        window=win, center=True, length=L
    )  # (B,L), float32
    return y


def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def _match_spec_shape(pred: torch.Tensor, target: torch.Tensor, debug: bool=False, where: str=""):
    """
    pred/target: (B,F,T,2)
    将 pred 裁/填到与 target 相同大小，优先保持左上对齐（不引入时间偏移）。
    """
    if pred.shape == target.shape:
        return pred
    B, F_t, T_t, C = target.shape
    Bp, F_p, T_p, Cp = pred.shape
    assert B == Bp and C == Cp, f"Batch/chan mismatch: pred {pred.shape}, tgt {target.shape}"
    out = pred.new_zeros((B, F_t, T_t, C))
    F = min(F_p, F_t)
    T = min(T_p, T_t)
    out[:, :F, :T, :] = pred[:, :F, :T, :]
    if debug:
        print(f"[shape-fix{('/'+where) if where else ''}] pred {tuple(pred.shape)} -> out {tuple(out.shape)} (tgt {tuple(target.shape)})",
              flush=True)
    return out


def _freeze_by_keywords(model: torch.nn.Module, keywords):
    """根据关键字冻结参数；keywords 为字符串列表"""
    if not keywords:
        return 0
    n = 0
    for name, p in model.named_parameters():
        if any(kw in name for kw in keywords):
            p.requires_grad = False
            n += p.numel()
    return n


# -------------------------
# 训练主入口
# -------------------------
def run(rank, config, args):
    # ============ DDP init ============
    if args.world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12354'
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(rank)
        dist.barrier()

    args.rank = rank
    args.device = torch.device(rank if torch.cuda.is_available() else "cpu")

    # ============ Ablation config ============
    ab_cfg = dict(config.get('ablation', {}))
    ab_shape_fix     = bool(ab_cfg.get('shape_fix', True))
    ab_use_ri        = bool(ab_cfg.get('use_ri', True))
    ab_use_mag       = bool(ab_cfg.get('use_mag', True))
    ab_compute_pesq  = bool(ab_cfg.get('compute_pesq', True))
    ab_pesq_jobs     = int(ab_cfg.get('pesq_jobs', -1))

    # 冻结设置（使用关键词匹配，模块名不严格要求）
    fr_cfg = dict(config.get('freeze', {}))
    freeze_encoder = bool(fr_cfg.get('encoder', False))
    freeze_dpgrnn  = bool(fr_cfg.get('dpgrnn', False))
    freeze_decoder = bool(fr_cfg.get('decoder', False))

    # ============ Dataloader ============
    collate_fn = Dataset.collate_fn if hasattr(Dataset, "collate_fn") else None
    shuffle = False if args.world_size > 1 else True

    train_dataset = Dataset(**dict(config['train_dataset']))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.world_size > 1 else None
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        shuffle=(False if train_sampler is not None else shuffle),
        collate_fn=collate_fn,
        **dict(config['train_dataloader'])
    )

    validation_dataset = Dataset(**dict(config['validation_dataset']))
    validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset) if args.world_size > 1 else None
    validation_dataloader = torch.utils.data.DataLoader(
        dataset=validation_dataset,
        sampler=validation_sampler,
        shuffle=False,
        collate_fn=collate_fn,
        **dict(config['validation_dataloader'])
    )

    # === Print data & config summary (rank0) ===
    debug = bool(config.get('debug', False))
    if args.rank == 0:
        _p0("\n===== CONFIG SUMMARY =====", args.rank)
        _p0(f"network_config: {dict(config.get('network_config', {}))}", args.rank)
        _p0(f"FFT           : {dict(config.get('FFT', {}))}", args.rank)
        _p0(f"samplerate    : {config.get('samplerate', 16000)}", args.rank)
        _p0(f"train SCP     : {config['train_dataset'].get('scp_path','')}", args.rank)
        _p0(f"valid SCP     : {config['validation_dataset'].get('scp_path','')}", args.rank)
        _p0(f"train batches : {len(train_dataloader)}  (bs={config['train_dataloader']['batch_size']})", args.rank)
        _p0(f"valid  batches: {len(validation_dataloader)} (bs={config['validation_dataloader']['batch_size']})", args.rank)
        _p0(f"ablation      : shape_fix={ab_shape_fix}, use_ri={ab_use_ri}, use_mag={ab_use_mag}, "
            f"compute_pesq={ab_compute_pesq}, pesq_jobs={ab_pesq_jobs}", args.rank)
        _p0(f"freeze        : encoder={freeze_encoder}, dpgrnn={freeze_dpgrnn}, decoder={freeze_decoder}", args.rank)
        if debug:
            try:
                sample_noisy, sample_clean = next(iter(train_dataloader))
                _p0(f"sample noisy shape: {tuple(sample_noisy.shape)}  clean shape: {tuple(sample_clean.shape)}", args.rank)
            except Exception as e:
                _p0(f"[debug] sample batch failed: {e}", args.rank)

    # ============ Model ============
    net_cfg = dict(config.get('network_config', {}))
    try:
        model = Model(**net_cfg).to(args.device)
    except TypeError:
        model = Model().to(args.device)

    # 可选冻结
    if any([freeze_encoder, freeze_dpgrnn, freeze_decoder]):
        keywords = []
        if freeze_encoder: keywords += ['enc', 'encoder']
        if freeze_dpgrnn:  keywords += ['dpgrnn', 'dpgru', 'rnn', 'gru']
        if freeze_decoder: keywords += ['dec', 'decoder']
        n_freeze = _freeze_by_keywords(model, keywords)
        _p0(f"[freeze] #params frozen: {n_freeze}", args.rank)

    if args.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # 打印模型摘要（rank0）
    if args.rank == 0:
        tot, trn = count_params(model if args.world_size == 1 else model.module)
        _p0(f"\n===== MODEL SUMMARY =====", args.rank)
        _p0(f"Model class     : {model.__class__.__name__}", args.rank)
        _p0(f"Trainable params: {trn:,}  (total: {tot:,})", args.rank)
        _p0(f"Device(s)       : CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}", args.rank)

    # ============ Optim/Sched/Loss ============
    optimizer = torch.optim.Adam(params=model.parameters(), **dict(config['optimizer']))
    scheduler = WarmupLR(optimizer, **dict(config['scheduler']['kwargs']))
    loss_func = Loss(**dict(config['loss'])).to(args.device)

    # 按消融动态设权重
    if not ab_use_ri:
        if hasattr(loss_func, 'lamda_ri'): loss_func.lamda_ri = 0.0
    if not ab_use_mag:
        if hasattr(loss_func, 'lamda_mag'): loss_func.lamda_mag = 0.0

    # ============ Trainer ============
    trainer = Trainer(
        config=config, model=model, optimizer=optimizer, scheduler=scheduler, loss_func=loss_func,
        train_dataloader=train_dataloader, validation_dataloader=validation_dataloader,
        train_sampler=train_sampler, args=args,
        ablation=dict(shape_fix=ab_shape_fix, compute_pesq=ab_compute_pesq, pesq_jobs=ab_pesq_jobs),
        debug=debug
    )
    trainer.train()

    if args.world_size > 1:
        dist.destroy_process_group()


# -------------------------
# Trainer
# -------------------------
class Trainer:
    def __init__(self, config, model, optimizer, scheduler, loss_func,
                 train_dataloader, validation_dataloader, train_sampler, args,
                 ablation=None, debug=False):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_func = loss_func

        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.train_sampler = train_sampler

        self.rank = args.rank
        self.device = args.device
        self.world_size = args.world_size

        # === 训练配置兜底 ===
        if 'DDP' not in config or config['DDP'] is None:
            config['DDP'] = {}
        config['DDP']['world_size'] = args.world_size

        trainer_cfg = dict(config.get('trainer', {}))
        trainer_cfg.setdefault('epochs', 100)
        trainer_cfg.setdefault('save_checkpoint_interval', 1)
        trainer_cfg.setdefault('clip_grad_norm_value', 3.0)
        trainer_cfg.setdefault('exp_path', './runs/tmp_spec')
        trainer_cfg.setdefault('log_interval', 100)
        trainer_cfg.setdefault('val_interval', 1)
        trainer_cfg.setdefault('resume', False)
        trainer_cfg.setdefault('resume_datetime', datetime.now().strftime("%Y-%m-%d-%Hh%Mm"))
        self.trainer_config = trainer_cfg
        config['trainer'] = trainer_cfg  # 回写

        self.epochs = trainer_cfg['epochs']
        self.save_checkpoint_interval = trainer_cfg['save_checkpoint_interval']
        self.clip_grad_norm_value = trainer_cfg['clip_grad_norm_value']
        self.resume = trainer_cfg['resume']

        self.exp_path = (trainer_cfg['exp_path'] + '_' +
                         (trainer_cfg['resume_datetime'] if self.resume else datetime.now().strftime("%Y-%m-%d-%Hh%Mm")))
        self.log_path = os.path.join(self.exp_path, 'logs')
        self.checkpoint_path = os.path.join(self.exp_path, 'checkpoints')
        self.sample_path = os.path.join(self.exp_path, 'val_samples')
        self.code_path = os.path.join(self.exp_path, 'codes')
        for p in [self.log_path, self.checkpoint_path, self.sample_path, self.code_path]:
            _ensure_dir(p)

        # 保存配置与代码快照（rank0）
        if self.rank == 0:
            OmegaConf.save(OmegaConf.create(self.config), os.path.join(self.exp_path, 'config.yaml'))
            shutil.copy2(__file__, self.exp_path)
            for file in Path(__file__).parent.iterdir():
                if file.is_file():
                    shutil.copy2(file, self.code_path)
            shutil.copytree(Path(__file__).parent / 'models', Path(self.code_path) / 'models', dirs_exist_ok=True)
            self.writer = SummaryWriter(self.log_path)

        # AMP
        self.use_amp = bool(self.config.get('amp', False))
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.start_epoch = 1
        self.best_score = -1e9  # 监控 PESQ，越大越好

        if self.resume:
            self._resume_checkpoint()

        # 频谱/ISTFT参数
        self._n_fft = int(self.config['FFT']['n_fft'])
        self._hop = int(self.config['FFT']['hop_length'])
        self._win = int(self.config['FFT']['win_length'])
        self._sr = int(self.config['samplerate'])
        self._len_sec_train = float(self.config['train_dataset']['length_in_seconds'])
        self._len_sec_valid = float(self.config['validation_dataset']['length_in_seconds'])

        # Ablation/Debug
        self.ablation = dict(ablation or {})
        self.shape_fix = bool(self.ablation.get('shape_fix', True))
        self.compute_pesq = bool(self.ablation.get('compute_pesq', True))
        self.pesq_jobs = int(self.ablation.get('pesq_jobs', -1))
        self.debug = bool(debug)

    # ---------- 基础状态 ----------
    def _set_train_mode(self):
        self.model.train()

    def _set_eval_mode(self):
        self.model.eval()

    # ---------- Checkpoint ----------
    def _save_checkpoint(self, epoch, score):
        model_dict = self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict()
        state_dict = {
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'model': model_dict
        }
        save_path = os.path.join(self.checkpoint_path, f'model_{str(epoch).zfill(3)}.tar')
        torch.save(state_dict, save_path)
        _p0(f"[ckpt] saved: {save_path}  (pesq={score:.4f})", self.rank)

        if score > self.best_score:
            self.state_dict_best = state_dict.copy()
            self.best_score = float(score)
            _p0(f"[ckpt] new best at epoch {epoch}: pesq={score:.4f}", self.rank)

    def _resume_checkpoint(self):
        ckpts = sorted(glob(os.path.join(self.checkpoint_path, 'model_*.tar')))
        assert len(ckpts) > 0, f"No checkpoints in {self.checkpoint_path}"
        latest = ckpts[-1]
        checkpoint = torch.load(latest, map_location=self.device)
        self.start_epoch = int(checkpoint['epoch']) + 1
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        if self.world_size > 1:
            self.model.module.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint['model'])

    # ---------- 训练 1 个 epoch ----------
    def _train_epoch(self, epoch):
        if hasattr(self.train_dataloader.dataset, "sample_data_per_epoch"):
            self.train_dataloader.dataset.sample_data_per_epoch()

        total_loss = 0.0
        self.train_bar = tqdm(self.train_dataloader, ncols=110)
        for step, (noisy, clean) in enumerate(self.train_bar, 1):
            noisy = noisy.to(self.device)   # (B,F,T,2)
            clean = clean.to(self.device)   # (B,F,T,2)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                enhanced = self.model(noisy)                 # (B,F,T,2)
                if self.shape_fix:
                    enhanced = _match_spec_shape(enhanced, clean, debug=self.debug, where="train")
                loss = self.loss_func(enhanced, clean)

            if self.debug and self.rank == 0 and step == 1:
                _p0(f"[debug][train] noisy {tuple(noisy.shape)} -> enhanced {tuple(enhanced.shape)}", self.rank)
                _p0(f"[debug][train] clean {tuple(clean.shape)}", self.rank)
                _p0(f"[debug][train] lr={self.optimizer.param_groups[0]['lr']:.6e}", self.rank)

            if self.world_size > 1:
                loss = reduce_value(loss)

            total_loss += float(loss.item())
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.config['scheduler']['update_interval'] == 'step':
                self.scheduler.step()

            self.train_bar.desc = f'   train[{epoch}/{self.epochs + self.start_epoch - 1}][{datetime.now().strftime("%Y-%m-%d-%H:%M")}]'
            self.train_bar.postfix = f'train_loss={total_loss / step:.3f}'

        if self.world_size > 1 and (self.device != torch.device("cpu")):
            torch.cuda.synchronize(self.device)

        if self.rank == 0:
            self.writer.add_scalars('lr', {'lr': self.optimizer.param_groups[0]['lr']}, epoch)
            self.writer.add_scalars('train_loss', {'train_loss': total_loss / step}, epoch)

    # ---------- 验证 ----------
    @torch.inference_mode()
    def _validation_epoch(self, epoch):
        total_loss = 0.0
        pesq_sum = 0.0
        steps = 0

        self.validation_bar = tqdm(self.validation_dataloader, ncols=123)
        for step, (noisy, clean) in enumerate(self.validation_bar, 1):
            steps = step
            noisy = noisy.to(self.device)  # (B,F,T,2)
            clean = clean.to(self.device)  # (B,F,T,2)

            enhanced = self.model(noisy)   # (B,F,T,2)
            if self.shape_fix:
                enhanced = _match_spec_shape(enhanced, clean, debug=self.debug, where="valid")

            loss = self.loss_func(enhanced, clean)
            if self.world_size > 1:
                loss = reduce_value(loss)
            total_loss += float(loss.item())

            if self.compute_pesq:
                # 谱 → 波形（PESQ 要 1D 波形）
                enh_wav = ri_batch_to_wav(
                    enhanced, self._n_fft, self._hop, self._win, self._sr, self._len_sec_valid, self.device
                ).detach().cpu().numpy()
                clean_wav = ri_batch_to_wav(
                    clean, self._n_fft, self._hop, self._win, self._sr, self._len_sec_valid, self.device
                ).detach().cpu().numpy()

                def _pesq_safe(sr, ref, deg):
                    try:
                        return float(pesq(sr, ref.astype(np.float32), deg.astype(np.float32), 'wb'))
                    except Exception:
                        return np.nan

                scores = Parallel(n_jobs=self.pesq_jobs)(
                    delayed(_pesq_safe)(self._sr, clean_wav[i], enh_wav[i]) for i in range(enh_wav.shape[0])
                )
                scores = np.array(scores, dtype=np.float32)
                batch_mean = float(np.nanmean(scores)) if np.isfinite(scores).any() else 0.0

                if self.debug and self.rank == 0 and step == 1:
                    _p0(f"[debug][valid] noisy {tuple(noisy.shape)} clean {tuple(clean.shape)} enhanced {tuple(enhanced.shape)}", self.rank)
                    _p0(f"[debug][valid] pesq_batch_mean={batch_mean:.4f}", self.rank)

                pesq_tensor = torch.tensor(batch_mean, device=self.device, dtype=torch.float32)
                if self.world_size > 1:
                    pesq_tensor = reduce_value(pesq_tensor)
                pesq_sum += float(pesq_tensor.item())

            # 保存样例音频（每个 epoch 前几条）
            if self.rank == 0 and (epoch == 1 or epoch % 10 == 0) and step <= 3:
                noisy_wav = ri_batch_to_wav(
                    noisy, self._n_fft, self._hop, self._win, self._sr, self._len_sec_valid, self.device
                ).detach().cpu().numpy()

                # 若 compute_pesq 关闭，这里也要导出 enh_wav/clean_wav
                if not self.compute_pesq:
                    enh_wav = ri_batch_to_wav(enhanced, self._n_fft, self._hop, self._win, self._sr,
                                              self._len_sec_valid, self.device).detach().cpu().numpy()
                    clean_wav = ri_batch_to_wav(clean, self._n_fft, self._hop, self._win, self._sr,
                                                self._len_sec_valid, self.device).detach().cpu().numpy()

                sf.write(os.path.join(self.sample_path, f'sample_{step}_noisy.wav'),    noisy_wav[0], samplerate=self._sr)
                sf.write(os.path.join(self.sample_path, f'sample_{step}_clean.wav'),    clean_wav[0], samplerate=self._sr)
                sf.write(os.path.join(self.sample_path, f'sample_{step}_enh_epoch{str(epoch).zfill(3)}.wav'),
                         enh_wav[0], samplerate=self._sr)

            # UI
            if self.compute_pesq:
                postfix = f'valid_loss={total_loss / step:.3f}, pesq={pesq_sum / step:.4f}'
            else:
                postfix = f'valid_loss={total_loss / step:.3f}'
            self.validation_bar.desc = f'validate[{epoch}/{self.epochs + self.start_epoch - 1}][{datetime.now().strftime("%Y-%m-%d-%H:%M")}]'
            self.validation_bar.postfix = postfix

        if (self.world_size > 1) and (self.device != torch.device("cpu")):
            torch.cuda.synchronize(self.device)

        if self.rank == 0 and steps > 0:
            scalars = {'val_loss': total_loss / steps}
            if self.compute_pesq:
                scalars['pesq'] = pesq_sum / steps
            self.writer.add_scalars('val', scalars, epoch)

        return (total_loss / max(steps, 1)), ((pesq_sum / max(steps, 1)) if self.compute_pesq else 0.0)

    # ---------- 训练主循环 ----------
    def train(self):
        if self.resume:
            self._resume_checkpoint()

        for epoch in range(self.start_epoch, self.epochs + self.start_epoch):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            self._set_train_mode()
            self._train_epoch(epoch)

            self._set_eval_mode()
            valid_loss, score = self._validation_epoch(epoch)

            if self.config['scheduler']['update_interval'] == 'epoch':
                if self.config['scheduler'].get('use_plateau', False):
                    self.scheduler.step(score)
                else:
                    self.scheduler.step()

            if (self.rank == 0) and (epoch % self.save_checkpoint_interval == 0):
                self._save_checkpoint(epoch, score)

        if self.rank == 0 and hasattr(self, "state_dict_best"):
            torch.save(self.state_dict_best,
                       os.path.join(self.checkpoint_path, f'best_model_{str(self.state_dict_best["epoch"]).zfill(3)}.tar'))
            print(f'------------Training for {self.epochs} epochs is done!------------')


# -------------------------
# CLI
# -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--config', default='configs/cfg_train_gtcrn_spec_light.yaml')
    parser.add_argument('-D', '--device', default='0', help='可用 GPU 索引，如 0 或 0,1')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    args.world_size = len(args.device.split(','))

    config = OmegaConf.load(args.config)

    if args.world_size > 1:
        torch.multiprocessing.spawn(run, args=(config, args,), nprocs=args.world_size, join=True)
    else:
        run(0, config, args)
