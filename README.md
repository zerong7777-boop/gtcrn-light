# GTCRN-Light v3

> Lightweight, deployment-friendly reimplementation of GTCRN for single-channel speech enhancement.

（中文简介）GTCRN-Light v3（下文简称 v3）是在官方 GTCRN 代码基础上做的等价轻量化版本：只改算子、不改语义，保持输入输出接口完全兼容，方便直接替换到现有 SEtrain / TRT-SE 等流水线中。

## 1. Introduction

GTCRN-Light v3 is a structurally equivalent, heavily compressed version of the original Grouped Temporal Convolutional Recurrent Network (GTCRN) for speech enhancement.  
Instead of changing the task setting or loss, v3 keeps:

- complex ratio mask (CRM) prediction in the ERB domain,
- the encoder → DPGRNN → decoder pipeline,
- the overall STFT → ERB → enhancement → iERB → iSTFT data flow,

and only replaces heavy building blocks with lighter operators.

Design goals:

- Keep the same input / output interface as the official GTCRN implementation.
- Reduce parameters and MACs while preserving DPGRNN’s long-term time–frequency modelling.
- Maintain strictly aligned frequency shapes (e.g. 129 → 65 → 33) for stable deployment on edge devices.

### 1.1 Complexity & Example Metrics

Complexity for a 1 s, 16 kHz input (n_fft=512, hop=256, win=512):

| Model              | Params (M) | MACs (G/s) | Note                       |
| ------------------ | ---------: | ---------: | -------------------------- |
| GTCRN (official)   |     ≈0.048 |     ≈0.033 | From the original repo     |
| **GTCRN-Light v3** |  **0.008** |  **0.017** | Spec core, this repository |

Example evaluation (one internal experiment, VCTK-DEMAND style setting):

- PESQ (wb): 2.5966  
- STOI: 0.9338  
- ESTOI: 0.8299  
- SI-SNR: 17.89 dB  

Numbers above are for reference only; your results will depend on the dataset and training recipe.

## 2. Features

- **Operator-level lightweighting**  
  Depthwise separable convolutions, bottlenecked RNNs and lightweight temporal gating drastically reduce parameters and MACs.

- **Semantic equivalence to GTCRN**  
  Preserves ERB subband processing, CRM prediction and dual-path GRNN ordering; no shortcut in modelling capability.

- **Shape-stable encoder / decoder**  
  Integer frequency down/up sampling (129 → 65 → 33 and back) avoids odd–even drift and keeps skip connections aligned.

- **Deployment friendly**  
  Fewer stateful components, no exotic ops, and an ERB projection implemented as a fixed buffer, which makes exporting / quantising easier.

For more design details and training notes (Chinese), please refer to `轻量化文档.md`.

## 3. Environment & Installation

Tested with:

- Python 3.10
- PyTorch 2.0.1 + CUDA 11.8
- Linux x86_64

Minimal dependencies:

```bash
conda create -n gtcrn_light python=3.10
conda activate gtcrn_light

pip install torch torchaudio librosa numpy scipy soundfile tqdm pyyaml
# Optional: for evaluation
pip install pesq pystoi
```

You can of course adapt the environment to your own infrastructure.

## 4. Repository Layout

A minimal layout is expected as:

```text
.
├── configs/
│   └── cfg_train_gtcrn_spec.yaml
├── models/
│   └── gtrcn_light_v3.py
├── tools/
│   ├── batch_infer_gtcrn_spec_light.py
│   ├── make_pairs_from_outputs.py
│   └── print_model_stats.py
├── train_my_spec_light_v3.py
├── 轻量化文档.md
└── README.md
```

Your actual project may contain more scripts (data preparation, logging, etc.); adjust accordingly.

## 5. Training

The main training script for the spectrogram-domain model is:

```bash
python train_my_spec_light_v3.py -C configs/cfg_train_gtcrn_spec.yaml -D 0
```

Typical configuration items in `cfg_train_gtcrn_spec.yaml` include:

- Dataset paths for noisy / clean speech
- STFT parameters: `sr=16000`, `n_fft=512`, `hop=256`, `win=512`
- Optimiser and LR schedule
- GTCRN-Light specific knobs: `width_mult`, bottleneck ratio `r`, `use_two_dpgrnn`, `rnn_bidirectional`

See the comments inside your config file for details.

### 5.1 Training Tips (from practice)

A few practical suggestions when training v3:

- Use combined losses on real/imaginary parts, magnitude, and optionally phase or waveform-based criteria (e.g. SI-SDR).
- Warm up the learning rate for a few epochs and apply gradient clipping (e.g. 5.0) to stabilise DPGRNN.
- SpecAug-style light masking in time / frequency often improves robustness.
- If you already have a strong GTCRN model, you can distil intermediate features from it to accelerate convergence.

More detailed recipes are summarised in `轻量化文档.md`.

## 6. Inference

### 6.1 Batch inference (spectrogram version)

```bash
python tools/batch_infer_gtcrn_spec_light.py   --ckpt /path/to/your/best_model_xxx.tar   --in_dir  ../data/test/noisy   --out_dir ../SEtrain/outs/gtcrn_spec_test_light3   --sr 16000 --n_fft 512 --hop 256 --win 512 --device cuda
```

This script reads all noisy `.wav` files from `--in_dir`, applies the model and writes enhanced files to `--out_dir`.

### 6.2 Build paired list for evaluation

```bash
python tools/make_pairs_from_outputs.py   --enh_dir  ../SEtrain/outs/gtcrn_spec_test_light3   --clean_dir ../data/test/clean   --out pairs_gtcrn_light_v3.txt
```

The generated text file lists `(enhanced, clean)` pairs and can be used by your metric script.

### 6.3 Model complexity

Before running the complexity script, temporarily ensure that the class name in `models/gtrcn_light_v3.py` matches what `print_model_stats.py` expects (for example, rename `GTCRN_light_v3` to `GTCRN_light`).

```bash
python tools/print_model_stats.py   --mode spec   --arch gtrcn_light   --model-path models/gtrcn_light_v3.py   --width-mult 1.0   --two-rnn   --rnn-bidir
```

The script reports parameter count and MACs for a 1-second input:

```text
=== GTCRN-Light v3 (Spec Core) ===
Config  : width_mult=1.0, use_two_dpgrnn=True, rnn_bidirectional=True
Input(1s): F=257, T=63  (n_fft=512, hop=256, win=512)
Para. (M):  0.008
MACs (G/s): 0.017  # core network, without STFT/ISTFT
```

## 7. Design Highlights

Very briefly, GTCRN-Light v3 introduces:

- **GT-ConvLite**  
  Depthwise separable temporal convolutions followed by pointwise projection and lightweight gating, providing a GT-Conv-like receptive field with far fewer parameters.

- **TRALite**  
  RNN-free temporal energy gating implemented with 1D depthwise + pointwise convolutions on frame-wise energy trajectories.

- **Bottlenecked DPGRNN**  
  Intra / inter RNN paths operate in a low-rank space `C → r → C`, greatly reducing parameters while preserving dual-path modelling.

- **Fixed ERB projection**  
  ERB mapping is implemented as a fixed buffer instead of learnable Linear layers, cutting parameters and easing deployment.

See `轻量化文档.md` for diagrams and a more detailed discussion.

## 8. Acknowledgements

This repository is a lightweight, engineering-oriented reimplementation built on top of the official GTCRN project:

- Xiaobin Rong *et al.*, “GTCRN: A Speech Enhancement Model Requiring Ultralow Computational Resources,” ICASSP 2024.  
- Official code repository: https://github.com/Xiaobin-Rong/gtcrn

If you use GTCRN-Light v3 in academic work, please consider citing both the original GTCRN paper and this repository.

## 9. License

Choose a license that matches your plan (for example, MIT, Apache-2.0, or GPL-3.0), update this section, and add a `LICENSE` file accordingly.
