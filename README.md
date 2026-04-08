# VAE Continual Learning

CVAE(Conditional VAE) with Continual Learning on CIFAR-10 and MNIST.

Investigates whether **Generative Replay + Supervised Contrastive Loss** can prevent catastrophic forgetting when learning classes sequentially.

---

## Problem

In class-incremental learning, a model trained sequentially on tasks forgets previously learned classes — known as **Catastrophic Forgetting**.

**Proposed method** addresses this with two components:
1. **Generative Replay** — after each task, freeze the model as an "anchor". During the next task, generate images of previous classes from the anchor and mix them into training.
2. **Supervised Contrastive Loss** — pulls same-class latent vectors together and pushes different-class apart, keeping the latent space well-organized across tasks.

**Baseline** — same CVAE architecture trained sequentially with no replay and no contrastive loss.

---

## Model Architecture

```
Encoder: Image → Conv(BN+LeakyReLU) × 4 → flatten → concat(class one-hot) → μ, logvar
Decoder: z + class one-hot → FC → reshape → ConvTranspose(BN+LeakyReLU) × 3 → Sigmoid
```

- Conditional on class label (one-hot) at both encoder and decoder
- `logvar.clamp(-10, 10)` to prevent NaN
- KLD warmup over first N steps to stabilize early training
- Gradient clipping (`max_norm=1.0`)

**Loss:**
```
Total = ReconLoss(MSE) + kld_weight × KLDLoss + lambda_cl × SupConLoss
```

---

## Experiment Scenarios

Each scenario is run with both **Proposed** (replay + contrastive) and **Baseline** (no replay, lambda=0):

| Scenario | Task 1 | Task 2 | ... |
|----------|--------|--------|-----|
| `5-5`    | class 0-4 | class 5-9 | |
| `4-6`    | class 0-3 | class 4-9 | |
| `3-7`    | class 0-2 | class 3-9 | |
| `2-8`    | class 0-1 | class 2-9 | |
| `1-9`    | class 0   | class 1-9 | |
| `2x5`    | class 0-1 | class 2-3 | ... class 8-9 (5 tasks) |

---

## Metrics

| Group | Metric | Description |
|-------|--------|-------------|
| `TestLoss` | `recon`, `kld` | MSE + KLD on full test set (all 10 classes) |
| `ReconQuality` | `SSIM`, `LPIPS`, `FID`, `IS` | x → encode → decode quality |
| `GenQuality` | `FID`, `IS` | random z → decode quality |
| `TaskForgetting` | `task{n}_recon`, `task{n}_ssim` | Per-task metrics tracked across all steps — rising `task1_recon` during task2 training = catastrophic forgetting |
| `PerClass` | `recon_{class}`, `ssim_{class}` | Per-class reconstruction quality |
| `Vis` | `GT_vs_Recon`, `Generated_PerClass` | Visual grids logged to W&B |

---

## Usage

### Step 1. Find optimal hyperparameters (Sweep)

```bash
# Bayesian sweep over lambda_cl × lr (18 combinations)
CUDA_VISIBLE_DEVICES=0 python main_VAE_CIFAR10.py --runtype sweep --tasks 5-5
CUDA_VISIBLE_DEVICES=1 python main_VAE_MNIST.py   --runtype sweep --tasks 5-5
```

### Step 2. Run all scenarios

```bash
# Runs 6 scenarios × {Proposed, Baseline} = 12 runs sequentially
CUDA_VISIBLE_DEVICES=0 python main_VAE_CIFAR10.py --runtype train --scenarios all --lambda_cl 1.0 --lr 1e-3
CUDA_VISIBLE_DEVICES=1 python main_VAE_MNIST.py   --runtype train --scenarios all --lambda_cl 1.0 --lr 5e-4
```

### Single run

```bash
python main_VAE_CIFAR10.py --runtype train --tasks 5-5 --lambda_cl 1.0 --lr 1e-3
python main_VAE_CIFAR10.py --runtype train --tasks joint --lambda_cl 0.0 --no_replay
```

---

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--runtype` | required | `sweep` or `train` |
| `--tasks` | `5-5` | Task split: `joint`, `5-5`, `4-6`, `3-7`, `2-8`, `1-9`, `2x5`, `1x10` |
| `--scenarios` | — | Run all scenarios at once: `all` or `"5-5,2x5"` |
| `--lambda_cl` | `0.1` | Contrastive loss weight (0.0 = Baseline) |
| `--lr` | `1e-3` | Learning rate |
| `--no_replay` | — | Disable generative replay (Baseline mode) |
| `--latent_dim` | `256` | Latent space dimension |
| `--total_steps` | `30000` | Total training steps |
| `--hidden_config` | `xlarge` | Model size: `small`, `large`, `xlarge` |
| `--kld_weight` | `0.0025` | KLD loss weight |
| `--kld_warmup` | `5000` | Steps to ramp KLD weight from 0 to full |
| `--eval_interval` | `1000` | Evaluation frequency |
| `--project` | dataset name | W&B project name |

---

## How to Compare Proposed vs Baseline in W&B

1. Go to W&B project → filter by group (e.g. `Scenario_5-5`)
2. Select both `5-5_Proposed_lam1.0` and `5-5_Baseline_lam0.0`
3. Key charts to compare:

| Chart | What to look for |
|-------|-----------------|
| `TaskForgetting/task1_recon` | Baseline rises after task2 starts; Proposed stays flat |
| `TestLoss/recon` | Lower = better overall memory |
| `ReconQuality/SSIM` | Higher = better reconstruction |
| `GenQuality/FID` | Lower = better generation quality |

---

## Requirements

```bash
pip install torch torchvision wandb torchmetrics[image]
```
