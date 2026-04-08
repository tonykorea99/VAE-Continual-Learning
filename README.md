# VAE Continual Learning

CVAE(Conditional VAE) with Continual Learning on CIFAR-10 and MNIST.

Investigates whether **Generative Replay + Supervised Contrastive Loss** can prevent catastrophic forgetting when learning classes sequentially.

---

## Problem

In class-incremental learning, a model trained sequentially on tasks forgets previously learned classes — known as **Catastrophic Forgetting**.

**Proposed** addresses this with two components:
1. **Generative Replay** — after each task, freeze the model as an "anchor". During the next task, generate images of previous classes from the anchor and mix them into training.
2. **Supervised Contrastive Loss** — pulls same-class latent vectors together and pushes different-class apart, keeping the latent space organized across tasks.

**Baseline** — same CVAE architecture, no replay, no contrastive loss (`--no_replay --lambda_cl 0.0`).

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

Each scenario runs **Proposed → Baseline** automatically via `--scenarios`.

| Scenario | Task 1 | Task 2 | Task 3+ |
|----------|--------|--------|---------|
| `5-5` | class 0-4 | class 5-9 | — |
| `4-6` | class 0-3 | class 4-9 | — |
| `3-7` | class 0-2 | class 3-9 | — |
| `2-8` | class 0-1 | class 2-9 | — |
| `1-9` | class 0 | class 1-9 | — |
| `2x5` | class 0-1 | class 2-3 | class 4-5, 6-7, 8-9 (5 tasks total) |

---

## Metrics

All metrics are computed **only on seen classes** (classes the model has been trained on so far). Unseen classes are never included.

| W&B Group | Key | Description |
|-----------|-----|-------------|
| `SeenLoss` | `recon`, `kld` | Overall loss on all seen classes — mirrors Train loss on test set |
| `SeenRecon` | `SSIM`, `LPIPS`, `FID`, `IS` | Perceptual quality of x → encode → decode (seen classes only) |
| `CurrentTask` | `recon`, `ssim` | Performance on the task being trained **right now** — should decrease |
| `PastTasks` | `recon`, `ssim` | Performance on all **previous** task classes — **key forgetting indicator** |
| `TaskForgetting` | `task{n}_recon`, `task{n}_ssim` | Per-task breakdown (1-indexed, matches `Meta/task`) |
| `GenQuality` | `FID`, `IS` | Random z → decode quality (seen classes only) |
| `PerClass` | `recon_{class}`, `ssim_{class}` | Per-class reconstruction quality (seen classes only) |
| `Train` | `total_loss`, `recon_loss`, `kld_loss`, `con_loss` | Training loss logged every `--log_interval` steps |
| `Vis` | `GT_vs_Recon`, `Generated_SeenClasses` | Visual grids — seen classes only, caption shows which classes |

---

## How to Compare Proposed vs Baseline in W&B

1. W&B project → select group (e.g. `Scenario_5-5`)
2. Check both `5-5_Proposed_lam1.0` and `5-5_Baseline_lam0.0`

| Chart | What to look for |
|-------|-----------------|
| `PastTasks/recon` | **Core forgetting indicator** — Baseline rises when task2 starts; Proposed stays flat |
| `CurrentTask/recon` | Should decrease as the model learns the current task |
| `TaskForgetting/task1_recon` | Task 1 classes tracked throughout — compare Proposed vs Baseline |
| `SeenLoss/recon` | Overall performance on everything seen so far |
| `SeenRecon/SSIM` | Higher = better reconstruction quality |
| `GenQuality/FID` | Lower = better generation quality |

---

## Usage

### Step 1. Find optimal hyperparameters (Sweep)

```bash
CUDA_VISIBLE_DEVICES=0 python main_VAE_CIFAR10.py --runtype sweep --tasks 5-5
CUDA_VISIBLE_DEVICES=1 python main_VAE_MNIST.py   --runtype sweep --tasks 5-5
```

Bayesian sweep over `lambda_cl` × `lr` (18 combinations). Optimizes `SeenLoss/recon`.

### Step 2. Run all scenarios (Proposed + Baseline)

```bash
# 6 scenarios × {Proposed, Baseline} = 12 runs per dataset
CUDA_VISIBLE_DEVICES=0 python main_VAE_CIFAR10.py --runtype train --scenarios all --lambda_cl 1.0 --lr 1e-3
CUDA_VISIBLE_DEVICES=1 python main_VAE_MNIST.py   --runtype train --scenarios all --lambda_cl 1.0 --lr 5e-4
```

### Run a subset of scenarios

```bash
CUDA_VISIBLE_DEVICES=0 python main_VAE_CIFAR10.py --runtype train --scenarios 5-5,2x5 --lambda_cl 1.0 --lr 1e-3
```

### Single run

```bash
# Proposed
python main_VAE_CIFAR10.py --runtype train --tasks 5-5 --lambda_cl 1.0 --lr 1e-3

# Baseline
python main_VAE_CIFAR10.py --runtype train --tasks 5-5 --lambda_cl 0.0 --lr 1e-3 --no_replay
```

---

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--runtype` | required | `sweep` or `train` |
| `--tasks` | `5-5` | Task split: `joint`, `5-5`, `4-6`, `3-7`, `2-8`, `1-9`, `2x5`, `1x10` |
| `--scenarios` | — | Batch run: `all` or comma-separated e.g. `"5-5,2x5"` — runs Proposed+Baseline for each |
| `--lambda_cl` | `0.1` | Contrastive loss weight (`0.0` = Baseline) |
| `--lr` | `1e-3` | Learning rate |
| `--no_replay` | — | Disable generative replay (Baseline mode) |
| `--latent_dim` | `256` | Latent space dimension |
| `--total_steps` | `30000` | Total training steps |
| `--hidden_config` | `xlarge` | Model size: `small`, `large`, `xlarge` |
| `--kld_weight` | `0.0025` | KLD loss weight |
| `--kld_warmup` | `5000` | Steps to ramp KLD weight from 0 to full |
| `--eval_interval` | `1000` | Evaluation interval (steps) |
| `--log_interval` | `50` | Training loss logging interval (steps) |
| `--project` | dataset name | W&B project name |
| `--group` | auto | W&B group name (auto: `Scenario_{tasks}`) |
| `--device` | auto | `cuda` or `cpu` |

---

## Requirements

```bash
pip install torch torchvision wandb torchmetrics[image]
```
