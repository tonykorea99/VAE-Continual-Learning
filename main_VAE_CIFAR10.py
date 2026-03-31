#!/usr/bin/env python3
"""
main_VAE_CIFAR10.py — CVAE Continual Learning on CIFAR-10
==========================================================
Usage:
  # Sweep: find optimal lambda_cl and lr
  python main_VAE_CIFAR10.py --runtype sweep --tasks 5-5
  python main_VAE_CIFAR10.py --runtype sweep --tasks 5-5 --sweep_count 30 --sweep_steps 10000

  # Train: single run with specified hyperparameters
  python main_VAE_CIFAR10.py --runtype train --tasks 5-5 --lambda_cl 0.1 --lr 1e-3
  python main_VAE_CIFAR10.py --runtype train --tasks joint --lambda_cl 0.0 --no_replay

  # Available tasks: joint, 2-8, 5-5, 6-4, 7-3, 9-1, 2x5, 1x10
"""

import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
from tqdm import tqdm
import wandb

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

# ============================================================
# Dataset Constants
# ============================================================
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]
NUM_CLASSES = 10
IN_CHANNELS = 3
IMAGE_SIZE = 32

HIDDEN_CONFIGS = {
    "small":  [32, 64, 128],
    "large":  [64, 128, 256],
    "xlarge": [64, 128, 256, 512],
}


# ============================================================
# 1. Argument Parser  (맨 위 — 관리 편의를 위해)
# ============================================================
def build_parser():
    parser = argparse.ArgumentParser(
        description="CVAE Continual Learning — CIFAR-10",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_VAE_CIFAR10.py --runtype sweep --tasks 5-5
  python main_VAE_CIFAR10.py --runtype train --tasks 5-5 --lambda_cl 0.1 --lr 1e-3
  python main_VAE_CIFAR10.py --runtype train --tasks joint --lambda_cl 0.0 --no_replay
        """)

    # --- Run type ---
    parser.add_argument('--runtype', type=str, required=True,
                        choices=['sweep', 'train'],
                        help='sweep: find optimal lambda/lr | train: single run')

    # --- Task configuration ---
    parser.add_argument('--tasks', type=str, default='5-5',
                        help='Task split (joint/2-8/5-5/6-4/7-3/9-1/2x5/1x10)')
    parser.add_argument('--use_replay', action='store_true', default=True,
                        help='Enable generative replay (default: on)')
    parser.add_argument('--no_replay', dest='use_replay', action='store_false',
                        help='Disable generative replay')

    # --- Hyperparameters ---
    parser.add_argument('--lambda_cl', type=float, default=0.1,
                        help='Contrastive loss weight (default: 0.1)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='Latent dimension (default: 256)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--total_steps', type=int, default=30000,
                        help='Total training steps (default: 30000)')
    parser.add_argument('--kld_weight', type=float, default=0.0025,
                        help='KLD loss weight (default: 0.0025)')
    parser.add_argument('--kld_warmup', type=int, default=5000,
                        help='KLD warmup steps (default: 5000)')
    parser.add_argument('--hidden_config', type=str, default='xlarge',
                        choices=['small', 'large', 'xlarge'],
                        help='Model size (default: xlarge)')

    # --- Logging intervals ---
    parser.add_argument('--log_interval', type=int, default=50,
                        help='Training loss logging interval (default: 50)')
    parser.add_argument('--eval_interval', type=int, default=1000,
                        help='Full evaluation interval (default: 1000)')
    parser.add_argument('--forgetting_interval', type=int, default=0,
                        help='Per-task forgetting eval interval; 0 = same as eval_interval')

    # --- W&B ---
    parser.add_argument('--project', type=str, default='VAE_CL_CIFAR10',
                        help='W&B project name')

    # --- Sweep ---
    parser.add_argument('--sweep_count', type=int, default=18,
                        help='Number of sweep trials (default: 18 = 6λ x 3lr)')
    parser.add_argument('--sweep_steps', type=int, default=5000,
                        help='Steps per sweep trial (default: 5000)')
    parser.add_argument('--sweep_eval_interval', type=int, default=2500,
                        help='Eval interval during sweep (default: 2500)')

    # --- Device ---
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    return parser


# ============================================================
# 2. Model
# ============================================================
class SupConLoss(nn.Module):
    """Supervised Contrastive Loss for latent-space class separation."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        bs = features.shape[0]
        if bs <= 1:
            return torch.tensor(0.0, device=device)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        features = F.normalize(features, dim=1)
        sim = torch.matmul(features, features.T) / self.temperature
        sim_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - sim_max.detach()
        self_mask = 1.0 - torch.eye(bs, device=device)
        mask = mask * self_mask
        exp_logits = torch.exp(logits) * self_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        pos_mean = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        return -pos_mean.mean()


class CVAE(nn.Module):
    """Conditional VAE — Conv encoder/decoder with BatchNorm + LeakyReLU."""
    def __init__(self, latent_dim, num_classes=NUM_CLASSES,
                 in_channels=IN_CHANNELS, image_size=IMAGE_SIZE,
                 hidden_channels=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels or [64, 128, 256]

        # --- Encoder ---
        enc = []
        ch_in = in_channels
        for ch_out in self.hidden_channels:
            enc.append(nn.Sequential(
                nn.Conv2d(ch_in, ch_out, 3, 1, 0),
                nn.BatchNorm2d(ch_out),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            ch_in = ch_out
        self.encoder = nn.Sequential(*enc)

        n_layers = len(self.hidden_channels)
        self.final_size = image_size - 2 * n_layers
        flat_dim = self.hidden_channels[-1] * self.final_size ** 2

        self.fc_mu     = nn.Linear(flat_dim + num_classes, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim + num_classes, latent_dim)
        self.fc_dec    = nn.Linear(latent_dim + num_classes, flat_dim)

        # --- Decoder ---
        dec = []
        rev = self.hidden_channels[::-1]
        for i in range(len(rev) - 1):
            dec.append(nn.Sequential(
                nn.ConvTranspose2d(rev[i], rev[i + 1], 3, 1, 0),
                nn.BatchNorm2d(rev[i + 1]),
                nn.LeakyReLU(0.2, inplace=True),
            ))
        self.decoder_layers = nn.Sequential(*dec)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(rev[-1], in_channels, 3, 1, 0),
            nn.Sigmoid(),
        )

    def encode(self, x, y_oh):
        h = self.encoder(x).view(x.size(0), -1)
        h = torch.cat([h, y_oh], dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z, y_oh):
        h = self.fc_dec(torch.cat([z, y_oh], dim=1))
        h = h.view(-1, self.hidden_channels[-1],
                    self.final_size, self.final_size)
        return self.final_layer(self.decoder_layers(h))

    def forward(self, x, y):
        y_oh = F.one_hot(y, self.num_classes).float().to(x.device)
        mu, logvar = self.encode(x, y_oh)
        logvar = logvar.clamp(-10, 10)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z, y_oh), mu, logvar


# ============================================================
# 3. Data Utilities
# ============================================================
def parse_tasks(task_str, num_classes=NUM_CLASSES):
    """Parse task string -> list of class lists.
    'joint' -> [[0..9]]   '5-5' -> [[0..4],[5..9]]
    '2x5'  -> [[0,1],[2,3],..,[8,9]]   '1x10' -> [[0],[1],..[9]]
    """
    if task_str == "joint":
        return [list(range(num_classes))]
    if "x" in task_str:
        n_per, n_tasks = int(task_str.split("x")[0]), int(task_str.split("x")[1])
        return [list(range(i * n_per, (i + 1) * n_per)) for i in range(n_tasks)]
    if "-" in task_str:
        splits = [int(x) for x in task_str.split("-")]
        tasks, start = [], 0
        for s in splits:
            tasks.append(list(range(start, start + s)))
            start += s
        return tasks
    raise ValueError(f"Unknown task format: {task_str}")


# ============================================================
# 4. Evaluation  (Training Loss / Test Loss / Recon Metrics /
#                  Generation Metrics / Per-Class / Visualizations)
# ============================================================
def evaluate(model, test_loader, metrics, device, args,
             step, task_idx, fixed_x, fixed_y, lightweight=False,
             task_history=None):
    """
    [REQ-1] Test Loss        — All/recon_loss + kld on full test set (all 10 classes)
    [REQ-2] Recon Metrics    — SSIM, LPIPS, FID, IS  (x -> encode -> decode)
    [REQ-3] Gen Metrics      — FID, IS               (random z -> decode)
    [REQ-4] Per-Class        — recon loss + SSIM per class
    [REQ-5] TaskPerf         — per-task group recon/SSIM to track forgetting
                               e.g. TaskPerf/task0_recon during task1 training
                               shows if task0 classes are being forgotten
    [REQ-6] Visualizations   — GT vs Recon grid, per-class generation grid

    lightweight=True (sweep mode): Test Loss + SSIM only — skips FID/IS/LPIPS/Vis
    task_history: list of class lists [[0..4], [5..9], ...] for tasks seen so far
    """
    model.eval()
    full = not lightweight

    # --- Phase 1: Test Loss + Reconstruction Metrics ---
    if full:
        metrics['fid'].reset()
        metrics['is'].reset()

    test_recon, test_kld, n_batch = 0.0, 0.0, 0
    recon_ssim = 0.0
    recon_lpips = 0.0
    pc_recon = defaultdict(list)
    pc_ssim = defaultdict(list)

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            recon, mu, logvar = model(x, y)

            # [REQ-1] Test loss
            test_recon += F.mse_loss(recon, x, reduction='sum').item() / x.size(0)
            test_kld += (-0.5 * torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp())).item() / x.size(0)
            n_batch += 1

            # [REQ-2] Overall recon SSIM
            recon_ssim += metrics['ssim'](recon, x).item()

            if full:
                # LPIPS
                x64 = F.interpolate(x, size=(64, 64))
                r64 = F.interpolate(recon, size=(64, 64))
                recon_lpips += metrics['lpips'](r64, x64).item()

                # Recon FID / IS
                xu8 = (x * 255).byte()
                ru8 = (recon * 255).clamp(0, 255).byte()
                metrics['fid'].update(xu8, real=True)
                metrics['fid'].update(ru8, real=False)
                metrics['is'].update(ru8)

            # [REQ-4] Per-class recon loss & SSIM
            for c in range(NUM_CLASSES):
                m = y == c
                if m.sum() >= 2:
                    pc_recon[c].append(F.mse_loss(recon[m], x[m]).item())
                    pc_ssim[c].append(metrics['ssim'](recon[m], x[m]).item())

    # --- Log dict ---
    log = {
        # All/recon_loss = full test set (all 10 classes) — explicit overall metric
        "All/recon_loss":   test_recon / n_batch,
        "All/kld_loss":     test_kld / n_batch,
        "Recon/SSIM":       recon_ssim / n_batch,
        "Meta/step":        step,
        "Meta/task":        task_idx,
    }

    # [REQ-4] Per-class
    for c in range(NUM_CLASSES):
        name = CLASS_NAMES[c]
        if pc_recon[c]:
            log[f"PerClass_Recon/{name}"] = np.mean(pc_recon[c])
        if pc_ssim[c]:
            log[f"PerClass_SSIM/{name}"] = np.mean(pc_ssim[c])

    # [REQ-5] Per-task group metrics — key forgetting indicator
    # TaskPerf/task0_recon going UP during task1 training = catastrophic forgetting
    if task_history:
        for t_idx, t_classes in enumerate(task_history):
            t_recon = [v for c in t_classes for v in pc_recon[c]]
            t_ssim  = [v for c in t_classes for v in pc_ssim[c]]
            if t_recon:
                log[f"TaskPerf/task{t_idx}_recon"] = np.mean(t_recon)
            if t_ssim:
                log[f"TaskPerf/task{t_idx}_ssim"]  = np.mean(t_ssim)

    if full:
        r_fid = metrics['fid'].compute().item()
        r_is = metrics['is'].compute()[0].item()

        # --- Phase 2: Generation Metrics (random z -> decode) ---
        metrics['fid'].reset()
        metrics['is'].reset()

        with torch.no_grad():
            for x, y in test_loader:
                xu8 = (x.to(device) * 255).byte()
                metrics['fid'].update(xu8, real=True)

            for c in range(NUM_CLASSES):
                z = torch.randn(100, args.latent_dim, device=device)
                yc = torch.full((100,), c, dtype=torch.long, device=device)
                yoh = F.one_hot(yc, NUM_CLASSES).float()
                gen = model.decode(z, yoh)
                gu8 = (gen * 255).clamp(0, 255).byte()
                metrics['fid'].update(gu8, real=False)
                metrics['is'].update(gu8)

        g_fid = metrics['fid'].compute().item()
        g_is = metrics['is'].compute()[0].item()

        # --- Visualizations ---
        with torch.no_grad():
            fx_recon, _, _ = model(fixed_x, fixed_y)
            n = min(32, fixed_x.size(0))
            gt_grid = torchvision.utils.make_grid(fixed_x[:n], nrow=8)
            rc_grid = torchvision.utils.make_grid(fx_recon[:n], nrow=8)
            vis_recon = torch.cat([gt_grid, rc_grid], dim=1)

            gen_imgs = []
            for c in range(NUM_CLASSES):
                z = torch.randn(8, args.latent_dim, device=device)
                yc = torch.full((8,), c, dtype=torch.long, device=device)
                yoh = F.one_hot(yc, NUM_CLASSES).float()
                gen_imgs.append(model.decode(z, yoh))
            vis_gen = torchvision.utils.make_grid(
                torch.cat(gen_imgs), nrow=8)

        log.update({
            "Recon/LPIPS":      recon_lpips / n_batch,
            "Recon/FID":        r_fid,
            "Recon/IS":         r_is,
            "Gen/FID":          g_fid,
            "Gen/IS":           g_is,
            "Vis/GT_vs_Recon": wandb.Image(
                vis_recon, caption=f"Top:GT Bottom:Recon | Step {step}"),
            "Vis/Generated_PerClass": wandb.Image(
                vis_gen, caption=f"Rows=class 0-9, 8 samples each | Step {step}"),
        })

    wandb.log(log)
    model.train()

    if full:
        print(f"  [Step {step}] "
              f"TestRecon: {test_recon / n_batch:.4f} | "
              f"SSIM: {recon_ssim / n_batch:.4f} | "
              f"ReconFID: {r_fid:.1f} | GenFID: {g_fid:.1f}")
    else:
        print(f"  [Step {step}] "
              f"TestRecon: {test_recon / n_batch:.4f} | "
              f"SSIM: {recon_ssim / n_batch:.4f}")

    return test_recon / n_batch


# ============================================================
# 5. Core Training Loop
# ============================================================
def run_experiment(args, lightweight=False):
    """Main training + evaluation loop. Assumes wandb is already initialized.
    lightweight=True: sweep mode — only loads SSIM metric, skips FID/IS/LPIPS.
    """
    device = torch.device(args.device)

    # --- Data ---
    transform = transforms.ToTensor()
    train_full = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    test_full = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_full, batch_size=100, shuffle=False)

    # Fixed samples for consistent visualization
    fixed_x, fixed_y = next(iter(DataLoader(test_full, batch_size=64, shuffle=False)))
    fixed_x, fixed_y = fixed_x.to(device), fixed_y.to(device)

    # --- Metrics (created once, reset each eval) ---
    metrics = {
        'ssim':  StructuralSimilarityIndexMeasure(data_range=1.0).to(device),
    }
    if not lightweight:
        metrics['lpips'] = LearnedPerceptualImagePatchSimilarity(
            net_type='alex', normalize=True).to(device)
        metrics['fid'] = FrechetInceptionDistance(feature=64).to(device)
        metrics['is'] = InceptionScore(feature='logits_unbiased', splits=1).to(device)

    # --- Model ---
    hidden_ch = HIDDEN_CONFIGS[args.hidden_config]
    model = CVAE(args.latent_dim, NUM_CLASSES, IN_CHANNELS,
                 IMAGE_SIZE, hidden_ch).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    con_crit = SupConLoss().to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model params: {n_params:.2f}M | Hidden: {hidden_ch}")

    # --- Tasks ---
    tasks = parse_tasks(args.tasks)
    steps_per_task = args.total_steps // len(tasks)

    anchor_model = None
    seen_classes = []
    global_step = 0
    task_history = []  # grows as tasks are added: [[0..4], [5..9], ...]

    # forgetting_interval: 0 means use same as eval_interval
    forget_interval = args.forgetting_interval or args.eval_interval

    # Initial eval (step 0, no task history yet)
    evaluate(model, test_loader, metrics, device, args,
             0, 0, fixed_x, fixed_y, lightweight=lightweight,
             task_history=[])

    for task_id, task_classes in enumerate(tasks):
        print(f"\n{'=' * 60}")
        print(f"[Task {task_id + 1}/{len(tasks)}] Classes: {task_classes} "
              f"| Steps: {steps_per_task}")
        print(f"{'=' * 60}")

        indices = [i for i, t in enumerate(train_full.targets)
                   if t in task_classes]
        train_loader = DataLoader(
            Subset(train_full, indices),
            batch_size=args.batch_size, shuffle=True, drop_last=True)
        train_iter = iter(train_loader)

        seen_classes = seen_classes + task_classes
        task_history = task_history + [task_classes]  # add before training starts
        model.train()

        pbar = tqdm(total=steps_per_task,
                    desc=f"Task {task_id + 1} cls{task_classes}",
                    unit="step", dynamic_ncols=True)

        for _ in range(steps_per_task):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)

            # Generative Replay
            if args.use_replay and anchor_model is not None:
                bs = x.size(0)
                y_rep = torch.tensor(
                    np.random.choice(seen_classes, size=bs),
                    device=device)
                z_rep = torch.randn(bs, args.latent_dim, device=device)
                with torch.no_grad():
                    x_rep = anchor_model.decode(
                        z_rep, F.one_hot(y_rep, NUM_CLASSES).float())
                x = torch.cat([x, x_rep])
                y = torch.cat([y, y_rep])

            recon, mu, logvar = model(x, y)
            recon_loss = F.mse_loss(recon, x, reduction='sum') / x.size(0)
            kld_loss = -0.5 * torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

            con_loss = torch.tensor(0.0, device=device)
            if args.lambda_cl > 0:
                con_loss = con_crit(mu, y)

            kld_w = args.kld_weight * min(
                1.0, global_step / max(args.kld_warmup, 1))
            total_loss = (recon_loss
                          + kld_w * kld_loss
                          + args.lambda_cl * con_loss)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            global_step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{total_loss.item():.3f}",
                             recon=f"{recon_loss.item():.3f}")

            # [REQ — Training Loss: logged every log_interval steps]
            if global_step % args.log_interval == 0:
                wandb.log({
                    "Train/total_loss":     total_loss.item(),
                    "Train/recon_loss":     recon_loss.item(),
                    "Train/kld_loss":       kld_loss.item(),
                    "Train/con_loss":       con_loss.item(),
                    "Train/kld_weight":     kld_w,
                    "Meta/global_step":     global_step,
                    "Meta/task":            task_id + 1,
                })

            # [REQ — Full Evaluation: test loss + all metrics + per-task forgetting]
            if global_step % args.eval_interval == 0:
                evaluate(model, test_loader, metrics, device, args,
                         global_step, task_id + 1, fixed_x, fixed_y,
                         lightweight=lightweight, task_history=task_history)
                model.train()

        pbar.close()

        # Anchor update for replay
        if args.use_replay:
            anchor_model = copy.deepcopy(model)
            anchor_model.eval()
            print(f"  Anchor updated after Task {task_id + 1}")

    # Final evaluation (always full — even in sweep, see the last trial's quality)
    evaluate(model, test_loader, metrics, device, args,
             global_step, len(tasks), fixed_x, fixed_y,
             lightweight=lightweight, task_history=task_history)

    print(f"\nExperiment finished. Total steps: {global_step}")


# ============================================================
# 6. Sweep
# ============================================================
def run_sweep(args):
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'Test/recon_loss',
            'goal': 'minimize',
        },
        'parameters': {
            'lambda_cl': {
                'values': [0.0, 0.01, 0.05, 0.1, 0.5, 1.0],
            },
            'lr': {
                'values': [1e-3, 5e-4, 1e-4],
            },
        },
    }

    sweep_id = wandb.sweep(sweep_config, project=args.project)

    def sweep_fn():
        with wandb.init():
            trial_args = copy.copy(args)
            trial_args.lambda_cl = wandb.config.lambda_cl
            trial_args.lr = wandb.config.lr
            trial_args.total_steps = args.sweep_steps
            trial_args.eval_interval = args.sweep_eval_interval
            # sweep always uses replay (proposed method)
            trial_args.use_replay = True
            print(f"\n[Sweep Trial] lambda={trial_args.lambda_cl} "
                  f"lr={trial_args.lr}")
            run_experiment(trial_args, lightweight=True)

    wandb.agent(sweep_id, function=sweep_fn, count=args.sweep_count)


# ============================================================
# 7. Main
# ============================================================
def main():
    args = build_parser().parse_args()

    print(f"{'=' * 60}")
    print(f"  CVAE Continual Learning — CIFAR-10")
    print(f"  Run type : {args.runtype}")
    print(f"  Tasks    : {args.tasks} -> {parse_tasks(args.tasks)}")
    print(f"  Device   : {args.device}")
    print(f"{'=' * 60}")

    if args.runtype == 'sweep':
        run_sweep(args)
    elif args.runtype == 'train':
        wandb.init(
            project=args.project,
            name=f"{args.tasks}_lam{args.lambda_cl}_lr{args.lr}",
            config=vars(args),
        )
        run_experiment(args)
        wandb.finish()


if __name__ == "__main__":
    main()
