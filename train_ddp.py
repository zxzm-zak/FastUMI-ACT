# Distributed ACT training with AMP.
# ---------------------------------------------------------------
from config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG , USE_DEPTH_ANYTHING# must import first
import os, argparse, datetime, pickle, json
from copy import deepcopy

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast
# from torch.amp import GradScaler
from torch.utils.data import DataLoader, distributed

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

from model.utils import *

# ---------------------------------------------------------------
# Distributed initialization.
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

torch.backends.cudnn.benchmark = True   # Enable cuDNN convolution autotuning.
try:
    from torch.amp import GradScaler                          # ≥ 2.1
    scaler = GradScaler()
except ImportError:
    from torch.cuda.amp import GradScaler                     # ≤ 2.0
    scaler = GradScaler()
# ---------------------------------------------------------------

# Parse the task name.
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="duel_gripper_test")
args = parser.parse_args()
task = args.task

# Paths and configuration.
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
task_cfg, train_cfg, policy_cfg = TASK_CONFIG, TRAIN_CONFIG, POLICY_CONFIG
checkpoint_dir = os.path.join(train_cfg["checkpoint_dir"], task, timestamp)
if dist.get_rank() == 0:
    os.makedirs(checkpoint_dir, exist_ok=True)
dist.barrier()  # Wait until rank zero has created the output directory.

# ---------------------------------------------------------------
def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data = image_data.to(device, non_blocking=True).to(torch.float16).div_(255.)
    qpos_data  = qpos_data.to(device,  non_blocking=True).to(torch.float32)
    action_data = action_data.to(device, non_blocking=True).to(torch.float32)
    is_pad = is_pad.to(device, non_blocking=True)

    with autocast(dtype=torch.bfloat16):
        return policy(qpos_data, image_data, action_data, is_pad)


def reduce_metrics(metrics):
    """Average detached scalar metrics across all distributed ranks."""
    reduced = {}
    for name, value in metrics.items():
        value = value.detach().clone()
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        reduced[name] = value / dist.get_world_size()
    return reduced


def plot_history(train_hist, val_hist, num_epochs, ckpt_dir, seed):
    for key in train_hist[0]:
        plt.figure()
        train_vals = [d[key].item() for d in train_hist]
        val_vals   = [d[key].item() for d in val_hist]
        plt.plot(np.linspace(0, num_epochs - 1, len(train_hist)), train_vals, label="train")
        plt.plot(np.linspace(0, num_epochs - 1, len(val_hist)),   val_vals,   label="val")
        plt.tight_layout(); plt.legend(); plt.title(key)
        plt.savefig(os.path.join(ckpt_dir, f"train_val_{key}_seed_{seed}.png"))
        plt.close()

def train_bc(train_loader, val_loader):
    # policy & optimizer
    policy = make_policy(policy_cfg["policy_class"], policy_cfg).to(device)
    optimizer = make_optimizer(policy_cfg["policy_class"], policy)
    policy = DDP(
        policy,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True
        )

    train_hist, val_hist = [], []
    best_val = float("inf"); best_state = None

    epoch_iter = tqdm(range(train_cfg["num_epochs"]),
                      disable=(dist.get_rank() != 0),
                      dynamic_ncols=True,
                      desc="Epoch")
    for epoch in epoch_iter:
        if dist.get_rank() == 0:
            tqdm.write(f"=== Epoch {epoch} ===")
        # Validation.
        policy.eval(); epoch_dicts = []
        with torch.inference_mode(), autocast(dtype=torch.bfloat16):
            for data in val_loader:
                epoch_dicts.append(forward_pass(data, policy))
        val_summary = reduce_metrics(compute_dict_mean(epoch_dicts))
        val_hist.append(val_summary); val_loss = val_summary["loss"].item()

        # Record validation metrics on rank zero.
        if dist.get_rank() == 0:
            val_log = {f"val_{k}": v.item() for k, v in val_summary.items()}
            val_log["epoch"] = epoch
            wandb.log(val_log)
            if val_loss < best_val:
                best_val, best_state = val_loss, deepcopy(policy.module.state_dict())

        # Training.
        policy.train(); epoch_batch_dicts = []
        train_loader.sampler.set_epoch(epoch)
        for data in train_loader:
            optimizer.zero_grad(set_to_none=True)
            forward_dict = forward_pass(data, policy)
            loss = forward_dict["loss"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_batch_dicts.append(detach_dict(forward_dict))

        train_summary = reduce_metrics(compute_dict_mean(epoch_batch_dicts))
        train_hist.append(train_summary)

        if dist.get_rank() == 0:
            # Update the progress display.
            epoch_iter.set_postfix({
                "train_loss": f"{train_summary['loss'].item():.4f}",
                "val_loss":   f"{val_summary['loss'].item():.4f}"
            })

            # Log training metrics.
            train_log = {f"train_{k}": v.item() for k, v in train_summary.items()}
            train_log["epoch"] = epoch
            wandb.log(train_log)

            # Save an intermediate checkpoint and metrics plot every 200 epochs.
            if epoch % 200 == 0 and epoch > 0:
                torch.save(policy.module.state_dict(),
                           os.path.join(checkpoint_dir, f"policy_epoch_{epoch}_seed_{train_cfg['seed']}.ckpt"))
                plot_history(train_hist, val_hist, epoch, checkpoint_dir, train_cfg["seed"])

    # Save final and best checkpoints on rank zero.
    if dist.get_rank() == 0:
        torch.save(policy.module.state_dict(), os.path.join(checkpoint_dir, "policy_last.ckpt"))
        if best_state is not None:
            torch.save(best_state, os.path.join(checkpoint_dir, "policy_best.ckpt"))
        plot_history(train_hist, val_hist, train_cfg["num_epochs"] - 1, checkpoint_dir, train_cfg["seed"])

# ---------------------------------------------------------------
if __name__ == "__main__":
    set_seed(train_cfg["seed"])

    # Dataset.
    data_dir   = task_cfg["dataset_dir"]
    num_epi    = len(os.listdir(data_dir))
    train_dl, val_dl, stats, _ = load_data(
        data_dir, num_epi, task_cfg["camera_names"],
        train_cfg["batch_size_train"], train_cfg["batch_size_val"]
    )

    # Use a distributed sampler.
    train_dl = DataLoader(
        train_dl.dataset, batch_size=train_cfg["batch_size_train"],
        sampler=distributed.DistributedSampler(train_dl.dataset,
                                               num_replicas=dist.get_world_size(),
                                               rank=dist.get_rank(), shuffle=True),
        num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    val_dl = DataLoader(
        val_dl.dataset, batch_size=train_cfg["batch_size_val"],
        sampler=distributed.DistributedSampler(val_dl.dataset,
                                               num_replicas=dist.get_world_size(),
                                               rank=dist.get_rank(), shuffle=False),
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    # Save statistics and initialize Weights & Biases on rank zero.
    if dist.get_rank() == 0:
        with open(os.path.join(checkpoint_dir, "dataset_stats.pkl"), "wb") as f:
            pickle.dump(stats, f)
        wandb.init(project="ACT_training", name=task, config=policy_cfg, mode="offline")

    train_bc(train_dl, val_dl)

    if dist.get_rank() == 0:
        wandb.finish()

    dist.destroy_process_group()
