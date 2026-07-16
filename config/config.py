import os
from pathlib import Path

import torch


os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

USE_DEPTH_ANYTHING = False
USE_LANG_SAM = False
ROBOT_TYPE = os.getenv("ACT_ROBOT_TYPE", "XARM6")
CONTROL_SPACE = os.getenv("ACT_CONTROL_SPACE", "joint").lower()

CONTROL_SPACE_DIMS = {
    "joint": (7, 7),
    "tcp": (8, 8),
}
if CONTROL_SPACE not in CONTROL_SPACE_DIMS:
    supported_modes = ", ".join(sorted(CONTROL_SPACE_DIMS))
    raise ValueError(
        f"Unsupported ACT_CONTROL_SPACE={CONTROL_SPACE!r}. "
        f"Supported values: {supported_modes}."
    )
STATE_DIM, ACTION_DIM = CONTROL_SPACE_DIMS[CONTROL_SPACE]

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = os.getenv("ACT_DATA_DIR", str(REPOSITORY_ROOT / "data" / "example_task"))
CHECKPOINT_DIR = os.getenv("ACT_CHECKPOINT_DIR", str(REPOSITORY_ROOT / "checkpoints"))

device = os.getenv("ACT_DEVICE")
if device is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["DEVICE"] = device

TASK_CONFIG = {
    "dataset_dir": DATA_DIR,
    "episode_len": 180,
    "state_dim": STATE_DIM,
    "action_dim": ACTION_DIM,
    "cam_width": 1920,
    "cam_height": 1080,
    "camera_names": ["front"],
    "camera_port": 0,
}

POLICY_CONFIG = {
    "lr": 1e-5,
    "device": device,
    "num_queries": 60,
    "kl_weight": 100,
    "hidden_dim": 512,
    "dim_feedforward": 3200,
    "lr_backbone": 1e-5,
    "backbone": "resnet18",
    "enc_layers": 4,
    "dec_layers": 7,
    "nheads": 8,
    "camera_names": ["front"],
    "policy_class": "ACT",
    "temporal_agg": False,
    "state_dim": TASK_CONFIG["state_dim"],
    "action_dim": TASK_CONFIG["action_dim"],
}

TRAIN_CONFIG = {
    "seed": 42,
    "num_epochs": 5000,
    "batch_size_val": 1,
    "batch_size_train": 1,
    "grad_accum": 2,
    "eval_ckpt_name": "policy_last.ckpt",
    "checkpoint_dir": CHECKPOINT_DIR,
}
