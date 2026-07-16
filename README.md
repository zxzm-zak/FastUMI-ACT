# ACT for XArm6

This branch packages the XArm6 variant of Action Chunking Transformer (ACT) for
vision-based robot imitation learning. It keeps the original XArm6 policy
architecture and training settings: a 7-dimensional joint-and-gripper state,
60 action queries, and a ResNet-18 visual backbone.

`smooth_act` is a separate branch in this repository. It is not merged into
this branch because it has a different 8-dimensional policy head and an
additional GRU action-refinement head.

## Installation

Python 3.9 or later and a PyTorch build suitable for the target CUDA version
are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration

The repository contains no datasets, checkpoints, robot SDKs, robot addresses,
or private filesystem paths. Configure local paths with environment variables:

```bash
export ACT_DATA_DIR=/absolute/path/to/dataset
export ACT_CHECKPOINT_DIR=/absolute/path/to/checkpoints
export ACT_DEVICE=cuda  # optional; defaults to cuda when available
```

The dataset directory is expected to contain ACT-compatible HDF5 episodes, for
example `episode_0.hdf5`. Checkpoints and dataset statistics are written below
`$ACT_CHECKPOINT_DIR/<task>/<timestamp>/`.

## Training

Run one process for a smoke test or single-GPU job:

```bash
ACT_DATA_DIR=/data/xarm6 \
ACT_CHECKPOINT_DIR=/runs/act \
bash train_ddp_script.sh --task unplug_charger --nproc_per_node 1
```

For multi-GPU training, set `--nproc_per_node` to the number of local GPUs.
The model and data contracts on this branch remain 7-dimensional.

## XArm6 rollout

The XArm6 driver is not bundled. Install or obtain the compatible BestMan XArm
SDK separately, then explicitly provide both the SDK path and robot address:

```bash
python rollout_Xarm_joint.py \
  --xarm-sdk-dir /absolute/path/to/RoboticsToolBox \
  --robot-ip 192.168.1.100 \
  --task unplug_charger \
  --output-dir /absolute/path/to/rollouts
```

Use `rollout_Xarm_tcp.py` for TCP control. Rollout commands send physical robot
commands; verify the workspace, emergency-stop system, camera calibration, and
joint limits before execution.

## Repository layout

- `config/config.py`: portable defaults and environment-variable configuration.
- `detr/`: ACT model, backbone, and transformer implementation.
- `model/`: policy wrappers and dataset utilities.
- `train.py` and `train_ddp.py`: single-process and distributed training.
- `rollout_Xarm_joint.py` and `rollout_Xarm_tcp.py`: XArm6 rollout scripts.

## Attribution

This implementation builds on the Action Chunking Transformer (ACT) project.
The `detr/` source files retain their existing third-party copyright notices.
Before publishing, the repository maintainer must select a license that is
compatible with all contributed and third-party code.
