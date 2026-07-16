# ACT for XArm6

This branch packages the XArm6 variant of Action Chunking Transformer (ACT) for
vision-based robot imitation learning. Joint-space training uses a
7-dimensional joint-and-gripper state, 60 action queries, and a ResNet-18
visual backbone.

`smooth_act` is a separate branch in this repository. It is not merged into
this branch because it has a different 8-dimensional policy head and an
additional GRU action-refinement head.

## Installation

Use a Python version supported by both the selected PyTorch build and the
BestMan environment. The official BestMan environment currently uses Python
3.8.

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
The default model and data contracts on this branch remain 7-dimensional.

For TCP pose policies, the state and action vectors are 8-dimensional
(position, quaternion, and gripper). Set `ACT_CONTROL_SPACE=tcp` for both
training and rollout, and use a dataset and checkpoint trained with that
setting:

```bash
ACT_CONTROL_SPACE=tcp \
ACT_DATA_DIR=/data/xarm6_tcp \
ACT_CHECKPOINT_DIR=/runs/act_tcp \
bash train_ddp_script.sh --task pick_bear_tcp --nproc_per_node 1
```

## XArm6 rollout

The XArm6 driver is not bundled. This branch supports the official
[UMI-Robotics/BestMan_Xarm](https://github.com/UMI-Robotics/BestMan_Xarm)
checkout. Create its environment as documented upstream, then provide the
checkout root and robot address explicitly:

```bash
git clone https://github.com/UMI-Robotics/BestMan_Xarm.git
conda env create -f BestMan_Xarm/Install/basic_environment.yaml
conda activate Xarm
pip install -r requirements.txt

python rollout_Xarm_joint.py \
  --bestman-root /absolute/path/to/BestMan_Xarm \
  --robot-ip <ROBOT_IP> \
  --task unplug_charger \
  --output-dir /absolute/path/to/rollouts
```

Alternatively set `ACT_BESTMAN_XARM_ROOT` once. The legacy
`--xarm-sdk-dir` option remains available for a directory containing
`Bestman_real_xarm6.py` directly. Use TCP control only with its 8-dimensional
policy contract:

```bash
ACT_CONTROL_SPACE=tcp python rollout_Xarm_tcp.py \
  --bestman-root /absolute/path/to/BestMan_Xarm \
  --robot-ip <ROBOT_IP> \
  --task pick_bear_tcp \
  --output-dir /absolute/path/to/rollouts
```

Rollout commands send physical robot commands; verify the workspace,
emergency-stop system, camera calibration, and joint limits before execution.

## Repository layout

- `config/config.py`: portable defaults and environment-variable configuration.
- `detr/`: ACT model, backbone, and transformer implementation.
- `model/`: policy wrappers and dataset utilities.
- `train.py` and `train_ddp.py`: single-process and distributed training.
- `rollout_Xarm_joint.py` and `rollout_Xarm_tcp.py`: XArm6 rollout scripts.

## Attribution

This implementation builds on the Action Chunking Transformer (ACT) project.
The `detr/` source files retain their existing third-party copyright notices.
The XArm driver is an external dependency; BestMan_Xarm is not vendored here
and remains subject to its own MIT license. This repository is released under
the MIT license in `LICENSE`.
