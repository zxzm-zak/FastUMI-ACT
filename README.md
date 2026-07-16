# Smooth ACT for Flexiv

This branch contains the Smooth ACT variant of Action Chunking Transformer
(ACT). It extends the policy head with a GRU action-refinement stage and uses
an 8-dimensional state and action contract: seven arm joints plus one gripper
value. It is kept separate from the 7-dimensional XArm6 implementation on the
`xarm6` branch.

## Installation

Use Python 3.9 or later and install the PyTorch build appropriate for the
target CUDA version before installing the remaining dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration

No datasets, checkpoints, robot SDKs, robot addresses, or private filesystem
paths are included. Configure local resources explicitly:

```bash
export ACT_DATA_DIR=/absolute/path/to/dataset
export ACT_CHECKPOINT_DIR=/absolute/path/to/checkpoints
export ACT_DEVICE=cuda  # optional; defaults to cuda when available
```

The dataset directory must contain ACT-compatible HDF5 episodes, for example
`episode_0.hdf5`. The policy expects 8-dimensional state and action tensors.

## Training

```bash
ACT_DATA_DIR=/data/flexiv \
ACT_CHECKPOINT_DIR=/runs/smooth-act \
bash train_ddp_script.sh --task open_container --nproc_per_node 1
```

Set `--nproc_per_node` to the number of local GPUs for a multi-GPU job. The
Smooth ACT loss is the sum of the base ACT L1 loss, the GRU-refined L1 loss,
and the KL term.

## Flexiv rollout

The Flexiv driver is not bundled. Obtain the compatible BestMan Flexiv SDK,
then supply all deployment-specific values explicitly:

```bash
python rollout_Flexiv_joint.py \
  --flexiv-sdk-dir /absolute/path/to/RoboticsToolBox \
  --robot-ip 192.168.1.100 \
  --local-ip 192.168.1.101 \
  --task open_container \
  --output-dir /absolute/path/to/rollouts
```

This command sends physical robot commands. Verify the workspace, emergency
stop, camera calibration, joint limits, and the correct model checkpoint before
running it.

## Repository layout

- `config/config.py`: portable defaults and environment-variable configuration.
- `detr/`: ACT model, backbone, and transformer implementation.
- `model/`: policy wrappers and dataset utilities.
- `train.py` and `train_ddp.py`: single-process and distributed training.
- `rollout_Flexiv_joint.py`: Flexiv joint-control rollout.

## Attribution

This implementation builds on the Action Chunking Transformer (ACT) project.
The `detr/` source files retain their existing third-party copyright notices.
Before publishing, the repository maintainer must select a license that is
compatible with all contributed and third-party code.
