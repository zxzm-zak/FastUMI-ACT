# FastUMI-ACT

FastUMI-ACT publishes two maintained ACT variants as separate branches. They
are intentionally not merged: their state/action contracts, rollout drivers,
and inference behavior differ.

| Branch | Target | Contract |
| --- | --- | --- |
| [`xarm6`](../../tree/xarm6) | XArm6 joint or TCP control | 7-dimensional state and action vectors |
| [`smooth_act`](../../tree/smooth_act) | Smooth ACT for Flexiv | 8-dimensional vectors and GRU action refinement |

Switch to the branch that matches the robot and dataset contract:

```bash
git clone --branch xarm6 https://github.com/zxzm-zak/FastUMI-ACT.git
# or
git clone --branch smooth_act https://github.com/zxzm-zak/FastUMI-ACT.git
```

Each branch includes its own installation, training, and rollout instructions.
Datasets, checkpoints, robot SDKs, robot addresses, and private filesystem
paths are intentionally excluded from this repository.

## License

FastUMI-ACT is released under the [MIT License](LICENSE). See each branch for
third-party attribution and hardware-specific safety notes.
