import argparse
import copy
import os
import pickle
import sys
import time
from datetime import date

import cv2
import h5py
import numpy as np
import torch

from config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG
from model.utils import get_image, make_policy


def parse_args():
    parser = argparse.ArgumentParser(description="Run a Smooth ACT policy on a Flexiv robot.")
    parser.add_argument("--task", default="open_container_200_joint")
    parser.add_argument("--flexiv-sdk-dir", default=os.getenv("ACT_FLEXIV_SDK_DIR"))
    parser.add_argument("--robot-ip", required=True)
    parser.add_argument("--local-ip", required=True)
    parser.add_argument("--frequency", type=int, default=10)
    parser.add_argument("--camera-port", type=int, default=TASK_CONFIG["camera_port"])
    parser.add_argument("--output-dir", default=os.getenv("ACT_ROLLOUT_DIR"))
    parser.add_argument("--n-rollouts", type=int, default=1)
    parser.add_argument("--extra-time", type=int, default=1500)
    parser.add_argument("--query-frequency", type=int)
    parser.add_argument("--home-after", action="store_true")
    args = parser.parse_args()
    if args.frequency <= 0:
        parser.error("--frequency must be positive")
    if args.query_frequency is not None and args.query_frequency <= 0:
        parser.error("--query-frequency must be positive")
    return args


def load_flexiv_driver(sdk_dir):
    if sdk_dir:
        sys.path.insert(0, sdk_dir)
    try:
        import flexivrdk
        from Bestman_flexiv import Bestman_Real_Flexiv
    except ImportError as error:
        raise RuntimeError(
            "Unable to import the Flexiv driver. Set --flexiv-sdk-dir or ACT_FLEXIV_SDK_DIR "
            "to the directory containing Bestman_flexiv.py and flexivrdk."
        ) from error
    return flexivrdk, Bestman_Real_Flexiv


def capture_image(camera):
    success, frame = camera.read()
    if not success:
        raise RuntimeError("Unable to read a frame from the configured camera.")
    return frame


def load_policy(task, device):
    checkpoint_path = os.path.join(
        TRAIN_CONFIG["checkpoint_dir"], task, TRAIN_CONFIG["eval_ckpt_name"]
    )
    stats_path = os.path.join(TRAIN_CONFIG["checkpoint_dir"], task, "dataset_stats.pkl")
    policy = make_policy(POLICY_CONFIG["policy_class"], POLICY_CONFIG)
    policy.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(device)))
    policy.to(device)
    policy.eval()
    with open(stats_path, "rb") as stats_file:
        stats = pickle.load(stats_file)
    return policy, stats


def save_rollout(observations, actions, output_dir, rollout_index):
    if not observations:
        raise RuntimeError("No observations were recorded; refusing to write an empty rollout.")

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{date.today()}_rollout_{rollout_index}.hdf5")
    camera_names = TASK_CONFIG["camera_names"]
    max_timesteps = len(observations)

    with h5py.File(path, "w", rdcc_nbytes=1024**2 * 2) as root:
        root.attrs["sim"] = False
        observation_group = root.create_group("observations")
        image_group = observation_group.create_group("images")
        for camera_name in camera_names:
            frames = [observation["images"][camera_name] for observation in observations]
            image_shape = np.asarray(frames[0]).shape
            dataset = image_group.create_dataset(
                camera_name,
                (max_timesteps, *image_shape),
                dtype="uint8",
                chunks=(1, *image_shape),
            )
            dataset[...] = frames

        observation_group.create_dataset(
            "qpos", data=np.asarray([observation["qpos"] for observation in observations])
        )
        observation_group.create_dataset(
            "qvel", data=np.asarray([observation["qvel"] for observation in observations])
        )
        root.create_dataset("action", data=np.asarray(actions))

    return path


def run_rollout(bestman, policy, stats, device, args):
    camera = cv2.VideoCapture(args.camera_port)
    if not camera.isOpened():
        raise RuntimeError(f"Unable to open camera port {args.camera_port}.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, TASK_CONFIG["cam_width"])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, TASK_CONFIG["cam_height"])

    try:
        for _ in range(100):
            capture_image(camera)

        query_frequency = args.query_frequency or POLICY_CONFIG["num_queries"]
        num_queries = POLICY_CONFIG["num_queries"]
        total_steps = TASK_CONFIG["episode_len"] + args.extra_time
        action_history = None
        if POLICY_CONFIG["temporal_agg"]:
            action_history = torch.zeros(
                [total_steps, total_steps + num_queries, TASK_CONFIG["action_dim"]],
                device=device,
            )

        observations = []
        executed_actions = []
        for step in range(total_steps):
            joint_positions = list(bestman.get_current_joint_angles())
            gripper_position = bestman.get_gripper_position()
            qpos = joint_positions + [gripper_position]
            qvel = list(bestman.get_current_joint_velocities()) + [0.0]
            expected_dimension = TASK_CONFIG["state_dim"]
            if len(qpos) != expected_dimension or len(qvel) != expected_dimension:
                raise RuntimeError(
                    "The Flexiv driver returned an unexpected state dimension: "
                    f"qpos={len(qpos)}, qvel={len(qvel)}, expected={expected_dimension}."
                )

            image = copy.deepcopy(capture_image(camera))
            left = int(image.shape[1] * 0.12)
            right = int(image.shape[1] * 0.88)
            image = image[:, left:right, :]
            observation = {
                "qpos": qpos,
                "qvel": qvel,
                "images": {name: image for name in TASK_CONFIG["camera_names"]},
            }

            qpos_normalized = (np.asarray(qpos) - stats["qpos_mean"]) / stats["qpos_std"]
            qpos_tensor = torch.from_numpy(qpos_normalized).float().to(device).unsqueeze(0)
            image_tensor = get_image(observation["images"], TASK_CONFIG["camera_names"], device)

            if step % query_frequency == 0:
                all_actions = policy(qpos_tensor, image_tensor)

            if POLICY_CONFIG["temporal_agg"]:
                action_history[[step], step : step + num_queries] = all_actions
                populated = action_history[:, step]
                populated = populated[torch.all(populated != 0, dim=1)]
                weights = np.exp(-0.05 * np.arange(len(populated)))
                weights = torch.from_numpy((weights / weights.sum()).astype(np.float32)).to(device)
                raw_action = (populated * weights.unsqueeze(1)).sum(dim=0)
            else:
                raw_action = all_actions[0, step % query_frequency]

            action = raw_action.detach().cpu().numpy()
            action = action * stats["action_std"] + stats["action_mean"]
            gripper_command = int(np.clip((1 - action[-1]) * 400, 0, 255))
            bestman.gripper_goto(gripper_command)
            time.sleep(1 / args.frequency)
            bestman.move_arm_to_joint_angles(action[:-1])

            observations.append(observation)
            executed_actions.append(action)

        return observations, executed_actions
    finally:
        camera.release()


def main():
    args = parse_args()
    device = os.environ["DEVICE"]
    flexivrdk, Bestman_Real_Flexiv = load_flexiv_driver(args.flexiv_sdk_dir)
    log = flexivrdk.Log()
    bestman = Bestman_Real_Flexiv(args.robot_ip, args.local_ip, args.frequency)

    if bestman.robot.isFault():
        log.warn("Fault occurred on the robot server; attempting to clear it.")
        bestman.robot.clearFault()
    if bestman.robot.isFault():
        raise RuntimeError("The Flexiv robot remains in a fault state.")

    bestman.robot.enable()
    while not bestman.robot.isOperational():
        time.sleep(1)
    bestman.connect_gripper()

    policy, stats = load_policy(args.task, device)
    output_dir = args.output_dir or TASK_CONFIG["dataset_dir"]
    for rollout_index in range(args.n_rollouts):
        observations, actions = run_rollout(bestman, policy, stats, device, args)
        output_path = save_rollout(observations, actions, output_dir, rollout_index)
        print(f"Saved rollout to {output_path}")

    if args.home_after:
        bestman.go_home(100)


if __name__ == "__main__":
    main()
