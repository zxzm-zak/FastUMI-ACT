from config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG
import copy
import os
import sys
import cv2
import numpy as np
import torch
import pickle
import argparse
import time
from datetime import date
import datetime
from model.utils import *
from scipy.spatial.transform import Rotation as R

cfg = TASK_CONFIG
policy_config = POLICY_CONFIG
train_cfg = TRAIN_CONFIG
device = os.environ['DEVICE']


def parse_args():
    parser = argparse.ArgumentParser(description="Run a TCP ACT policy on an XArm6 robot.")
    parser.add_argument("--task", default="pick_bear_200_tcp")
    parser.add_argument("--xarm-sdk-dir", default=os.getenv("ACT_XARM_SDK_DIR"))
    parser.add_argument("--robot-ip", required=True)
    parser.add_argument("--output-dir", default=os.getenv("ACT_ROLLOUT_DIR"))
    return parser.parse_args()


def load_xarm_driver(sdk_dir):
    if sdk_dir:
        sys.path.insert(0, sdk_dir)
    try:
        from Bestman_real_xarm6 import Bestman_Real_Xarm6
    except ImportError as error:
        raise RuntimeError(
            "Unable to import the XArm6 driver. Set --xarm-sdk-dir or ACT_XARM_SDK_DIR "
            "to the directory containing Bestman_real_xarm6.py."
        ) from error
    return Bestman_Real_Xarm6


def calculate_new_pose(x, y, z, quaternion, distance):
    """Translate a quaternion pose by ``distance`` along its negative z axis."""
    rotation = R.from_quat(quaternion)
    rotation_matrix = rotation.as_matrix()
    z_axis = rotation_matrix[:, 2]
    new_position = np.array([x, y, z]) - distance * z_axis
    return new_position[0], new_position[1], new_position[2], quaternion


def capture_image(cam):
    success, frame = cam.read()
    if not success:
        raise RuntimeError("Unable to read a frame from the configured camera.")
    return frame

if __name__ == "__main__":
    args = parse_args()
    task = args.task
    Bestman_Real_Xarm6 = load_xarm_driver(args.xarm_sdk_dir)
    bestman = Bestman_Real_Xarm6(args.robot_ip, None, None)
    # bestman.go_home(100) # parameter is distance

    # load the policy
    ckpt_path = os.path.join(train_cfg['checkpoint_dir'], task, train_cfg['eval_ckpt_name'])

    print(f'Loaded: {ckpt_path}')
    policy = make_policy(policy_config['policy_class'], policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)))
    print(loading_status)
    policy.to(device)
    policy.eval()
    stats_path = os.path.join(train_cfg['checkpoint_dir'], task, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # turn on/off temporal_agg
    if policy_config['temporal_agg']:
        query_frequency = policy_config['num_queries'] #  1 to 20
        num_queries = policy_config['num_queries']
    else:
        query_frequency = policy_config['num_queries'] # (1 to 20) k=call policy at each k step
        num_queries = policy_config['num_queries'] # chunk size



    n_rollouts = 1 # 10 trails
    extra_time = 1500 # total time = episode_len + extra_time
    for i in range(n_rollouts):

        if policy_config['temporal_agg']:
            all_time_actions = torch.zeros([cfg['episode_len']+extra_time, cfg['episode_len']+num_queries+extra_time, cfg['state_dim']]).to(device)
        qpos_history = torch.zeros((1, cfg['episode_len']+extra_time, cfg['state_dim'])).to(device)

        with torch.inference_mode():
            # init buffers
            obs_replay = []
            action_replay = []
            cam = cv2.VideoCapture(cfg['camera_port'])
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            # Warm-up frames
            for _ in range(100):
                _ = capture_image(cam)
            print("Warm-up done.")

            print('#'*50)
            print("READY TP START!")
            print('#'*50)

            for t in range(cfg['episode_len'] + extra_time): # TODO remove redunt actions
                t0 = datetime.datetime.now()

                # get real time obs
                _, qpos = bestman.robot.get_position()

                x, y, z, roll, pitch, yaw = qpos
                x, y, z = x/1000, y/1000, z/1000,
                r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
                qx, qy, qz, qw = r.as_quat()
                qpos = [x, y, z, qx, qy, qz, qw]
                q = [qx, qy, qz, qw]
                # print('old qpos:',qpos)


                a1 = bestman.get_gripper_position()
                gripper_open_width = bestman.get_gripper_position() / 850
                distance = - 0.09 + 0.015 * (a1 / 255.0)
                x, y, z, _ = calculate_new_pose(x, y, z, q, distance)
                qpos = [x, y, z, qx, qy, qz, qw]
                # print('new qpos:',qpos)


                # exit()
                qpos = qpos + [gripper_open_width]
                # print('new qpos2:',qpos)
                qvel = bestman.get_current_joint_velocities()

                # try:
                image_gopro = capture_image(cam)
                image1 = copy.deepcopy(image_gopro)
                left = int(1920 * 0.12)
                right = int(1920 * 0.88)
                image1 = image1[:, left:right,:]
                # print(image1.shape)
                # exit()
                cv2.imshow('RealSense', image1)
                cv2.waitKey(1)
                cv2.imwrite("cam.png", image1)
                # continue

                obs = {
                    'qpos': qpos,
                    'qvel': qvel,
                    'images': {cn: image1 for cn in cfg['camera_names']}
                }

                qpos_numpy = np.array(obs['qpos'])
                # print(qpos_numpy.shape)
                qpos = pre_process(qpos_numpy)
                # print('new qpos3:',qpos)
                qpos = torch.from_numpy(qpos).float().to(device).unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(obs['images'], cfg['camera_names'], device)


                query_frequency = 45
                if t % query_frequency == 0:
                    # print("policy called: ", t)
                    # input("pause here")
                    # t00 = datetime.datetime.now()
                    all_actions = policy(qpos, curr_image) #! policy called here
                    all_actions1 = copy.deepcopy(all_actions)
                    # t11 = datetime.datetime.now()
                    # print("time for policy called once : ", t11-t00)
                    # input("pause here")
                if policy_config['temporal_agg']:
                    print("smoothed action: ",t)
                    all_time_actions[[t], t:t+num_queries] = all_actions
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.11 # 0.001-0.1 no difference
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights.astype(np.float32)).to(device).unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    print("raw action: ",t)
                    # input("pause here")
                    raw_action = all_actions[:, t % query_frequency]
                raw_action_1 = all_actions1[:, t % query_frequency]
                raw_action_1 = raw_action_1.squeeze(0).cpu().numpy()
                # post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                action1 = post_process(raw_action_1)
                print(f"controller execute once!{raw_action} at {time.time()}")
                # ! take action
                print("controller execute once!",action)
                # input("pause here")
                xyz_data = action[:3]
                q_data = action[3:7]
                # print(f"q_data: {q_data}, shape: {q_data.shape}")
                # print(f"quat_action: {quat_action}")
                a = int((1-action[7]) * 255)
                a = np.clip(a, 0, 255)
                current_distance = 0.082 + 0.015 * (a / 255.0)
                rotation = R.from_quat(q_data)
                euler_angles_data = rotation.as_euler('xyz', degrees=True)
                # print('new qpos3:',xyz_data[0], xyz_data[1] , xyz_data[2], euler_angles_data[0], euler_angles_data[2], euler_angles_data[2] )
                quat_action = q_data
                x_new, y_new, z_new, _ = calculate_new_pose(
                    xyz_data[0], xyz_data[1], xyz_data[2],
                    q_data, current_distance
                )
                # Send the target pose through the robot SDK.
                bestman.robot.set_state(0)

                # Convert the target pose to millimeters and degrees.
                bestman.robot.set_position(
                    x_new * 1000,   # x (mm)
                    y_new * 1000,   # y (mm)
                    z_new * 1000,   # z (mm)
                    euler_angles_data[0],  # roll (deg)
                    euler_angles_data[1],  # pitch (deg)
                    euler_angles_data[2]   # yaw (deg)
                )


                bestman.gripper_goto(a, wait_motion=False)

                time.sleep(1/8)
                obs_replay.append(obs)
                action_replay.append(action)
                t1 = datetime.datetime.now()


        print('#'*50)
        print("STOP!")
        print('#'*50)
        bestman.go_home(100)
        cam.release()

        # create a dictionary to store the data
        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        # there may be more than one camera
        for cam_name in cfg['camera_names']:
                data_dict[f'/observations/images/{cam_name}'] = []

        # store the observations and actions
        for o, a in zip(obs_replay, action_replay):
            data_dict['/observations/qpos'].append(o['qpos'])
            data_dict['/observations/qvel'].append(o['qvel'])
            data_dict['/action'].append(a)
            # store the images
            for cam_name in cfg['camera_names']:
                data_dict[f'/observations/images/{cam_name}'].append(o['images'][cam_name])

        max_timesteps = len(data_dict['/observations/qpos'])
        output_dir = args.output_dir or cfg['dataset_dir']
        os.makedirs(output_dir, exist_ok=True)
        today = date.today()
        path = os.path.join(output_dir, f"{today}_rollout_{i}.hdf5")
        with h5py.File(path, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = False
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in cfg['camera_names']:
                image_shape = np.asarray(data_dict[f'/observations/images/{cam_name}'][0]).shape
                _ = image.create_dataset(cam_name, (max_timesteps, *image_shape), dtype='uint8',
                                        chunks=(1, *image_shape))
            qpos = obs.create_dataset('qpos', (max_timesteps, cfg['state_dim']))
            qvel = obs.create_dataset('qvel', (max_timesteps, cfg['state_dim']))
            # image = obs.create_dataset("image", (max_timesteps, 240, 320, 3), dtype='uint8', chunks=(1, 240, 320, 3))
            action = root.create_dataset('action', (max_timesteps, cfg['action_dim']))

            for name, array in data_dict.items():
                root[name][...] = array

    # disable robot
    # xarm.disconnect()+
