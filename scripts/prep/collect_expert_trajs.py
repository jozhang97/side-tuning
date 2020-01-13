# script based off notebooks/CollectExpertTrajectories.ipynb and notebooks/ProcessTrajectories.ipynb
# e.g. python -m scripts.collect_expert_trajs medium 10000 train
from evkit.preprocess.transforms import rescale_centercrop_resize, map_pool_collated, taskonomy_features_transform
from evkit.env.util.occupancy_map import OccupancyMap
from evkit.env.habitat.habitatnavenv import transform_target
from evkit.models.shortest_path_follower import ShortestPathFollower

import habitat
# from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video

import collections
import cv2
import numpy as np
import os
from PIL import Image  # Image.fromarray
import sys
import torch
from tqdm import tqdm

DATA_SIZE = sys.argv[1] if len(sys.argv) >= 4 else 'small'
SAMPLE_RATE = eval(sys.argv[2]) if len(sys.argv) >= 4 else 100000
DATA_SPLIT = sys.argv[3] if len(sys.argv) >= 4 else 'train'
DATA_DIR = sys.argv[4] if len(sys.argv) >= 5 else '/mnt/data/expert_trajs'
SAVE_VIDEO = False

class SimpleRLEnv(habitat.RLEnv):
    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


def draw_top_down_map(info, heading, output_size):
    top_down_map = maps.colorize_topdown_map(
        info["top_down_map"]["map"]
    )
    original_map_size = top_down_map.shape[:2]
    map_scale = np.array(
        (1, original_map_size[1] * 1.0 / original_map_size[0])
    )
    new_map_size = np.round(output_size * map_scale).astype(np.int32)
    # OpenCV expects w, h but map size is in h, w
    top_down_map = cv2.resize(top_down_map, (new_map_size[1], new_map_size[0]))

    map_agent_pos = info["top_down_map"]["agent_map_coord"]
    map_agent_pos = np.round(
        map_agent_pos * new_map_size / original_map_size
    ).astype(np.int32)
    top_down_map = maps.draw_agent(
        top_down_map,
        map_agent_pos,
        heading - np.pi / 2,
        agent_radius_px=top_down_map.shape[0] / 40,
    )
    return top_down_map

# set up env
config = habitat.get_config(f"/root/perception_module/habitat-api/configs/tasks/pointnav_gibson_{DATA_SPLIT}.yaml")
config.defrost()
config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
config.TASK.SENSORS.append("HEADING_SENSOR")
config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = 0
config.freeze()
env = SimpleRLEnv(config=config)
env.habitat_env.episodes = env.habitat_env.episodes[::SAMPLE_RATE]
print(f"Environment creation successful, collecting {len(env.habitat_env.episodes)} episodes.")

# set up expert
follower = ShortestPathFollower(env.habitat_env.sim, goal_radius=0.2, return_one_hot=False)
print(f'Loaded shortest path follower from {ShortestPathFollower}')
follower.mode = 'geodesic_path'

# Transforms: need rgb_filled, map, ~target, ~taskonomy
DEFAULT_MAP_KWARGS = {
                'map_building_size': 22,   # How large to make the IMU-based map
                'map_max_pool': False,     # Use max-pooling on the IMU-based map
                'use_cuda': False,
                'history_size': None,      # How many prior steps to include on the map
            }
map_transform = map_pool_collated((3,84,84))(None)[0]
Shape = collections.namedtuple('Shape', 'shape')
features_transform_pre = rescale_centercrop_resize((3,256,256))(Shape(shape=(256,256,3)))[0]
tasks = ['curvature', 'denoising']
features_transform_dict = {task: taskonomy_features_transform(f'/mnt/models/{task}_encoder.dat')(None)[0] for task in tasks}
target_dim = 16

# set up data directory
EXPERT_DIR = os.path.join(DATA_DIR, DATA_SIZE)
IMAGE_DIR = os.path.join(EXPERT_DIR, 'videos')
TRAJ_DIR = os.path.join(EXPERT_DIR, DATA_SPLIT)

if not os.path.exists(IMAGE_DIR) and SAVE_VIDEO:
    os.makedirs(IMAGE_DIR)

if not os.path.exists(TRAJ_DIR):
    os.makedirs(TRAJ_DIR)

# main loop: collect data
for episode in tqdm(range(len(env.habitat_env.episodes))):
    observations = env.reset()

    images = []
    traj = []
    omap = OccupancyMap(initial_pg=observations['pointgoal'], map_kwargs=DEFAULT_MAP_KWARGS)
    while not env.habitat_env.episode_over:
        # postprocess and log (state, action) pairs
        observations['rgb_filled'] = observations['rgb']
        observations['target'] = np.moveaxis(np.tile(transform_target(observations['pointgoal']), (target_dim, target_dim, 1)), -1, 0)
        observations['map'] = omap.construct_occupancy_map()
        del observations['rgb']

        # agent step
        best_action = follower.get_next_action(env.habitat_env.current_episode.goals[0].position).value
        traj.append([observations, best_action])

        # env step
        observations, reward, done, info = env.step(best_action)
        omap.add_pointgoal(observations['pointgoal'])  # s_{t+1}
        omap.step(best_action)  # a_t

        # viz
        if SAVE_VIDEO:
            im = observations["rgb"]
            top_down_map = draw_top_down_map(
                info, observations["heading"], im.shape[0]
            )
            output_im = np.concatenate((im, top_down_map), axis=1)
            images.append(output_im)

    # batched processing
    batch_size = 50

    all_taskonomys = []
    taskonomys_input = torch.stack([features_transform_pre(s['rgb_filled']) for s, a in traj])
    for task in tasks:
        taskonomys_lst = []
        for taskonomys_chunk in torch.split(taskonomys_input, batch_size):
            taskonomys_chunk = features_transform_dict[task](taskonomys_chunk)
            taskonomys_lst.append(taskonomys_chunk.to('cpu', non_blocking=True))
        taskonomys = torch.cat(taskonomys_lst).numpy()
        all_taskonomys.append(taskonomys)

    maps_lst = []
    maps = torch.stack([torch.Tensor(s['map']) for s, a in traj])
    for maps_chunk in torch.split(maps, batch_size):
        maps_chunk = map_transform(maps_chunk)
        maps_lst.append(maps_chunk.to('cpu', non_blocking=True))
    maps = torch.cat(maps_lst).numpy()

    for (s, _), mapp in zip(traj, maps):
        s['map'] = mapp

    for task, taskonomy_data in zip(tasks, all_taskonomys):
        for (s, _), taskonomy_frame in zip(traj, taskonomy_data):
            s[f'taskonomy_{task}'] = taskonomy_frame
            if task == 'curvature':  # for backwards support
                s['taskonomy'] = taskonomy_frame


    # write trajectory
    cur_scene_name = env._env.current_episode.scene_id.split('/')[-1].split('.')[0]
    episode_id = env._env.current_episode.episode_id
    single_traj_dir = os.path.join(TRAJ_DIR, cur_scene_name, f'episode_{episode_id}')
    os.makedirs(single_traj_dir, exist_ok=True)

    for i, (obs, act) in enumerate(traj):
        Image.fromarray(obs['rgb_filled']).save(os.path.join(single_traj_dir, f'rgb_filled_{i:03d}.png'))
        del obs['rgb_filled']
        np.savez_compressed(os.path.join(single_traj_dir, f'action_{i:03d}.npz'), act)
        for k, v in obs.items():
            np.savez_compressed(os.path.join(single_traj_dir, f'{k}_{i:03d}.npz'), v)

    if SAVE_VIDEO:
        images_to_video(images, IMAGE_DIR, str(episode))