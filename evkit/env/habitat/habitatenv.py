# This code is a heavily modified version of a similar file in Habitat
import copy
from collections import deque, Counter
from gym import spaces
import gzip
import numpy as np
import os
import random
from time import time
import torch
import torch.nn as nn
import warnings

from habitat.config.default import get_config as cfg_env
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
import habitat

from .config.default import cfg as cfg_baseline
from evkit.env.wrappers import ProcessObservationWrapper, VisdomMonitor
from evkit.env.habitat.wrapperenv import HabitatPreprocessVectorEnv, PreprocessEnv
from evkit.env.habitat.habitatexpenv import ExplorationRLEnv
from evkit.env.habitat.habitatnavenv import MidlevelNavRLEnv
from evkit.env.habitat.utils import shuffle_episodes


def make_habitat_vector_env(scenario='PointNav',
                            num_processes=2,
                            target_dim=7,
                            preprocessing_fn=None,
                            log_dir=None,
                            visdom_name='main',
                            visdom_log_file=None,
                            visdom_server='localhost',
                            visdom_port='8097',
                            vis_interval=200,
                            train_scenes=None,
                            val_scenes=None,
                            num_val_processes=0,
                            swap_building_k_episodes=10,
                            gpu_devices=[0],
                            map_kwargs={},
                            reward_kwargs={},
                            seed=42,
                            test_mode=False,
                            debug_mode=False,
                            scenario_kwargs={},
                           ):
    assert map_kwargs['map_building_size'] > 0, 'Map building size must be positive!'
    default_reward_kwargs = {
                'slack_reward': -0.01,
                'success_reward': 10,
                'use_visit_penalty': False,
                'visit_penalty_coef': 0,
                'penalty_eps': 999,
                'sparse': False,
                'dist_coef': 1.0,
            }
    for k, v in default_reward_kwargs.items():
        if k not in reward_kwargs:
            reward_kwargs[k] = v

    habitat_path = os.path.dirname(os.path.dirname(habitat.__file__))
    if scenario == 'PointNav' or scenario == 'Exploration':
        task_config = os.path.join(habitat_path, 'configs/tasks/pointnav_gibson_train.yaml')
        # only difference is that Exploration needs DEPTH_SENSOR but that is added in the Env
        # task_config = os.path.join(habitat_path, 'configs/tasks/exploration_gibson.yaml')
    else:
        assert False, f'Do not recognize scenario {scenario}'

    env_configs = []
    baseline_configs = []
    encoders = []
    target_dims = []
    is_val = []

    # Assign specific episodes to each process
    config_env = cfg_env(task_config)

    # Load dataset
    print('Loading val dataset (partition by episode)...')
    datasetfile_path = config_env.DATASET.POINTNAVV1.DATA_PATH.format(split='val')
    dataset = PointNavDatasetV1()
    with gzip.open(datasetfile_path, "rt") as f:
        dataset.from_json(f.read())
    val_datasets = get_splits(dataset, max(num_val_processes, 1))
#     for d in val_datasets:
#         d.episodes = [d.episodes[0]]
    print('Loaded.')

    print('Loading train dataset (partition by building)...')
    train_datasets = []
    if num_processes - num_val_processes > 0:
#         dataset = PointNavDatasetV1(config_env.DATASET)
        train_datasets = [None for _ in range(num_processes - num_val_processes)]
    print('Loaded.')


    # Assign specific buildings to each process
    if num_processes > num_val_processes:
        train_process_scenes = [[] for _ in range(num_processes - num_val_processes)]
        if train_scenes is None:
            train_scenes = PointNavDatasetV1.get_scenes_to_load(config_env.DATASET)
            random.shuffle(train_scenes)

        for i, scene in enumerate(train_scenes):
            train_process_scenes[i % len(train_process_scenes)].append(scene)

        # If n processes > n envs, some processes can use all envs
        for j, process in enumerate(train_process_scenes):
            if len(process) == 0:
                train_process_scenes[j] = list(train_scenes)

    get_scenes = lambda d: list(Counter([e.scene_id.split('/')[-1].split(".")[0] for e in d.episodes]).items())
    for i in range(num_processes):
        config_env = cfg_env(task_config)
        config_env.defrost()

        if i < num_processes - num_val_processes:
            config_env.DATASET.SPLIT = 'train'
#             config_env.DATASET.POINTNAVV1.CONTENT_SCENES = get_scenes(train_datasets[i])
            config_env.DATASET.POINTNAVV1.CONTENT_SCENES = train_process_scenes[i]
        else:
            val_i = i - (num_processes - num_val_processes)
            config_env.DATASET.SPLIT = 'val'
            if val_scenes is not None:
                config_env.DATASET.POINTNAVV1.CONTENT_SCENES = val_scenes
            else:
                config_env.DATASET.POINTNAVV1.CONTENT_SCENES = get_scenes(val_datasets[val_i])

        print("Env {}:".format(i), config_env.DATASET.POINTNAVV1.CONTENT_SCENES)

        config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu_devices[i % len(gpu_devices)]
        config_env.SIMULATOR.SCENE = os.path.join(habitat_path, config_env.SIMULATOR.SCENE)
        config_env.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR"]

        # Now define the config for the sensor
#         config.TASK.AGENT_POSITION_SENSOR = habitat.Config()
#         config.TASK.AGENT_POSITION_SENSOR.TYPE = "agent_position_sensor"
#         config.TASK.SENSORS.append("AGENT_POSITION_SENSOR")
        
        config_env.TASK.MEASUREMENTS.append('COLLISIONS')

        config_env.freeze()
        env_configs.append(config_env)
        config_baseline = cfg_baseline()
        baseline_configs.append(config_baseline)
        encoders.append(preprocessing_fn)
        target_dims.append(target_dim)

    should_record = [(i == 0 or i == (num_processes - num_val_processes)) for i in range(num_processes)]
    if debug_mode:
        env = make_env_fn(scenario, env_configs[0], baseline_configs[0], 0, 0, 1, target_dim, log_dir,
                           visdom_name, visdom_log_file, vis_interval, visdom_server, visdom_port,
                           swap_building_k_episodes, map_kwargs, reward_kwargs, True, seed,
                           test_mode, (train_datasets + val_datasets)[0], scenario_kwargs)
        envs = PreprocessEnv(env, preprocessing_fn = preprocessing_fn)
    else:
        envs = HabitatPreprocessVectorEnv(
            make_env_fn=make_env_fn,
            env_fn_args=tuple(
                tuple(
                    zip([scenario for _ in range(num_processes)],
                    env_configs,
                    baseline_configs,
                    range(num_processes),
                    [num_val_processes for _ in range(num_processes)],
                    [num_processes for _ in range(num_processes)],
                    target_dims,
                    [log_dir for _ in range(num_processes)],
                    [visdom_name for _ in range(num_processes)],
                    [visdom_log_file for _ in range(num_processes)],
                    [vis_interval for _ in range(num_processes)],
                    [visdom_server for _ in range(num_processes)],
                    [visdom_port for _ in range(num_processes)],
                    [swap_building_k_episodes for _ in range(num_processes)],
                    [map_kwargs for _ in range(num_processes)],
                    [reward_kwargs for _ in range(num_processes)],
                    should_record,
                    [seed + i for i in range(num_processes)],
                    [test_mode for _ in range(num_processes)],
                    train_datasets + val_datasets,
                    [scenario_kwargs for _ in range(num_processes)],
                   )
                )
            ),
            preprocessing_fn=preprocessing_fn,
        )
        envs.observation_space = envs.observation_spaces[0]
    envs.action_space = spaces.Discrete(3)
    envs.reward_range = None
    envs.metadata = None
    envs.is_embodied = True
    return envs


def make_env_fn(scenario,
                config_env,
                config_baseline,
                rank,
                num_val_processes,
                num_processes,
                target_dim,
                log_dir,
                visdom_name,
                visdom_log_file,
                vis_interval,
                visdom_server,
                visdom_port,
                swap_building_k_episodes,
                map_kwargs,
                reward_kwargs,
                should_record,
                seed, 
                test_mode,
                dataset,
                scenario_kwargs):
    if config_env.DATASET.SPLIT == 'train':
        dataset = PointNavDatasetV1(config_env.DATASET)

    habitat_path = os.path.dirname(os.path.dirname(habitat.__file__))
    for ep in dataset.episodes:
        ep.scene_id = os.path.join(habitat_path, ep.scene_id)

    config_env.defrost()
    config_env.SIMULATOR.SCENE = dataset.episodes[0].scene_id
    config_env.freeze()

    if scenario == 'PointNav':
        dataset.episodes = [epi for epi in dataset.episodes if epi.info['geodesic_distance'] < scenario_kwargs['max_geodesic_dist']]
        env = MidlevelNavRLEnv(config_env=config_env,
                           config_baseline=config_baseline,
                           dataset=dataset,
                           target_dim=target_dim,
                           map_kwargs=map_kwargs,
                           reward_kwargs=reward_kwargs,
                           loop_episodes=not test_mode,
                           scenario_kwargs=scenario_kwargs)
    elif scenario == 'Exploration':
        env = ExplorationRLEnv(config_env=config_env,
                               config_baseline=config_baseline,
                               dataset=dataset,
                               map_kwargs=map_kwargs,
                               reward_kwargs=reward_kwargs,
                               loop_episodes=not test_mode,
                               scenario_kwargs=scenario_kwargs)
    else:
        assert False, f'do not recognize scenario {scenario}'

    if test_mode:
        env.episodes = env.episodes
    else:
        env.episodes = shuffle_episodes(env, swap_every_k=swap_building_k_episodes)

    env.seed(seed)
    if should_record and visdom_log_file is not None:
        print(f"Recording videos from env {rank} every {vis_interval} episodes (via visdom)")
        env = VisdomMonitor(env,
                       directory=os.path.join(log_dir, visdom_name),
                       video_callable=lambda x: x % vis_interval == 0,
                       uid=str(rank),
                       server=visdom_server,
                       port=visdom_port,
                       visdom_log_file=visdom_log_file,
                       visdom_env=visdom_name)

    return env


def get_splits(
        dataset,
        num_splits: int,
        episodes_per_split: int = None,
        remove_unused_episodes: bool = False,
        collate_scene_ids: bool = True,
        sort_by_episode_id: bool = False,
        allow_uneven_splits: bool = True,
    ) :
        r"""Returns a list of new datasets, each with a subset of the original
        episodes. All splits will have the same number of episodes, but no
        episodes will be duplicated.
        Args:
            num_splits: the number of splits to create.
            episodes_per_split: if provided, each split will have up to
                this many episodes. If it is not provided, each dataset will
                have ``len(original_dataset.episodes) // num_splits`` 
                episodes. If max_episodes_per_split is provided and is 
                larger than this value, it will be capped to this value.
            remove_unused_episodes: once the splits are created, the extra
                episodes will be destroyed from the original dataset. This
                saves memory for large datasets.
            collate_scene_ids: if true, episodes with the same scene id are
                next to each other. This saves on overhead of switching 
                between scenes, but means multiple sequential episodes will 
                be related to each other because they will be in the 
                same scene.
            sort_by_episode_id: if true, sequences are sorted by their episode
                ID in the returned splits.
            allow_uneven_splits: if true, the last split can be shorter than
                the others. This is especially useful for splitting over
                validation/test datasets in order to make sure that all
                episodes are copied but none are duplicated.
        Returns:
            a list of new datasets, each with their own subset of episodes.
        """
        assert (
            len(dataset.episodes) >= num_splits
        ), "Not enough episodes to create this many splits."
        if episodes_per_split is not None:
            assert not allow_uneven_splits, (
                "You probably don't want to specify allow_uneven_splits"
                " and episodes_per_split."
            )
            assert num_splits * episodes_per_split <= len(dataset.episodes)

        new_datasets = []

        if allow_uneven_splits:
            stride = int(np.ceil(len(dataset.episodes) * 1.0 / num_splits))
            split_lengths = [stride] * (num_splits - 1)
            split_lengths.append(
                (len(dataset.episodes) - stride * (num_splits - 1))
            )
        else:
            if episodes_per_split is not None:
                stride = episodes_per_split
            else:
                stride = len(dataset.episodes) // num_splits
            split_lengths = [stride] * num_splits

        num_episodes = sum(split_lengths)

        rand_items = np.random.choice(
            len(dataset.episodes), num_episodes, replace=False
        )
        if collate_scene_ids:
            scene_ids = {}
            for rand_ind in rand_items:
                scene = dataset.episodes[rand_ind].scene_id
                if scene not in scene_ids:
                    scene_ids[scene] = []
                scene_ids[scene].append(rand_ind)
            rand_items = []
            list(map(rand_items.extend, scene_ids.values()))
        ep_ind = 0
        new_episodes = []
        for nn in range(num_splits):
            new_dataset = copy.copy(dataset)  # Creates a shallow copy
            new_dataset.episodes = []
            new_datasets.append(new_dataset)
            for ii in range(split_lengths[nn]):
                new_dataset.episodes.append(dataset.episodes[rand_items[ep_ind]])
                ep_ind += 1
            if sort_by_episode_id:
                new_dataset.episodes.sort(key=lambda ep: ep.episode_id)
            new_episodes.extend(new_dataset.episodes)
        if remove_unused_episodes:
            dataset.episodes = new_episodes
        return new_datasets


# # Define the sensor and register it with habitat
# # For the sensor, we will register it with a custom name
# @habitat.registry.register_sensor(name="agent_position_sensor")
# class AgentPositionSensor(habitat.Sensor):
#     def __init__(self, sim, config, **kwargs: Any):
#         super().__init__(config=config)

#         self._sim = sim
#         # Prints out the answer to life on init
#         print("The answer to life is", self.config.ANSWER_TO_LIFE)

#     # Defines the name of the sensor in the sensor suite dictionary
#     def _get_uuid(self, *args: Any, **kwargs: Any):
#         return "agent_position"

#     # Defines the type of the sensor
#     def _get_sensor_type(self, *args: Any, **kwargs: Any):
#         return habitat.SensorTypes.POSITION

#     # Defines the size and range of the observations of the sensor
#     def _get_observation_space(self, *args: Any, **kwargs: Any):
#         return spaces.Box(
#             low=np.finfo(np.float32).min,
#             high=np.finfo(np.float32).max,
#             shape=(3,),
#             dtype=np.float32,
#         )

#     # This is called whenver reset is called or an action is taken
#     def get_observation(self, observations, episode):
#         return self._sim.get_agent_state().position

