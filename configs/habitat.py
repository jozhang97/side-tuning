# Habitat configs
#   This should be sourced by the training script,
#   which must save a sacred experiment in the variable "ex"
#   For descriptions of all fields, see configs/core.py
# Related configs in ./configs/rl.py and ./configs/rl_extra.py
import numpy as np


@ex.named_config
def cfg_habitat():
    uuid = 'habitat_core'
    cfg = {}
    cfg['learner'] = {
        'algo': 'ppo',               # Learning algorithm for RL agent. Currently PPO, SLAM, imitation_learning
        'clip_param': 0.1,           # Clip param for trust region in PPO
        'entropy_coef': 1e-4,        # Weighting of the entropy term in PPO
        'eps': 1e-5,                 # Small epsilon to prevent divide-by-zero
        'gamma': 0.99,               # Gamma to use if env.observation_space.shape = 1
        'internal_state_size': 512,  # If using a recurrent policy, what state size to use (if no recurrent policy, make this small for memory savings)
        'lr': 1e-4,                  # Learning rate for algorithm
        'num_steps': 1000,            # Length of each rollout (grab 'num_steps' consecutive samples to form rollout)
        'num_mini_batch': 8,         # Size of PPO minibatch (from rollout, block into this many minibatchs to compute losses on)
        'num_stack': 4,              # Frames that each cell (CNN) can see
        'max_grad_norm': 0.5,        # Clip grads
        'ppo_epoch': 8,              # Number of times PPO goes over the buffer
        'recurrent_policy': False,   # Use a recurrent version with the cell as the standard model
        'tau': 0.95,                 # When using GAE
        'use_gae': True,             # Whether to use GAE
        'value_loss_coef': 1e-3,     # Weighting of value_loss in PPO
        'perception_network_reinit': False,  # reinitialize the perception_module, used when checkpoint is used
        'perception_network': 'AtariNet',
        'perception_network_kwargs': {
            'extra_kwargs': {
                'normalize_taskonomy': True
            }
        },
        'test': False,
        'use_replay': True,
        'replay_buffer_size': 3000,  # This is stored on CPU
        'on_policy_epoch': 8,        # Number of on policy rollouts in each update
        'off_policy_epoch': 8,
        'slam_class': None,
        'slam_kwargs': {},
        'loss_kwargs': {             # Used for regularization losses (e.g. weight tying)
            'intrinsic_loss_coefs': [],
            'intrinsic_loss_types': [],
        },
        'deterministic': False,
        'rollout_value_batch_multiplier': 2,
        'cache_kwargs': {},
        'optimizer_class': 'optim.Adam',
        'optimizer_kwargs': {},
    }
    cfg['env'] = {
        'add_timestep': False,             # Add timestep to the observation
        'env_name': 'Habitat_PointNav',    # Environment to use
        "env_specific_kwargs": {
            'swap_building_k_episodes': 10,
            'gpu_devices': [0],
            'scenario_kwargs': {           # specific to the scenario - pointnav or exploration
                'use_depth': False,
                'max_geodesic_dist': 99999 # For PointNav, we skip episodes that are too "hard"
            },
            'map_kwargs': {
                'map_building_size': 22,   # How large to make the IMU-based map
                'map_max_pool': False,     # Use max-pooling on the IMU-based map
                'use_cuda': False,         
                'history_size': None,      # How many prior steps to include on the map
            },
            'target_dim': 16,              # Taskonomy reps: 16, scratch: 9, map_only: 1
            # 'val_scenes': ['Denmark', 'Greigsville', 'Eudora', 'Pablo', 'Elmira', 'Mosquito', 'Sands', 'Swormville', 'Sisters', 'Scioto', 'Eastville', 'Edgemere', 'Cantwell', 'Ribera'],
            'val_scenes': None,
            'train_scenes': None,
#             'train_scenes': ['Beach'],
          },
        'sensors': {
            'features': None,
            'taskonomy': None,
            'rgb_filled': None,
            'map': None,
            'target': None,
            'depth': None,
            'global_pos': None,
            'pointgoal': None,
        },
        'transform_fn_pre_aggregation': None,       # Depreciated
        'transform_fn_pre_aggregation_fn': None,    # Transformation to apply to each individual image (before batching)
        'transform_fn_pre_aggregation_kwargs': {},  # Arguments - MUST BE ABLE TO CALL eval ON ALL STRING VALUES
        'transform_fn_post_aggregation': None,      # Depreciated
        'transform_fn_post_aggregation_fn': None,   # Arguments - MUST BE ABLE TO CALL eval ON ALL STRING VALUES
        'transform_fn_post_aggregation_kwargs': {},
        'num_processes': 8,
        'num_val_processes': 1,
        'additional_repeat_count': 0,
    }
    
    cfg['saving'] = {
        'checkpoint':None,
        'checkpoint_num': None,
        'checkpoint_configs': False,  # copy the metadata of the checkpoint. YMMV.
        'log_dir': LOG_DIR,
        'log_interval': 10,
        'save_interval': 100,
        'save_dir': 'checkpoints',
        'visdom_log_file': os.path.join(LOG_DIR, 'visdom_logs.json'),
        'results_log_file': os.path.join(LOG_DIR, 'result_log.pkl'),
        'reward_log_file': os.path.join(LOG_DIR, 'rewards.pkl'),
        'vis_interval': 200,
        'visdom_server': 'localhost',
        'visdom_port': '8097',
        'obliterate_logs': False,
    }
    
    cfg['training'] = {
        'cuda': True,
        'gpu_devices': None,   # None uses all devices, otherwise give a list of devices
        'seed': 42,
        'num_frames': 1e8,
        'resumable': False,
    }

@ex.named_config
def cfg_test():
    cfg = {}
    cfg['saving'] = {
        'resumable': True,
        'checkpoint_configs': True,
    }

    override = {}
    override['saving'] = {
        'visdom_server': 'localhost',
    }
    override['env'] = {
        'num_processes': 10,
        'num_val_processes': 10,
        'env_specific_kwargs': {
            'test_mode': True,
            'scenario_kwargs': {           # specific to the scenario - pointnav or exploration
                'max_geodesic_dist': 99999
            }
        }
    }
    override['learner'] = {
        'test_k_episodes': 994,
        'test': True,
    }

####################################
# Active Tasks
####################################
@ex.named_config
def planning():
    uuid = 'habitat_planning'
    cfg = {}
    cfg['learner'] = {
        'perception_network_kwargs': {  # while related to the agent, these are ENVIRONMENT specific
            'n_map_channels': 3,
            'use_target': True,
        }
    }
    cfg['env'] = {
        'env_name': 'Habitat_PointNav',    # Environment to use
        'transform_fn_pre_aggregation_fn': 'TransformFactory.independent',
        'transform_fn_pre_aggregation_kwargs': {  # the most basic sensors
            'names_to_transforms': {
                'map': 'identity_transform()',
                'global_pos': 'identity_transform()',
                'target': 'identity_transform()',
            },
            'keep_unnamed': False,
        },
        'transform_fn_post_aggregation_fn': 'TransformFactory.independent',
        'transform_fn_post_aggregation_kwargs': {
            'names_to_transforms': {
                'map':'map_pool_collated((3,84,84))',
            },
            'keep_unnamed': True,
        }
    }

@ex.named_config
def exploration():
    uuid = 'habitat_exploration'
    cfg = {}
    cfg['learner'] = {
        'lr': 1e-3,                  # Learning rate for algorithm
        'perception_network_kwargs': {  # while related to the agent, these are ENVIRONMENT specific
            'n_map_channels': 1,
            'use_target': False,
        }
    }
    cfg['env'] = {
        'env_name': 'Habitat_Exploration',    # Environment to use
        'transform_fn_pre_aggregation_fn': 'TransformFactory.independent',
        'transform_fn_pre_aggregation_kwargs': {  # the most basic sensors
            'names_to_transforms': {
                'map': 'identity_transform()',
                'global_pos': 'identity_transform()',
            },
            'keep_unnamed': False,
        },
        'transform_fn_post_aggregation_fn': 'TransformFactory.independent',
        'transform_fn_post_aggregation_kwargs': {
            'names_to_transforms': {
                'map':'map_pool_collated((1,84,84))',
            },
            'keep_unnamed': True,
        },
        # For exploration, always map_pool in the pre_aggregation, do not in post_aggregation
        # 'map':rescale_centercrop_resize((1,84,84)),
        "env_specific_kwargs": {
            'scenario_kwargs': {
                'max_episode_steps': 1000,
            },
            'map_kwargs': {
                'map_size': 84,
                'fov': np.pi / 2,
                'min_depth': 0,
                'max_depth': 1.5,
                'relative_range': True,  # scale the map_range so centered at initial agent position
                'map_x_range': [-11, 11],
                'map_y_range': [-11, 11],
                'fullvision': False,  # robot unlocks all cells in what it sees, as opposed to just center to ground
            },
            'reward_kwargs': {
                'slack_reward': 0,
            }
        },
    }

####################################
# Settings
####################################
@ex.named_config
def small_settings5():
    # hope this runs well on gestalt, one GPU (moderate RAM requirements)
    uuid = 'habitat_small_settings5'
    cfg = {}
    cfg['learner'] = {
        'num_steps': 512,            # Length of each rollout (grab 'num_steps' consecutive samples to form rollout)
        'replay_buffer_size': 1024,  # This is stored on CPU
        'on_policy_epoch': 5,        # Number of on policy rollouts in each update
        'off_policy_epoch': 10,
        'num_mini_batch': 24,         # Size of PPO minibatch (from rollout, block into this many minibatchs to compute losses on)
        'rollout_value_batch_multiplier': 1,
    }
    cfg['env'] = {
        'num_processes': 6,
        'num_val_processes': 1,
    }

@ex.named_config
def cvpr_settings():
    # Settings we used for CVPR 2019 Habitat Challenge, see https://github.com/facebookresearch/habitat-challenge
    uuid = 'habitat_cvpr_settings'
    cfg = {}
    cfg['learner'] = {
        'num_steps': 512,            # Length of each rollout (grab 'num_steps' consecutive samples to form rollout)
        'replay_buffer_size': 4096,  # This is stored on CPU
        'on_policy_epoch': 8,        # Number of on policy rollouts in each update
        'off_policy_epoch': 8,
        'num_mini_batch': 8,         # Size of PPO minibatch (from rollout, block into this many minibatchs to compute losses on)
        'rollout_value_batch_multiplier': 1,
    }
    cfg['env'] = {
        'num_processes': 6,
        'num_val_processes': 1,
    }

####################################
# Development
####################################
@ex.named_config
def prototype():
    uuid='test'
    cfg = {}
    cfg['env'] = {
        'num_processes': 2,
        'num_val_processes': 1,
        'env_specific_kwargs': {
            'train_scenes': ['Adrian'],
            'val_scenes': ['Denmark'],
        }
    }
    cfg['saving'] = {
        'log_interval': 2,
        'vis_interval': 1,
    }

@ex.named_config
def debug():
    # this does not use VectorizedEnv, so supports pdb
    uuid='test'
    cfg = {}
    override = {}
    cfg['learner'] = {
        'num_steps': 100,
        'replay_buffer_size': 300,
        'deterministic': True,
    }
    cfg['env'] = {
        'num_processes': 1,
        'num_val_processes': 0,
        'env_specific_kwargs': {
            'train_scenes': ['Adrian'],
            'debug_mode': True,
        }
    }
    cfg['saving'] = {
        'log_interval': 2,
        'vis_interval': 1,
    }
    override['env'] = {
        'num_processes': 1,
        'num_val_processes': 0,
        "env_specific_kwargs": {
            'debug_mode': True,
        }
    }

