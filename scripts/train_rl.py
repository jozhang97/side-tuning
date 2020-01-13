# train_rl.py
# Authors: Sasha Sax (1,3), Bradley Emi (2), Jeffrey Zhang (1) -- UC Berkeley, FAIR, Stanford VL
# Desc: Train or test an agent using PPO.
# Usage:
#    python -m scripts.train_rl DIRECTORY_TO_SAVE_RESULTS run_training with uuid=EXP_UUID [CFG1 ...] [cfg.SUB_CFG1.PROPERTY1 ...]
# Notes:
#     (i) must be run from parent directory (top-level of git)
#     (ii) currently, a visdom instance MUST be used or the script will fail. Defaults to localhost.

import os
import GPUtil

# If you need one GPU, I will pick it here for you
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    gpu = [str(g) for g in GPUtil.getAvailable(maxMemory=0.2, order='random')]
    assert len(gpu) > 0, 'No available GPUs'
    print('Using GPU', ','.join(gpu))
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu)
import sys
import shutil
import copy
import glob
from gym import logger
from gym import spaces
import gym
import json
import logging
import numpy as np
import pprint
import psutil
import random
import runpy
import sacred
import subprocess
import time
import torch
import torchvision.utils
import warnings
# torch.autograd.set_detect_anomaly(True)

from evkit.env.wrappers import ProcessObservationWrapper
from evkit.env import EnvFactory
from evkit.models.architectures import AtariNet, TaskonomyFeaturesOnlyNet
from evkit.models.sidetuning import RLSidetuneWrapper, RLSidetuneNetwork
from evkit.models.taskonomy_network import TaskonomyNetwork
from evkit.models.actor_critic_module import NaivelyRecurrentACModule
from evkit.models.expert import Expert
from evkit.preprocess.transforms import rescale_centercrop_resize, rescale, grayscale_rescale, cross_modal_transform, \
    identity_transform, rescale_centercrop_resize_collated, map_pool_collated, map_pool, taskonomy_features_transform, \
    image_to_input_collated, taskonomy_multi_features_transform
from evkit.preprocess.baseline_transforms import blind, pixels_as_state
from evkit.preprocess import TransformFactory
import evkit.rl.algo
from evkit.rl.policy import Policy, PolicyWithBase, BackoutPolicy
from evkit.rl.storage import RolloutSensorDictStorage, RolloutSensorDictReplayBuffer, StackedSensorDictStorage
from evkit.saving.checkpoints import checkpoint_name, save_checkpoint, last_archived_run, archive_current_run
from evkit.saving.observers import FileStorageObserverWithExUuid
from evkit.utils.misc import Bunch, cfg_to_md, compute_weight_norm, is_interactive, remove_whitespace, \
    update_dict_deepcopy, eval_dict_values
import evkit.utils.logging
from evkit.utils.parallel import _CustomDataParallel
from evkit.utils.viz.core import log_input_images
from evkit.utils.random import set_seed
from tlkit.utils import count_open
import tnt.torchnet as tnt

from evkit.models.srl_architectures import *
from evkit.models.srl_architectures import srl_features_transform
from evkit.models.alexnet import *

# Set up experiment using SACRED
from sacred.arg_parser import get_config_updates
from docopt import docopt
ex = sacred.Experiment(name="RL Training", interactive=is_interactive())
LOG_DIR = sys.argv[1].strip()
sys.argv.pop(1)
runpy.run_module('configs.core', init_globals=globals())
runpy.run_module('configs.rl', init_globals=globals())
runpy.run_module('configs.rl_extra', init_globals=globals())
runpy.run_module('configs.habitat', init_globals=globals())
runpy.run_module('configs.gibson', init_globals=globals())
runpy.run_module('configs.doom', init_globals=globals())
runpy.run_module('configs.imitation_learning', init_globals=globals())
runpy.run_module('configs.shared', init_globals=globals())

logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger()

@ex.command
def prologue(cfg, uuid):
    os.makedirs(LOG_DIR, exist_ok=True)
    assert not (cfg['saving']['obliterate_logs'] and cfg['training']['resumable']), 'cannot obliterate logs and resume training'
    if cfg['saving']['obliterate_logs']:
        assert LOG_DIR, 'LOG_DIR cannot be empty'
        subprocess.call(f'rm -rf {LOG_DIR}', shell=True)
    if cfg['training']['resumable']:
        archive_current_run(LOG_DIR, uuid)

@ex.main
def run_training(cfg, uuid, override={}):
    try:
        logger.info("-------------\nStarting with configuration:\n" + pprint.pformat(cfg))
        logger.info("UUID: " + uuid)
        torch.set_num_threads(1)
        set_seed(cfg['training']['seed'])

        # get new output_dir name (use for checkpoints)
        old_log_dir = cfg['saving']['log_dir']
        changed_log_dir = False
        existing_log_paths = []
        if os.path.exists(old_log_dir) and cfg['saving']['autofix_log_dir']:
            LOG_DIR, existing_log_paths = evkit.utils.logging.unused_dir_name(old_log_dir)
            os.makedirs(LOG_DIR, exist_ok=False)
            cfg['saving']['log_dir'] = LOG_DIR
            cfg['saving']['results_log_file'] = os.path.join(LOG_DIR, 'result_log.pkl')
            cfg['saving']['reward_log_file'] = os.path.join(LOG_DIR, 'rewards.pkl')
            cfg['saving']['visdom_log_file'] = os.path.join(LOG_DIR, 'visdom_logs.json')
            changed_log_dir = True

        # Load checkpoint, config, agent
        agent = None

        if cfg['training']['resumable']:
            if cfg['saving']['checkpoint']:
                prev_run_path = cfg['saving']['checkpoint']
                if cfg['saving']['checkpoint_num'] is None:
                    ckpt_fpath = os.path.join(prev_run_path, 'checkpoints', 'ckpt-latest.dat')
                else:
                    ckpt_fpath = os.path.join(prev_run_path, 'checkpoints', f"ckpt-{cfg['saving']['checkpoint_num']}.dat")
                if cfg['saving']['checkpoint_configs']:  # update configs with values from ckpt
                    prev_run_metadata_paths = [os.path.join(prev_run_path, f)
                                               for f in os.listdir(prev_run_path)
                                               if f.endswith('metadata')]
                    prev_run_config_path = os.path.join(prev_run_metadata_paths[0], 'config.json')
                    with open(prev_run_config_path) as f:
                        config = json.load(f)  # keys are ['cfg', 'uuid', 'seed']
                    true_log_dir = cfg['saving']['log_dir']
                    cfg = update_dict_deepcopy(cfg, config['cfg'])
                    uuid = config['uuid']
                    logger.warning("Reusing config from {}".format(prev_run_config_path))
                    # the saving files should always use the new log dir
                    cfg['saving']['log_dir'] = true_log_dir
                    cfg['saving']['results_log_file'] = os.path.join(true_log_dir, 'result_log.pkl')
                    cfg['saving']['reward_log_file'] = os.path.join(true_log_dir, 'rewards.pkl')
                    cfg['saving']['visdom_log_file'] = os.path.join(true_log_dir, 'visdom_logs.json')
                if ckpt_fpath is not None and os.path.exists(ckpt_fpath):
                    checkpoint_obj = torch.load(ckpt_fpath)
                    start_epoch = checkpoint_obj['epoch']
                    logger.info("Loaded learner (epoch {}) from {}".format(start_epoch, ckpt_fpath))
                    if cfg['learner']['algo'] == 'imitation_learning':
                        actor_critic = checkpoint_obj['model']
                        try:
                            actor_critic = actor_critic.module  # remove DataParallel
                        except:
                            pass
                    else:
                        agent = checkpoint_obj['agent']
                        actor_critic = agent.actor_critic
                else:
                    logger.warning("No checkpoint found at {}".format(ckpt_fpath))
        cfg = update_dict_deepcopy(cfg, override)
        logger.info("-------------\n Running with configuration:\n" + pprint.pformat(cfg))

        # Verify configs are consistent - baked version needs to match un-baked version
        try:
            taskonomy_transform = cfg['env']['transform_fn_post_aggregation_kwargs']['names_to_transforms']['taskonomy']
            taskonomy_encoder = cfg['learner']['perception_network_kwargs']['extra_kwargs']['sidetune_kwargs']['base_weights_path']
            assert taskonomy_encoder in taskonomy_transform, f'Taskonomy PostTransform and perception network base need to match. {taskonomy_encoder} != {taskonomy_transform}'
        except KeyError:
            pass

        if cfg['training']['gpu_devices'] is None:
            cfg['training']['gpu_devices'] = list(range(torch.cuda.device_count()))
        assert not (len(cfg['training']['gpu_devices']) > 1 and 'attributes' in cfg['learner']['cache_kwargs']), 'Cannot utilize cache with more than one model GPU'

        # Make environment
        simulator, scenario = cfg['env']['env_name'].split('_')

        transform_pre_aggregation = None
        if cfg['env']['transform_fn_pre_aggregation'] is not None:
            logger.warning('Using depreciated config transform_fn_pre_aggregation')
            transform_pre_aggregation = eval(cfg['env']['transform_fn_pre_aggregation'].replace("---", "'"))
        elif 'transform_fn_pre_aggregation_fn' in cfg['env'] and cfg['env'][
            'transform_fn_pre_aggregation_fn'] is not None:
            pre_aggregation_kwargs = copy.deepcopy(cfg['env']['transform_fn_pre_aggregation_kwargs'])
            transform_pre_aggregation = eval(cfg['env']['transform_fn_pre_aggregation_fn'].replace("---", "'"))(
                **eval_dict_values(pre_aggregation_kwargs))

        if 'debug_mode' in cfg['env']['env_specific_kwargs'] and cfg['env']['env_specific_kwargs']['debug_mode']:
            assert cfg['env']['num_processes'] == 1, 'Using debug mode requires you to only use one process'

        envs = EnvFactory.vectorized(
            cfg['env']['env_name'],
            cfg['training']['seed'],
            cfg['env']['num_processes'],
            cfg['saving']['log_dir'],
            cfg['env']['add_timestep'],
            env_specific_kwargs=cfg['env']['env_specific_kwargs'],
            num_val_processes=cfg['env']['num_val_processes'],
            preprocessing_fn=transform_pre_aggregation,
            addl_repeat_count=cfg['env']['additional_repeat_count'],
            sensors=cfg['env']['sensors'],
            vis_interval=cfg['saving']['vis_interval'],
            visdom_server=cfg['saving']['visdom_server'],
            visdom_port=cfg['saving']['visdom_port'],
            visdom_log_file=cfg['saving']['visdom_log_file'],
            visdom_name=uuid)

        transform_post_aggregation = None
        if 'transform_fn_post_aggregation' in cfg['env'] and cfg['env']['transform_fn_post_aggregation'] is not None:
            logger.warning('Using depreciated config transform_fn_post_aggregation')
            transform_post_aggregation = eval(cfg['env']['transform_fn_post_aggregation'].replace("---", "'"))
        elif 'transform_fn_post_aggregation_fn' in cfg['env'] and cfg['env'][
            'transform_fn_post_aggregation_fn'] is not None:
            post_aggregation_kwargs = copy.deepcopy(cfg['env']['transform_fn_post_aggregation_kwargs'])
            transform_post_aggregation = eval(cfg['env']['transform_fn_post_aggregation_fn'].replace("---", "'"))(
                **eval_dict_values(post_aggregation_kwargs))

        if transform_post_aggregation is not None:
            transform, space = transform_post_aggregation(envs.observation_space)
            envs = ProcessObservationWrapper(envs, transform, space)

        action_space = envs.action_space
        observation_space = envs.observation_space
        retained_obs_shape = {k: v.shape
                              for k, v in observation_space.spaces.items()
                              if k in cfg['env']['sensors']}
        logger.info(f"Action space: {action_space}")
        logger.info(f"Observation space: {observation_space}")
        logger.info(
            "Retaining: {}".format(set(observation_space.spaces.keys()).intersection(cfg['env']['sensors'].keys())))

        # Finish setting up the agent
        if agent == None and cfg['learner']['algo'] == 'ppo':
            perception_model = eval(cfg['learner']['perception_network'])(
                cfg['learner']['num_stack'],
                **cfg['learner']['perception_network_kwargs'])
            base = NaivelyRecurrentACModule(
                perception_unit=perception_model,
                use_gru=cfg['learner']['recurrent_policy'],
                internal_state_size=cfg['learner']['internal_state_size'])
            actor_critic = PolicyWithBase(
                base, action_space,
                num_stacks=cfg['learner']['num_stack'],
                takeover=None,
                loss_kwargs=cfg['learner']['loss_kwargs'],
                gpu_devices=cfg['training']['gpu_devices'],
            )
            if cfg['learner']['use_replay']:
                agent = evkit.rl.algo.PPOReplay(actor_critic,
                                                cfg['learner']['clip_param'],
                                                cfg['learner']['ppo_epoch'],
                                                cfg['learner']['num_mini_batch'],
                                                cfg['learner']['value_loss_coef'],
                                                cfg['learner']['entropy_coef'],
                                                cfg['learner']['on_policy_epoch'],
                                                cfg['learner']['off_policy_epoch'],
                                                cfg['learner']['num_steps'],
                                                cfg['learner']['num_stack'],
                                                lr=cfg['learner']['lr'],
                                                eps=cfg['learner']['eps'],
                                                max_grad_norm=cfg['learner']['max_grad_norm'],
                                                gpu_devices=cfg['training']['gpu_devices'],
                                                loss_kwargs=cfg['learner']['loss_kwargs'],
                                                cache_kwargs=cfg['learner']['cache_kwargs'],
                                                optimizer_class = cfg['learner']['optimizer_class'],
                                                optimizer_kwargs = cfg['learner']['optimizer_kwargs']
                )
            else:
                agent = evkit.rl.algo.PPO(actor_critic,
                                          cfg['learner']['clip_param'],
                                          cfg['learner']['ppo_epoch'],
                                          cfg['learner']['num_mini_batch'],
                                          cfg['learner']['value_loss_coef'],
                                          cfg['learner']['entropy_coef'],
                                          lr=cfg['learner']['lr'],
                                          eps=cfg['learner']['eps'],
                                          max_grad_norm=cfg['learner']['max_grad_norm']
                )
            start_epoch = 0

            # Set up data parallel
            if torch.cuda.device_count() > 1 and (cfg['training']['gpu_devices'] is None or len(cfg['training']['gpu_devices']) > 1):
                actor_critic.data_parallel(cfg['training']['gpu_devices'])

        elif agent == None and cfg['learner']['algo'] == 'slam':
            assert cfg['learner']['slam_class'] is not None, 'Must define SLAM agent class'
            actor_critic = eval(cfg['learner']['slam_class'])(**cfg['learner']['slam_kwargs'])
            start_epoch = 0

        elif cfg['learner']['algo'] == 'expert':
            actor_critic = eval(cfg['learner']['algo_class'])(**cfg['learner']['algo_kwargs'])
            start_epoch = 0

        if cfg['learner']['algo'] == 'expert':
            assert 'debug_mode' in cfg['env']['env_specific_kwargs'] and cfg['env']['env_specific_kwargs']['debug_mode'], 'need to use debug mode with expert algo'

        if cfg['learner']['perception_network_reinit'] and cfg['learner']['algo'] == 'ppo':
            logger.info('Reinit perception network, use with caution')
            # do not reset map_tower and other parts of the TaskonomyFeaturesOnlyNetwork
            old_perception_unit = actor_critic.base.perception_unit
            new_perception_unit = eval(cfg['learner']['perception_network'])(
                cfg['learner']['num_stack'],
                **cfg['learner']['perception_network_kwargs'])
            new_perception_unit.main_perception = old_perception_unit  # main perception does not change
            actor_critic.base.perception_unit = new_perception_unit  # only x['taskonomy'] changes

            # match important configs of old model
            if (actor_critic.gpu_devices == None or len(actor_critic.gpu_devices) == 1) and len(cfg['training']['gpu_devices']) > 1:
                actor_critic.data_parallel(cfg['training']['gpu_devices'])
            actor_critic.gpu_devices = cfg['training']['gpu_devices']
            agent.gpu_devices = cfg['training']['gpu_devices']

        # Machinery for storing rollouts
        num_train_processes = cfg['env']['num_processes'] - cfg['env']['num_val_processes']
        num_val_processes = cfg['env']['num_val_processes']
        assert cfg['learner']['test'] or (cfg['env']['num_val_processes'] < cfg['env']['num_processes']), \
            "Can't train without some training processes!"
        current_obs = StackedSensorDictStorage(cfg['env']['num_processes'], cfg['learner']['num_stack'],
                                               retained_obs_shape)
        if not cfg['learner']['test']:
            current_train_obs = StackedSensorDictStorage(num_train_processes, cfg['learner']['num_stack'],
                                                         retained_obs_shape)
        logger.debug(f'Stacked obs shape {current_obs.obs_shape}')

        if cfg['learner']['use_replay'] and not cfg['learner']['test']:
            rollouts = RolloutSensorDictReplayBuffer(
                cfg['learner']['num_steps'],
                num_train_processes,
                current_obs.obs_shape,
                action_space,
                cfg['learner']['internal_state_size'],
                actor_critic,
                cfg['learner']['use_gae'],
                cfg['learner']['gamma'],
                cfg['learner']['tau'],
                cfg['learner']['replay_buffer_size'],
                batch_multiplier=cfg['learner']['rollout_value_batch_multiplier']
            )
        else:
            rollouts = RolloutSensorDictStorage(
                cfg['learner']['num_steps'],
                num_train_processes,
                current_obs.obs_shape,
                action_space,
                cfg['learner']['internal_state_size'])

        # Set up logging
        if cfg['saving']['logging_type'] == 'visdom':
            mlog = tnt.logger.VisdomMeterLogger(
                title=uuid, env=uuid,
                server=cfg['saving']['visdom_server'],
                port=cfg['saving']['visdom_port'],
                log_to_filename=cfg['saving']['visdom_log_file'])
        elif cfg['saving']['logging_type'] == 'tensorboard':
            mlog = tnt.logger.TensorboardMeterLogger(
                env=uuid,
                log_dir=cfg['saving']['log_dir'],
                plotstylecombined=True)
        else:
            raise NotImplementedError("Unknown logger type: ({cfg['saving']['logging_type']})")

        # Add metrics and logging to TB/Visdom
        loggable_metrics = ['metrics/rewards',
                            'diagnostics/dist_perplexity',
                            'diagnostics/lengths',
                            'diagnostics/max_importance_weight',
                            'diagnostics/value',
                            'losses/action_loss',
                            'losses/dist_entropy',
                            'losses/value_loss',
                            'introspect/alpha']
        if 'intrinsic_loss_types' in cfg['learner']['loss_kwargs']:
            for iloss in cfg['learner']['loss_kwargs']['intrinsic_loss_types']:
                loggable_metrics.append(f"losses/{iloss}")
        core_metrics = ['metrics/rewards', 'diagnostics/lengths']
        debug_metrics = ['debug/input_images']
        if 'habitat' in cfg['env']['env_name'].lower():
            for metric in ['metrics/collisions', 'metrics/spl', 'metrics/success']:
                loggable_metrics.append(metric)
                core_metrics.append(metric)
        for meter in loggable_metrics:
            mlog.add_meter(meter, tnt.meter.ValueSummaryMeter())
        for debug_meter in debug_metrics:
            mlog.add_meter(debug_meter, tnt.meter.SingletonMeter(), ptype='image')
        try:
            for attr in cfg['learner']['perception_network_kwargs']['extra_kwargs']['attrs_to_remember']:
                mlog.add_meter(f'diagnostics/{attr}', tnt.meter.ValueSummaryMeter(), ptype='histogram')
        except KeyError:
            pass

        mlog.add_meter('config', tnt.meter.SingletonMeter(), ptype='text')
        mlog.update_meter(cfg_to_md(cfg, uuid), meters={'config'}, phase='train')

        # File loggers
        flog = tnt.logger.FileLogger(cfg['saving']['results_log_file'], overwrite=True)
        try:
            flog_keys_to_remove = [f'diagnostics/{k}' for k in cfg['learner']['perception_network_kwargs']['extra_kwargs']['attrs_to_remember']]
        except KeyError:
            warnings.warn('Unable to find flog keys to remove')
            flog_keys_to_remove = []
        reward_only_flog = tnt.logger.FileLogger(cfg['saving']['reward_log_file'], overwrite=True)

        # replay data to mlog, move metadata file
        if changed_log_dir:
            evkit.utils.logging.replay_logs(existing_log_paths, mlog)
            evkit.utils.logging.move_metadata_file(old_log_dir, cfg['saving']['log_dir'], uuid)

        ##########
        # LEARN! #
        ##########
        if cfg['training']['cuda']:
            if not cfg['learner']['test']:
                current_train_obs = current_train_obs.cuda(device=cfg['training']['gpu_devices'][0])
            current_obs = current_obs.cuda(device=cfg['training']['gpu_devices'][0])
            # rollouts.cuda(device=cfg['training']['gpu_devices'][0])  # rollout should be on RAM
            try:
                actor_critic.cuda(device=cfg['training']['gpu_devices'][0])
            except UnboundLocalError as e:
                logger.error(f'Cannot put actor critic on cuda. Are you using a checkpoint and is it being found/initialized properly? {e}')
                raise e

        # These variables are used to compute average rewards for all processes.
        episode_rewards = torch.zeros([cfg['env']['num_processes'], 1])
        episode_lengths = torch.zeros([cfg['env']['num_processes'], 1])
        episode_tracker = evkit.utils.logging.EpisodeTracker(cfg['env']['num_processes'])
        if cfg['learner']['test']:
            all_episodes = []
            actor_critic.eval()
            try:
                actor_critic.base.perception_unit.sidetuner.attrs_to_remember = []
            except:
                pass

        # First observation
        obs = envs.reset()
        current_obs.insert(obs)
        mask_done = torch.FloatTensor([[0.0] for _ in range(cfg['env']['num_processes'])]).cuda(device=cfg['training']['gpu_devices'][0], non_blocking=True)
        states = torch.zeros(cfg['env']['num_processes'], cfg['learner']['internal_state_size']).cuda(device=cfg['training']['gpu_devices'][0], non_blocking=True)
        try:
            actor_critic.reset(envs=envs)
        except:
            actor_critic.reset()

        # Main loop
        start_time = time.time()
        n_episodes_completed = 0
        num_updates = int(cfg['training']['num_frames']) // (cfg['learner']['num_steps'] * cfg['env']['num_processes'])
        if cfg['learner']['test']:
            logger.info(f"Running {cfg['learner']['test_k_episodes']}")
        else:
            logger.info(f"Running until num updates == {num_updates}")
        for j in range(start_epoch, num_updates, 1):
            for step in range(cfg['learner']['num_steps']):
                obs_unpacked = {k: current_obs.peek()[k].peek() for k in current_obs.peek()}
                if j == start_epoch and step < 10:
                    log_input_images(obs_unpacked, mlog, num_stack=cfg['learner']['num_stack'],
                                     key_names=['rgb_filled', 'map'], meter_name='debug/input_images', step_num=step)

                # Sample actions
                with torch.no_grad():
                    # value, action, action_log_prob, states = actor_critic.act(
                    #     {k:v.cuda(device=cfg['training']['gpu_devices'][0]) for k, v in obs_unpacked.items()},
                    #     states.cuda(device=cfg['training']['gpu_devices'][0]),
                    #     mask_done.cuda(device=cfg['training']['gpu_devices'][0]))
                    # All should already be on training.gpu_devices[0]
                    value, action, action_log_prob, states = actor_critic.act(
                        obs_unpacked, states, mask_done, cfg['learner']['deterministic'])
                cpu_actions = list(action.squeeze(1).cpu().numpy())
                obs, reward, done, info = envs.step(cpu_actions)
                mask_done_cpu = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                mask_done = mask_done_cpu.cuda(device=cfg['training']['gpu_devices'][0], non_blocking=True)
                reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
                episode_tracker.append(obs, cpu_actions)

                # log diagnostics
                if cfg['learner']['test']:
                    try:
                        mlog.update_meter(actor_critic.perplexity.cpu(), meters={'diagnostics/dist_perplexity'}, phase='val')
                        mlog.update_meter(actor_critic.entropy.cpu(), meters={'losses/dist_entropy'}, phase='val')
                        mlog.update_meter(value.cpu(), meters={'diagnostics/value'}, phase='val')
                    except AttributeError:
                        pass


                # Handle terminated episodes; logging values and computing the "done" mask
                episode_rewards += reward
                episode_lengths += (1 + cfg['env']['additional_repeat_count'])
                for i, (r, l, done_) in enumerate(zip(episode_rewards, episode_lengths, done)):  # Logging loop
                    if done_:
                        n_episodes_completed += 1
                        if cfg['learner']['test']:
                            info[i]['reward'] = r.item()
                            info[i]['length'] = l.item()
                            if 'debug_mode' in cfg['env']['env_specific_kwargs'] and cfg['env']['env_specific_kwargs']['debug_mode']:
                                info[i]['scene_id'] = envs.env.env.env._env.current_episode.scene_id
                                info[i]['episode_id'] = envs.env.env.env._env.current_episode.episode_id
                            all_episodes.append({
                                'info': info[i],
                                'history': episode_tracker.episodes[i][:-1]})
                        episode_tracker.clear_episode(i)
                        phase = 'train' if i < num_train_processes else 'val'
                        mlog.update_meter(r.item(), meters={'metrics/rewards'}, phase=phase)
                        mlog.update_meter(l.item(), meters={'diagnostics/lengths'}, phase=phase)
                        if 'habitat' in cfg['env']['env_name'].lower():
                            mlog.update_meter(info[i]["collisions"], meters={'metrics/collisions'}, phase=phase)
                            if scenario == 'PointNav':
                                mlog.update_meter(info[i]["spl"], meters={'metrics/spl'}, phase=phase)
                                mlog.update_meter(info[i]["success"], meters={'metrics/success'}, phase=phase)

                        # reset env then agent... note this only works for single process
                        if 'debug_mode' in cfg['env']['env_specific_kwargs'] and cfg['env']['env_specific_kwargs']['debug_mode']:
                            obs = envs.reset()
                        try:
                            actor_critic.reset(envs=envs)
                        except:
                            actor_critic.reset()
                episode_rewards *= mask_done_cpu
                episode_lengths *= mask_done_cpu

                # Insert the new observation into RolloutStorage
                current_obs.insert(obs, mask_done)
                if not cfg['learner']['test']:
                    for k in obs:
                        if k in current_train_obs.sensor_names:
                            current_train_obs[k].insert(obs[k][:num_train_processes], mask_done[:num_train_processes])
                    rollouts.insert(current_train_obs.peek(),
                                    states[:num_train_processes],
                                    action[:num_train_processes],
                                    action_log_prob[:num_train_processes],
                                    value[:num_train_processes],
                                    reward[:num_train_processes],
                                    mask_done[:num_train_processes])
                    mlog.update_meter(value[:num_train_processes].mean().item(), meters={'diagnostics/value'},
                                      phase='train')

            # Training update
            if not cfg['learner']['test']:
                if not cfg['learner']['use_replay']:
                    # Moderate compute saving optimization (if no replay buffer):
                    #     Estimate future-discounted returns only once
                    with torch.no_grad():
                        next_value = actor_critic.get_value(rollouts.observations.at(-1),
                                                            rollouts.states[-1],
                                                            rollouts.masks[-1]).detach()
                    rollouts.compute_returns(next_value, cfg['learner']['use_gae'], cfg['learner']['gamma'],
                                             cfg['learner']['tau'])
                value_loss, action_loss, dist_entropy, max_importance_weight, info = agent.update(rollouts)
                rollouts.after_update()  # For the next iter: initial obs <- current observation

                # Update meters with latest training info
                mlog.update_meter(dist_entropy, meters={'losses/dist_entropy'})
                mlog.update_meter(np.exp(dist_entropy), meters={'diagnostics/dist_perplexity'})
                mlog.update_meter(value_loss, meters={'losses/value_loss'})
                mlog.update_meter(action_loss, meters={'losses/action_loss'})
                mlog.update_meter(max_importance_weight, meters={'diagnostics/max_importance_weight'})
                if 'intrinsic_loss_types' in cfg['learner']['loss_kwargs'] and len(cfg['learner']['loss_kwargs']['intrinsic_loss_types']) > 0:
                    for iloss in cfg['learner']['loss_kwargs']['intrinsic_loss_types']:
                        mlog.update_meter(info[iloss], meters={f'losses/{iloss}'})
                try:
                    for attr in cfg['learner']['perception_network_kwargs']['extra_kwargs']['attrs_to_remember']:
                        mlog.update_meter(info[attr].cpu(), meters={f'diagnostics/{attr}'})
                except KeyError:
                    pass

                try:
                    if hasattr(actor_critic, 'module'):
                        alpha = [param for name, param in actor_critic.module.named_parameters() if 'alpha' in name][0]
                    else:
                        alpha = [param for name, param in actor_critic.named_parameters() if 'alpha' in name][0]
                    mlog.update_meter(torch.sigmoid(alpha).detach().item(), meters={f'introspect/alpha'})
                except IndexError:
                    pass

            # Main logging
            if (j) % cfg['saving']['log_interval'] == 0:
                torch.cuda.empty_cache()
                GPUtil.showUtilization()
                count_open()
                num_relevant_processes = num_val_processes if cfg['learner']['test'] else num_train_processes
                n_steps_since_logging = cfg['saving']['log_interval'] * num_relevant_processes * cfg['learner'][
                    'num_steps']
                total_num_steps = (j + 1) * num_relevant_processes * cfg['learner']['num_steps']

                logger.info("Update {}, num timesteps {}, FPS {}".format(
                    j + 1,
                    total_num_steps,
                    int(n_steps_since_logging / (time.time() - start_time))
                ))
                logger.info(f"Completed episodes: {n_episodes_completed}")
                viable_modes = ['val'] if cfg['learner']['test'] else ['train', 'val']
                for metric in core_metrics:  # Log to stdout
                    for mode in viable_modes:
                        if metric in core_metrics or mode == 'train':
                            mlog.print_meter(mode, total_num_steps, meterlist={metric})
                if not cfg['learner']['test']:
                    for mode in viable_modes:  # Log to files
                        results = mlog.peek_meter(phase=mode)
                        reward_only_flog.log(mode, {metric: results[metric] for metric in core_metrics})
                        if mode == 'train':
                            results_to_log = {}
                            results['step_num'] = j + 1
                            results_to_log['step_num'] = results['step_num']
                            for k,v in results.items():
                                if k in flog_keys_to_remove:
                                    warnings.warn(f'Removing {k} from results_log.pkl due to large size')
                                else:
                                    results_to_log[k] = v
                            flog.log('all_results', results_to_log)

                        mlog.reset_meter(total_num_steps, mode=mode)
                start_time = time.time()

            # Save checkpoint
            if not cfg['learner']['test'] and j % cfg['saving']['save_interval'] == 0:
                save_dir_absolute = os.path.join(cfg['saving']['log_dir'], cfg['saving']['save_dir'])
                save_checkpoint(
                    {'agent': agent, 'epoch': j},
                    save_dir_absolute, j)
            if 'test_k_episodes' in cfg['learner'] and n_episodes_completed >= cfg['learner']['test_k_episodes']:
                torch.save(all_episodes, os.path.join(cfg['saving']['log_dir'], 'validation.pth'))
                all_episodes = all_episodes[:cfg['learner']['test_k_episodes']]
                spl_mean = np.mean([episode['info']['spl'] for episode in all_episodes])
                success_mean = np.mean([episode['info']['success'] for episode in all_episodes])
                reward_mean = np.mean([episode['info']['reward'] for episode in all_episodes])
                logger.info('------------ done with testing -------------')
                logger.info(f'SPL: {spl_mean} --- Success: {success_mean} --- Reward: {reward_mean}')
                for metric in mlog.meter['val'].keys():
                    mlog.print_meter('val', -1, meterlist={metric})
                break

    # Clean up (either after ending normally or early [e.g. from a KeyboardInterrupt])
    finally:
        print(psutil.virtual_memory())
        GPUtil.showUtilization(all=True)
        try:
            logger.info("### Done - Killing envs.")
            if isinstance(envs, list):
                [env.close() for env in envs]
            else:
                envs.close()
            logger.info("Killed envs.")
        except UnboundLocalError:
            logger.info("No envs to kill!")


if is_interactive() and __name__ == '__main__':
    assert LOG_DIR, 'log dir cannot be empty'
    os.makedirs(LOG_DIR, exist_ok=True)
    subprocess.call("rm -rf {}/*".format(LOG_DIR), shell=True)
    ex.observers.append(FileStorageObserverWithExUuid.create(LOG_DIR))
    ex.run_commandline(
        'run_config with \
            uuid="gibson_random" \
            cfg.env.num_processes=1\
            '.format())
elif __name__ == '__main__':
    assert LOG_DIR, 'log dir cannot be empty'

    # Manually parse command line opts
    short_usage, usage, internal_usage = ex.get_usage()
    args = docopt(internal_usage, [str(a) for a in sys.argv[1:]], help=False)
    config_updates, named_configs = get_config_updates(args['UPDATE'])

    ex.run('prologue', config_updates, named_configs, options=args)
    ex.observers.append(FileStorageObserverWithExUuid.create(LOG_DIR))
    try:
        ex.run_commandline()
    except FileNotFoundError as e:
        logger.error(f'File not found! Are you trying to test an experiment with the uuid: {e}?')
        raise e
else:
    logger.info(__name__)