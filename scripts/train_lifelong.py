import os
import GPUtil

# If you need one GPU, I will pick it here for you
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    gpu = [str(g) for g in GPUtil.getAvailable(maxMemory=0.2)]
    assert len(gpu) > 0, 'No available GPUs'
    print('Using GPU', ','.join(gpu))
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu)

import argparse
import copy
from docopt import docopt
import functools
import json
import logging
import math
import numpy as np
import pprint
import psutil
import random
import runpy
from sacred.arg_parser import get_config_updates
from sacred import Experiment
import subprocess
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
import torchvision.utils as tvutils
import torch.nn.functional as F
import torchsummary
from tqdm import tqdm as tqdm
from multiprocessing.pool import ThreadPool
import threading
import warnings

from tlkit.data.synset import synset_arr
from tlkit.models.ewc import EWC
from tlkit.models.student_models import FCN3, FCN4, FCN5, FCN8
from tlkit.models.lifelong_framework import load_submodule
from tlkit.logging_helpers import log, log_image, reset_log, add_classification_specific_logging, get_logger, write_logs
from tlkit.utils import update, var_to_numpy, index_to_image, load_state_dict_from_path
import tlkit.utils
import tlkit.data.datasets.taskonomy_dataset as taskonomy_dataset
import tlkit.data.datasets.fashion_mnist_dataset as fashion_mnist_dataset
import tlkit.data.datasets.imagenet_dataset as imagenet_dataset
import tlkit.data.datasets.icifar_dataset as icifar_dataset
import tlkit.data.splits as splits
from tlkit.utils import LIST_OF_TASKS, TASKS_TO_CHANNELS, SINGLE_IMAGE_TASKS

from evkit.saving.observers import FileStorageObserverWithExUuid
import evkit.saving.checkpoints as checkpoints
from evkit.utils.profiler import Profiler
from evkit.utils.random import set_seed
from evkit.utils.misc import cfg_to_md, count_trainable_parameters, count_total_parameters, search_and_replace_dict
from evkit.utils.parallel import _CustomDataParallel
from evkit.utils.losses import heteroscedastic_double_exponential, heteroscedastic_normal, weighted_mse_loss, softmax_cross_entropy, weighted_l1_loss, perceptual_l1_loss, perceptual_l2_loss, perceptual_cross_entropy_loss, identity_regularizer, transfer_regularizer, perceptual_regularizer, dense_cross_entropy, dense_softmax_cross_entropy, weighted_l2_loss
from evkit.utils.viz.core import pack_images, imagenet_unnormalize

from evkit.models.taskonomy_network import TaskonomyEncoder, TaskonomyDecoder, TaskonomyNetwork
from evkit.models.unet import UNet, UNetHeteroscedasticFull, UNetHeteroscedasticIndep, UNetHeteroscedasticPooled
from tlkit.models.student_models import FCN4Reshaped
from tlkit.models.resnet_cifar import ResnetiCifar44
from tlkit.models.sidetune_architecture import GenericSidetuneNetwork, TransferConv3, PreTransferedDecoder
from tlkit.models.models_additional import BoostedNetwork, ConstantModel
from tlkit.models.lifelong_framework import LifelongSidetuneNetwork


import tnt.torchnet as tnt
from tnt.torchnet.logger import FileLogger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger()

ex = Experiment(name="Train Lifelong Learning agent")
LOG_DIR = sys.argv[1]
sys.argv.pop(1)
runpy.run_module('configs.vision_lifelong', init_globals=globals())
runpy.run_module('configs.icifar_cfg', init_globals=globals())
runpy.run_module('configs.seq_taskonomy_cfg', init_globals=globals())
runpy.run_module('configs.seq_taskonomy_cfg_extra', init_globals=globals())
runpy.run_module('configs.shared', init_globals=globals())

@ex.command
def prologue(cfg, uuid):
    os.makedirs(LOG_DIR, exist_ok=True)
    assert not (cfg['saving']['obliterate_logs'] and cfg['training']['resume_training']), 'Cannot obliterate logs and resume training'
    if cfg['saving']['obliterate_logs']:
        assert LOG_DIR, 'LOG_DIR cannot be empty'
        subprocess.call(f'rm -rf {LOG_DIR}', shell=True)
    if cfg['training']['resume_training']:
        checkpoints.archive_current_run(LOG_DIR, uuid)


@ex.main
def train(cfg, uuid):
    set_seed(cfg['training']['seed'])

    ############################################################
    # Logger
    ############################################################
    logger.setLevel(logging.INFO)
    logger.info(pprint.pformat(cfg))
    logger.debug(f'Loaded Torch version: {torch.__version__}')
    logger.debug(f'Using device: {device}')
    logger.info(f"Training following tasks: ")
    for i, (s, t) in enumerate(zip(cfg['training']['sources'], cfg['training']['targets'])):
        logger.info(f"\tTask {i}: {s} -> {t}")
    logger.debug(f'Starting data loaders')


    ############################################################
    # Model (and possibly resume from checkpoint)
    ############################################################
    logger.debug(f'Setting up model')
    search_and_replace_dict(cfg['learner']['model_kwargs'], cfg['training']['targets'][0][0])  # switches to the proper pretrained encoder
    model = eval(cfg['learner']['model'])(**cfg['learner']['model_kwargs'])
    logger.info(f"Created model. Number of trainable parameters: {count_trainable_parameters(model)}. Number of total parameters: {count_total_parameters(model)}")
    try:
        logger.info(f"Number of trainable transfer parameters: {count_trainable_parameters(model.transfers)}. Number of total transfer parameters: {count_total_parameters(model.transfers)}")
        if isinstance(model.encoder, nn.Module):
            logger.info(f"Number of trainable encoder parameters: {count_trainable_parameters(model.base)}. Number of total encoder parameters: {count_total_parameters(model.base)}")
        if isinstance(model.side_networks, nn.Module):
            logger.info(f"Number of trainable side parameters: {count_trainable_parameters(model.sides)}. Number of total side parameters: {count_total_parameters(model.sides)}")
        if isinstance(model.merge_operators, nn.Module):
            logger.info(f"Number of trainable merge (alpha) parameters: {count_trainable_parameters(model.merge_operators)}. Number of total merge (alpha) parameters: {count_total_parameters(model.merge_operators)}")
    except:
        pass

    ckpt_fpath = cfg['training']['resume_from_checkpoint_path']
    loaded_optimizer = None
    start_epoch = 0

    if ckpt_fpath is not None and not cfg['training']['resume_training']:
        warnings.warn('Checkpoint path provided but resume_training is set to False, are you sure??')
    if ckpt_fpath is not None and cfg['training']['resume_training']:
        if not os.path.exists(ckpt_fpath):
            logger.warning(f'Trying to resume training, but checkpoint path {ckpt_fpath} does not exist. Starting training from beginning...')
        else:
            model, checkpoint = load_state_dict_from_path(model, ckpt_fpath)
            start_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
            logger.info(f"Loaded model (epoch {start_epoch if 'epoch' in checkpoint else 'unknown'}) from {ckpt_fpath}")
            if 'optimizer' in checkpoint:
                loaded_optimizer = checkpoint['optimizer']
            else:
                warnings.warn('No optimizer in checkpoint, are you sure?')
            try:  # we do not use state_dict, do not let it take up precious CUDA memory
                del checkpoint['state_dict']
            except KeyError:
                pass

    model.to(device)
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs!")
        assert cfg['learner']['model'] != 'ConstantModel', 'ConstantModel (blind) does not operate with multiple devices'
        model = nn.DataParallel(model, range(torch.cuda.device_count()))
        model.to(device)

    ############################################################
    # Data Loading
    ############################################################
    for key in ['sources', 'targets', 'masks']:
        cfg['training']['dataloader_fn_kwargs'][key] = cfg['training'][key]

    dataloaders = eval(cfg['training']['dataloader_fn'])(**cfg['training']['dataloader_fn_kwargs'])
    if cfg['training']['resume_training']:
        if 'curr_iter_idx' in checkpoint and checkpoint['curr_iter_idx'] == -1:
            warnings.warn(f'curr_iter_idx is -1, Guessing curr_iter_idx to be start_epoch {start_epoch}')
            dataloaders['train'].start_dl = start_epoch
        elif 'curr_iter_idx' in checkpoint:
            logger.info(f"Starting dataloader at {checkpoint['curr_iter_idx']}")
            dataloaders['train'].start_dl = checkpoint['curr_iter_idx']
        else:
            warnings.warn(f'Guessing curr_iter_idx to be start_epoch {start_epoch}')
            dataloaders['train'].start_dl = start_epoch

    ############################################################
    # Loss Functions
    ############################################################
    loss_fn_lst = cfg['training']['loss_fn']
    loss_kwargs_lst = cfg['training']['loss_kwargs']
    if not isinstance(loss_fn_lst, list):
        loss_fn_lst = [ loss_fn_lst ]
        loss_kwargs_lst = [ loss_kwargs_lst ]
    elif isinstance(loss_kwargs_lst, dict):
        loss_kwargs_lst = [loss_kwargs_lst for _ in range(len(loss_fn_lst))]

    loss_fns = []
    assert len(loss_fn_lst) == len(loss_kwargs_lst), 'number of loss fn/kwargs not the same'
    for loss_fn, loss_kwargs in zip(loss_fn_lst, loss_kwargs_lst):
        if loss_fn == 'perceptual_l1':
            loss_fn = perceptual_l1_loss(cfg['training']['loss_kwargs']['decoder_path'], cfg['training']['loss_kwargs']['bake_decodings'])
        elif loss_fn == 'perceptual_l2':
            loss_fn = perceptual_l2_loss(cfg['training']['loss_kwargs']['decoder_path'], cfg['training']['loss_kwargs']['bake_decodings'])
        elif loss_fn == 'perceptual_cross_entropy':
            loss_fn = perceptual_cross_entropy_loss(cfg['training']['loss_kwargs']['decoder_path'], cfg['training']['loss_kwargs']['bake_decodings'])
        else:
            loss_fn = functools.partial(eval(loss_fn), **loss_kwargs)
        loss_fns.append(loss_fn)

    if len(loss_fns) == 1 and len(cfg['training']['sources']) > 1:
        loss_fns = [loss_fns[0] for _ in range(len(cfg['training']['sources']))]

    if 'regularizer_fn' in cfg['training'] and cfg['training']['regularizer_fn'] is not None:
        assert torch.cuda.device_count() <= 1, 'Regularization does not support multi GPU, unable to access model attributes from DataParallel wrapper'
        bare_model = model.module if torch.cuda.device_count() > 1 else model
        loss_fns = [eval(cfg['training']['regularizer_fn'])(loss_fn=loss_fn, model=bare_model, **cfg['training']['regularizer_kwargs']) for loss_fn in loss_fns]

    ############################################################
    # More Logging
    ############################################################
    flog = tnt.logger.FileLogger(cfg['saving']['results_log_file'], overwrite=True)
    mlog = get_logger(cfg, uuid)
    mlog.add_meter('config', tnt.meter.SingletonMeter(), ptype='text')
    mlog.update_meter(cfg_to_md(cfg, uuid), meters={'config'}, phase='train')
    for task, _ in enumerate(cfg['training']['targets']):
        mlog.add_meter(f'alpha/task_{task}', tnt.meter.ValueSummaryMeter())
        mlog.add_meter(f'output/task_{task}', tnt.meter.ValueSummaryMeter(), ptype='image')
        mlog.add_meter(f'input/task_{task}', tnt.meter.ValueSummaryMeter(), ptype='image')
        mlog.add_meter('weight_histogram/task_{task}', tnt.meter.ValueSummaryMeter(), ptype='histogram')
        for loss in cfg['training']['loss_list']:
            mlog.add_meter(f'losses/{loss}_{task}', tnt.meter.ValueSummaryMeter())

        if cfg['training']['task_is_classification'][task] :
            mlog.add_meter(f'accuracy_top1/task_{task}', tnt.meter.ClassErrorMeter(topk=[1], accuracy=True))
            mlog.add_meter(f'accuracy_top5/task_{task}', tnt.meter.ClassErrorMeter(topk=[5], accuracy=True))
            mlog.add_meter(f'perplexity_pred/task_{task}', tnt.meter.ValueSummaryMeter())
            mlog.add_meter(f'perplexity_label/task_{task}', tnt.meter.ValueSummaryMeter())


    ############################################################
    # Training
    ############################################################
    try:
        if cfg['training']['train']:
            # Optimizer
            if cfg['training']['resume_training'] and loaded_optimizer is not None:
                optimizer = loaded_optimizer
            else:
                optimizer = eval(cfg['learner']['optimizer_class'])(
                    [
                        {'params': [param for name, param in model.named_parameters() if 'merge_operator' in name or 'context' in name or 'alpha' in name], 'weight_decay': 0.0},
                        {'params': [param for name, param in model.named_parameters() if 'merge_operator' not in name and 'context' not in name and 'alpha' not in name]},
                    ],
                    lr=cfg['learner']['lr'], **cfg['learner']['optimizer_kwargs']
                )

            # Scheduler
            scheduler = None
            if cfg['learner']['lr_scheduler_method'] is not None:
                scheduler = eval(cfg['learner']['lr_scheduler_method'])(optimizer, **cfg['learner']['lr_scheduler_method_kwargs'])

            model.start_training()  # For PSP variant

            # Mixed precision training
            if cfg['training']['amp']:
                from apex import amp
                model, optimizer = amp.initialize(model, optimizer, opt_level='O1')


            logger.info("Starting training...")
            context = train_model(cfg, model, dataloaders, loss_fns, optimizer, start_epoch=start_epoch,
                                      num_epochs=cfg['training']['num_epochs'], save_epochs=cfg['saving']['save_interval'],
                                      scheduler=scheduler, mlog=mlog, flog=flog)
    finally:
        print(psutil.virtual_memory())
        GPUtil.showUtilization(all=True)

    ####################
    # Final Test
    ####################
    if cfg['training']['test']:
        run_kwargs = {
            'cfg': cfg,
            'mlog': mlog,
            'flog': flog,
            'optimizer': None,
            'loss_fns': loss_fns,
            'model': model,
            'use_thread': cfg['saving']['in_background'],
        }
        context, _ = run_one_epoch(dataloader=dataloaders['val'], epoch=0, train=False, **run_kwargs)
    
    logger.info('Waiting up to 10 minutes for all files to save...')
    mlog.flush()
    [c.join(600) for c in context]
    logger.info('All saving is finished.')


def train_model(cfg, model, dataloaders, loss_fns, optimizer, start_epoch=0, num_epochs=250, save_epochs=25, scheduler=None, mlog=None, flog=None):
    '''
        Main training loop. Multiple tasks might happen in the same epoch. 
        0 to 1     random validation only
        1 to 2     train task 0 labeled as epoch 2, validate all
        i to {i+1} train task {i-1} labeled as epoch {i+1}
    '''
    checkpoint_dir = os.path.join(cfg['saving']['log_dir'], cfg['saving']['save_dir'])
    run_kwargs = {
        'cfg': cfg,
        'mlog': mlog,
        'flog': flog,
        'optimizer': optimizer,
        'loss_fns': loss_fns,
        'model': model,
        'use_thread': cfg['saving']['in_background'],
    }
    context = []
    log_interval = cfg['saving']['log_interval']
    log_interval = int(log_interval) if log_interval > 1 else log_interval
    end_epoch = start_epoch + num_epochs
    print(f'training for {num_epochs} epochs')

    for epoch in range(start_epoch, end_epoch):
        # tlkit.utils.count_open()  # Turn on to check for memory leak
        torch.cuda.empty_cache()

        if epoch == 0 or epoch % save_epochs == save_epochs - 1:
            context += save_checkpoint(model, optimizer, epoch, dataloaders, checkpoint_dir, use_thread=cfg['saving']['in_background'])

        should_run_validation = (epoch == 0) or (log_interval <= 1) or ((epoch % log_interval) == (log_interval - 1))
        if should_run_validation:
            assert math.isnan(mlog.peek_meter()['losses/total_0']), 'Loggers are not empty at the beginning of evaluation. Were training logs cleared?'
            context1, loss_dict = run_one_epoch(dataloader=dataloaders['val'], epoch=epoch, train=False, **run_kwargs)
            context += context1

        if scheduler is not None:
            try:
                scheduler.step(loss_dict['total'])
            except:
                scheduler.step()

        # training starts logging at epoch 1, val epoch 0 is fully random, each task should only last ONE epoch
        context1, _ = run_one_epoch(dataloader=dataloaders['train'], epoch=epoch+1, train=True, **run_kwargs)
        context += context1

        # Compute needed after the end of an epoch -  e.g. EWC computes ~Fisher info matrix
        post_training_epoch(dataloader=dataloaders['train'], epoch=epoch, **run_kwargs)


    context1, _ = run_one_epoch(dataloader=dataloaders['val'], epoch=end_epoch, train=False, **run_kwargs)
    context += context1
    context += save_checkpoint(model, optimizer, end_epoch, dataloaders, checkpoint_dir, use_thread=cfg['saving']['in_background'])
    return context


def post_training_epoch(dataloader=None, epoch=-1, model=None, loss_fns=None, **kwargs):
    post_training_cache = {}

    if hasattr(loss_fns[dataloader.curr_iter_idx], 'post_training_epoch'):  # this lets respective loss_fn compute F
        loss_fns[dataloader.curr_iter_idx].post_training_epoch(model, dataloader, post_training_cache, **kwargs)

    for i, loss_fn in enumerate(loss_fns):
        if hasattr(loss_fn, 'post_training_epoch') and i != dataloader.curr_iter_idx:
            loss_fn.post_training_epoch(model, dataloader, post_training_cache, **kwargs)


def run_one_epoch(model: LifelongSidetuneNetwork, dataloader, loss_fns, optimizer, epoch, cfg, mlog, flog, train=True, use_thread=False)->(list,dict):
    # logs through the progress of the epoch from [epoch, epoch + 1)
    start_time = time.time()
    model.train(train)
    params_with_grad = model.parameters()
    phase = 'train' if train else 'val'
    sources = cfg['training']['sources']
    targets = cfg['training']['targets']
    tasks = [t for t in SINGLE_IMAGE_TASKS if len([tt for tt in cfg['training']['targets'] if t in tt]) > 0]
    cache = {'phase': phase, 'sources': sources, 'targets': targets, 'tasks': tasks}
    context = []
    losses = {x:[] for x in cfg['training']['loss_list']}

    log_steps = []
    log_interval = cfg['saving']['log_interval']
    log_interval = int(log_interval) if log_interval >= 1 else log_interval
    if log_interval < 1 and train:
        num_logs_per_epoch = int(1 // log_interval)
        log_steps = [i * int(len(dataloader)/num_logs_per_epoch) for i in range(1, num_logs_per_epoch)]

    if cfg['training']['post_aggregation_transform_fn'] is not None:
        post_agg_transform = eval(cfg['training']['post_aggregation_transform_fn'])

    if cfg['learner']['use_feedback']:
        num_passes = cfg['learner']['feedback_kwargs']['num_feedback_iter']
        backward_kwargs = {'retain_graph': True}
    else:
        num_passes = 1
        backward_kwargs = {}

    if isinstance(model, _CustomDataParallel):
        warnings.warn('DataParallel does not allow you to put part of the model on CPU')
        model.cuda()

    with torch.set_grad_enabled(train):
        # print(type(model.encoder.encoder), torch.norm(next(model.encoder.encoder.parameters())))
        # print(type(model.encoder.side_network), torch.norm(next(model.encoder.side_network.parameters())))
        seen = set()
        for i, (task_idx, batch_tuple) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch} ({phase})")):
            if cfg['training']['post_aggregation_transform_fn'] is not None:
                batch_tuple = post_agg_transform(batch_tuple, **cfg['training']['post_aggregation_transform_fn_kwargs'])

            # Determine and handle new task
            old_size = len(seen)
            seen.add(task_idx)
            if len(seen) > old_size:
                logger.info(f"Moving to task: {task_idx}")
                model.start_task(task_idx, train, print_alpha=True)

            # Decompose batch, Forward, Compute Loss
            x, label, masks = tlkit.utils.process_batch_tuple(batch_tuple, task_idx, cfg)

            for pass_i in range(num_passes):
                prediction = model(x, task_idx=task_idx, pass_i=pass_i)
                loss_dict = loss_fns[task_idx](prediction, label, masks, cache)

                # If training, Backward
                if train:
                    optimizer.zero_grad()
                    loss_dict['total'].backward(**backward_kwargs)
                    if cfg['learner']['max_grad_norm'] is not None:
                        torch.nn.utils.clip_grad_norm_(params_with_grad, cfg['learner']['max_grad_norm'])
                    optimizer.step()

            # Logging
            mlog.update_meter(model.merge_operator.param, meters={f'alpha/task_{task_idx}'}, phase=phase)
            for loss in cfg['training']['loss_list']:
                assert loss in loss_dict.keys(), f'Promised to report loss {loss}, but missing from loss_dict'
                mlog.update_meter(loss_dict[loss].detach().item(), meters={f'losses/{loss}_{task_idx}'}, phase=phase)

            if cfg['training']['task_is_classification'][task_idx]:
                add_classification_specific_logging(cache, mlog, task_idx, phase)

            if len(seen) > old_size:
                log_image(mlog, task_idx, cfg, x, label, prediction, masks=masks, cache=cache)

            #  for super long epochs where we want some information between epochs
            if i in log_steps:
                step = epoch + i / len(dataloader)
                step = int(np.floor(step * cfg['saving']['ticks_per_epoch']))
                for loss in cfg['training']['loss_list']:
                    losses[loss].append(mlog.peek_meter(phase=phase)[f'losses/{loss}_{task_idx}'].item())
                context += write_logs(mlog, flog, task_idx, step, cfg, cache, to_print=False)

    for loss in cfg['training']['loss_list']:
        losses[loss].append(mlog.peek_meter(phase=phase)[f'losses/{loss}_{task_idx}'].item())

    if log_interval <= 1 or epoch % log_interval == log_interval - 1 or epoch == 0:
        step = epoch + (len(dataloader) - 1) / len(dataloader)
        step = int(np.floor(step * cfg['saving']['ticks_per_epoch']))
        context += write_logs(mlog, flog, task_idx, step, cfg, cache, to_print=True)

    assert len(losses['total']) > 0, 'Need to report loss'
    for k in losses.keys():
        losses[k] = sum(losses[k]) / len(losses[k])

    loss_str = ''.join([' | ' + k + ' loss: {0:.6f} '.format(v) for k, v in losses.items()])
    duration = int(time.time() - start_time)
    logger.info(f'End of epoch {epoch} ({phase}) ({duration//60}m {duration%60}s) {loss_str}')  # this is cumulative from previous train epochs in the same log_interval
    return context, losses


def save_checkpoint(model, optimizer, epoch, dataloaders, checkpoint_dir, use_thread=False):
    dict_to_save = {
        'state_dict': model.state_dict(),
        'epoch': epoch,
        'model': model,
        'optimizer': optimizer,
        'curr_iter_idx': dataloaders['train'].curr_iter_idx,
    }
    checkpoints.save_checkpoint(dict_to_save, checkpoint_dir, epoch)
    return []

if __name__ == '__main__':
    assert LOG_DIR, 'log dir cannot be empty'

    # Manually parse command line opts
    short_usage, usage, internal_usage = ex.get_usage()
    args = docopt(internal_usage, [str(a) for a in sys.argv[1:]], help=False)
    config_updates, named_configs = get_config_updates(args['UPDATE'])

    ex.run('prologue', config_updates, named_configs, options=args)
    ex.observers.append(FileStorageObserverWithExUuid.create(LOG_DIR))
    ex.run_commandline()
else:
    print(__name__)
    
