import os
import GPUtil

# If you need one GPU, I will pick it here for you
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    gpu = [str(g) for g in GPUtil.getAvailable(maxMemory=0.2, order='random')]
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
import os
from PIL import Image
import pprint
import random
import runpy
from sacred.arg_parser import get_config_updates
from sacred import Experiment
from sklearn.metrics import confusion_matrix
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
from torchsummary import summary
from tqdm import tqdm as tqdm
from multiprocessing.pool import ThreadPool
import threading
import warnings

from tlkit.data.synset import synset_arr
# from tlkit.models.taskonomydecoder import TaskonomyDecoder, TaskonomyNetwork
from tlkit.models.student_models import FCN3, FCN5, FCN8
from tlkit.models.lifelong_framework import load_submodule
from tlkit.utils import update, var_to_numpy, save_checkpoint, index_to_image #, reset_log
from tlkit.logging_helpers import log, reset_log, get_logger, add_classification_specific_logging, add_imitation_specific_logging, multidim_apply, write_logs
import tlkit.utils
import tlkit.data.datasets.taskonomy_dataset as taskonomy_dataset
import tlkit.data.datasets.fashion_mnist_dataset as fashion_mnist_dataset
import tlkit.data.datasets.imagenet_dataset as imagenet_dataset
import tlkit.data.datasets.icifar_dataset as icifar_dataset
import tlkit.data.datasets.expert_dataset as expert_dataset
import tlkit.data.splits as splits
from tlkit.utils import LIST_OF_TASKS, TASKS_TO_CHANNELS, SINGLE_IMAGE_TASKS

from evkit.saving.observers import FileStorageObserverWithExUuid
import evkit.saving.checkpoints as checkpoints
from evkit.utils.profiler import Profiler
from evkit.utils.random import set_seed
from evkit.utils.misc import cfg_to_md, count_trainable_parameters
from evkit.utils.losses import heteroscedastic_double_exponential, heteroscedastic_normal, weighted_mse_loss, softmax_cross_entropy, weighted_l1_loss, perceptual_l1_loss, perceptual_l2_loss, perceptual_cross_entropy_loss, identity_regularizer, transfer_regularizer, perceptual_regularizer, dense_softmax_cross_entropy_loss, dense_softmax_cross_entropy, dense_cross_entropy
from evkit.utils.viz.core import pack_images, imagenet_unnormalize, hacky_resize, log_input_images

from evkit.rl.policy import PolicyWithBase
from evkit.models.taskonomy_network import TaskonomyEncoder, TaskonomyDecoder, TaskonomyNetwork
from evkit.models.unet import UNet, UNetHeteroscedasticFull, UNetHeteroscedasticIndep, UNetHeteroscedasticPooled
from tlkit.models.lifelong_framework import TransferConv3, GenericSidetuneNetwork
from tlkit.models.student_models import FCN5, FCN4
from tlkit.models.resnet_cifar import ResnetiCifar
from tlkit.models.models_additional import ConstantModel
from tlkit.utils import process_batch_tuple

import tnt.torchnet as tnt
from tnt.torchnet.logger import FileLogger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger()

ex = Experiment(name="Vision Transfer")
LOG_DIR = sys.argv[1]
sys.argv.pop(1)
runpy.run_module('configs.vision_transfer', init_globals=globals())
runpy.run_module('configs.imitation_learning', init_globals=globals())
runpy.run_module('configs.shared', init_globals=globals())

from sacred.arg_parser import get_config_updates

@ex.command
def prologue(cfg, uuid):
    os.makedirs(LOG_DIR, exist_ok=True)
    assert not (cfg['saving']['obliterate_logs'] and cfg['training']['resume_training']), 'cannot obliterate logs and resume training'
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

    assert len(cfg['training']['targets']) == 1, "Transferring is only supported for one target task"
    logger.info(f"Training ({ cfg['training']['sources']}) -> ({cfg['training']['targets']})")


    ############################################################
    # Verify configs are consistent - baked version needs to match un-baked version
    ############################################################
    taskonomy_sources = [src for src in cfg['training']['sources'] if 'taskonomy' in src]
    assert len(taskonomy_sources) <= 1, 'We have no way of handling multiple taskonomy features right now'
    if len(taskonomy_sources) == 1:
        # TODO refactor
        # GenericSidetuneNetwork for Vision Transfer tasks
        if 'encoder_weights_path' in cfg['learner']['model_kwargs']:
            assert cfg['learner']['model_kwargs']['encoder_weights_path'] is not None, 'if we have a taskonomy feature as a source, the model should reflect that'

        # PolicyWithBase for Imitation learning
        try:
            encoder_path = cfg['learner']['model_kwargs']['base_kwargs']['perception_unit_kwargs']['extra_kwargs']['sidetune_kwargs']['encoder_weights_path']
            assert encoder_path is not None, 'if we have a taskonomy feature as a source, the model should reflect that'
        except KeyError:
            pass

    ############################################################
    # Data Loading
    ############################################################
    logger.debug(f'Starting data loaders')
    data_subfolders = cfg['training']['sources'][:]
    if 'bake_decodings' in cfg['training']['loss_kwargs'] and cfg['training']['loss_kwargs']['bake_decodings']:
        # do not get encodings, convert encodings to decodings
        assert all(['encoding' in t for t in cfg['training']['targets']]), 'Do not bake_decodings if your target is not an encoding'
        target_decodings = [t.replace('encoding', 'decoding') for t in cfg['training']['targets']]
        data_subfolders += target_decodings
    elif not cfg['training']['suppress_target_and_use_annotator']:
        data_subfolders += cfg['training']['targets']
    else:  # use annotator
        cfg['training']['annotator'] = load_submodule(eval(cfg['training']['annotator_class']),
                                                      cfg['training']['annotator_weights_path'],
                                                      cfg['training']['annotator_kwargs']).eval()
        cfg['training']['annotator'] = cfg['training']['annotator'].to(device)

    if cfg['training']['use_masks']:
        data_subfolders += ['mask_valid']

    if cfg['training']['dataloader_fn'] is None: # Legacy support for old config type. 
        DeprecationWarning("Empty cfg.learner.dataloader_fn is deprecated and will be removed in a future version")
        logger.info(f"Using split: {cfg['training']['split_to_use']}")
        dataloaders = taskonomy_dataset.get_dataloaders(
                        cfg['training']['data_dir'],
                        data_subfolders,
                        batch_size=cfg['training']['batch_size'],
                        batch_size_val=cfg['training']['batch_size_val'],
                        zip_file_name=False,
                        train_folders=eval(cfg['training']['split_to_use'])['train'],
                        val_folders=eval(cfg['training']['split_to_use'])['val'],
                        test_folders=eval(cfg['training']['split_to_use'])['test'],
                        num_workers=cfg['training']['num_workers'],
                        load_to_mem=cfg['training']['load_to_mem'],
                        pin_memory=cfg['training']['pin_memory'])
    else:
        cfg['training']['dataloader_fn_kwargs']['tasks'] = data_subfolders
        dataloaders = eval(cfg['training']['dataloader_fn'])(**cfg['training']['dataloader_fn_kwargs'])

    ############################################################
    # Model (and possibly resume from checkpoint)
    ############################################################
    logger.debug(f'Setting up model')
    model = eval(cfg['learner']['model'])(**cfg['learner']['model_kwargs'])
    logger.info(f"Created model. Number of trainable parameters: {count_trainable_parameters(model)}.")

    loaded_optimizer = None
    start_epoch = 0
    ckpt_fpath = cfg['training']['resume_from_checkpoint_path']
    if ckpt_fpath is not None:
        if cfg['training']['resume_training'] and not os.path.exists(ckpt_fpath):
            logger.warning(f'Trying to resume training, but checkpoint path {ckpt_fpath} does not exist. Starting training from beginning...')
        else:
            checkpoint = torch.load(ckpt_fpath)
            start_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0

            state_dict = { k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items() }
            model.load_state_dict(state_dict)
            logger.info(f"Loaded model (epoch {start_epoch if 'epoch' in checkpoint else 'unknown'}) from {ckpt_fpath}")

            loaded_optimizer = checkpoint['optimizer']
            logger.info(f"Loaded optimizer (epoch {start_epoch if 'epoch' in checkpoint else 'unknown'}) from {ckpt_fpath}")

    model.to(device)
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs!")
        assert cfg['learner']['model'] != 'ConstantModel', 'ConstantModel (e.g. blind) does not operate with multiple devices'
        model = torch.nn.DataParallel(model)

    ############################################################
    # Loss Function
    ############################################################
    if cfg['training']['loss_fn'] == 'perceptual_l1':
        loss_fn = perceptual_l1_loss(cfg['training']['loss_kwargs']['decoder_path'], cfg['training']['loss_kwargs']['bake_decodings'])
    elif cfg['training']['loss_fn'] == 'perceptual_l2':
        loss_fn = perceptual_l2_loss(cfg['training']['loss_kwargs']['decoder_path'], cfg['training']['loss_kwargs']['bake_decodings'])
    elif cfg['training']['loss_fn'] == 'perceptual_cross_entropy':
        loss_fn = perceptual_cross_entropy_loss(cfg['training']['loss_kwargs']['decoder_path'], cfg['training']['loss_kwargs']['bake_decodings'])
    else:
        loss_fn = functools.partial(eval(cfg['training']['loss_fn']), **cfg['training']['loss_kwargs'])

    if 'regularizer_fn' in cfg['training'] and cfg['training']['regularizer_fn'] is not None:
        assert torch.cuda.device_count() <= 1, 'Regularization does not support multi GPU, unable to access model attributes from DataParallel wrapper'
        bare_model = model.module if torch.cuda.device_count() > 1 else model
        loss_fn = eval(cfg['training']['regularizer_fn'])(loss_fn=loss_fn, model=bare_model, **cfg['training']['regularizer_kwargs'])

    ############################################################
    # Logging
    ############################################################
    flog = tnt.logger.FileLogger(cfg['saving']['results_log_file'], overwrite=True)
    mlog = get_logger(cfg, uuid)
    mlog.add_meter('config', tnt.meter.SingletonMeter(), ptype='text')
    mlog.update_meter(cfg_to_md(cfg, uuid), meters={'config'}, phase='train')
    mlog.add_meter('input_image', tnt.meter.ValueSummaryMeter(), ptype='image')
    mlog.add_meter('decoded_image', tnt.meter.ValueSummaryMeter(), ptype='image')
    mlog.add_meter(f'introspect/alpha', tnt.meter.ValueSummaryMeter())
    for loss in cfg['training']['loss_list']:
        mlog.add_meter(f'losses/{loss}', tnt.meter.ValueSummaryMeter())

    # Add Classification logs
    tasks = [t for t in SINGLE_IMAGE_TASKS if len([tt for tt in cfg['training']['targets'] if t in tt]) > 0]
    if 'class_object' in tasks or 'class_scene' in tasks:
        mlog.add_meter('accuracy_top1', tnt.meter.ClassErrorMeter(topk=[1], accuracy=True))
        mlog.add_meter('accuracy_top5', tnt.meter.ClassErrorMeter(topk=[5], accuracy=True))
        mlog.add_meter('perplexity_pred', tnt.meter.ValueSummaryMeter())
        mlog.add_meter('perplexity_label', tnt.meter.ValueSummaryMeter())
        mlog.add_meter('diagnostics/class_histogram', tnt.meter.ValueSummaryMeter(), ptype='histogram')
        mlog.add_meter('diagnostics/confusion_matrix', tnt.meter.ValueSummaryMeter(), ptype='image')

    # Add Imitation Learning logs
    if cfg['training']['targets'][0] == 'action':
        mlog.add_meter('diagnostics/accuracy', tnt.meter.ClassErrorMeter(topk=[1], accuracy=True))
        mlog.add_meter('diagnostics/perplexity', tnt.meter.ValueSummaryMeter())
        mlog.add_meter('diagnostics/class_histogram', tnt.meter.ValueSummaryMeter(), ptype='histogram')
        mlog.add_meter('diagnostics/confusion_matrix', tnt.meter.ValueSummaryMeter(), ptype='image')

    ############################################################
    # Training
    ############################################################
    if cfg['training']['train']:
        if cfg['training']['resume_training'] and loaded_optimizer is None:
            warnings.warn('resume_training is set but the optimizer is not found, reinitializing optimizer')
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
        scheduler = None
        if cfg['learner']['lr_scheduler_method'] is not None:
            scheduler = eval(cfg['learner']['lr_scheduler_method'])(optimizer, **cfg['learner']['lr_scheduler_method_kwargs'])
        logger.info("Starting training...")
        context = train_model(cfg, model, dataloaders, loss_fn, optimizer, start_epoch=start_epoch, 
                                      num_epochs=cfg['training']['num_epochs'], save_epochs=cfg['saving']['save_interval'],
                                      scheduler=scheduler, mlog=mlog, flog=flog)  

    ####################
    # Final Test
    ####################
    if cfg['training']['test']:
        run_kwargs = {
            'cfg': cfg,
            'mlog': mlog,
            'flog': flog,
            'optimizer': None,
            'loss_fn': loss_fn,
            'model': model,
            'use_thread': cfg['saving']['in_background'],
        }
        context, _ = run_one_epoch(dataloader=dataloaders['val'], epoch=0, train=False, **run_kwargs)
    
    logger.info('Waiting up to 10 minutes for all files to save...')
    [c.join(600) for c in context]
    logger.info('All saving is finished.')


def train_model(cfg, model, dataloaders, loss_fn, optimizer, start_epoch=0, num_epochs=250, save_epochs=25, scheduler=None, mlog=None, flog=None):
    checkpoint_dir = os.path.join(cfg['saving']['log_dir'], cfg['saving']['save_dir'])
    run_kwargs = {
        'cfg': cfg,
        'mlog': mlog,
        'flog': flog,
        'optimizer': optimizer,
        'loss_fn': loss_fn,
        'model': model,
        'use_thread': cfg['saving']['in_background'],
    }
    context = []
    log_interval = cfg['saving']['log_interval']
    log_interval = int(log_interval) if log_interval > 1 else log_interval
    end_epoch = num_epochs if cfg['training']['resume_w_no_add_epochs'] else start_epoch + num_epochs
    print(f'Training from epoch {start_epoch} to {end_epoch}')

    for epoch in range(start_epoch, end_epoch):
        if epoch == 0 or epoch % save_epochs == save_epochs - 1:
            context += save_checkpoint(model, optimizer, epoch, checkpoint_dir, use_thread=cfg['saving']['in_background'])

        should_run_validation = (epoch == 0) or (log_interval <= 1) or ((epoch % log_interval) == (log_interval - 1))
        if should_run_validation:
            assert math.isnan(mlog.peek_meter()['losses/total']), 'Loggers are not empty at the beginning of evaluation. Were training logs cleared?'
            context1, loss_dict = run_one_epoch(dataloader=dataloaders['val'], epoch=epoch, train=False, **run_kwargs)
            context += context1

        # if scheduler is not None and log_time:
        if scheduler is not None:
            try:
                scheduler.step(loss_dict['total'])
            except:
                scheduler.step()

        # training starts logging at epoch 1, val epoch 0 is fully random 
        context1, _ = run_one_epoch(dataloader=dataloaders['train'], epoch=epoch+1, train=True, **run_kwargs)  
        context += context1

    context1, _ = run_one_epoch(dataloader=dataloaders['val'], epoch=end_epoch, train=False, **run_kwargs)
    context += context1
    context += save_checkpoint(model, optimizer, end_epoch, checkpoint_dir, use_thread=cfg['saving']['in_background'])
    return context

      

def run_one_epoch(model, dataloader, loss_fn, optimizer, epoch, cfg, mlog, flog, train=True, use_thread=False)->(list,dict):
    # logs through the progress of the epoch from [epoch, epoch + 1)
    start_time = time.time()
    model.train(train)
    phase = 'train' if train else 'val'
    sources = cfg['training']['sources']
    targets = cfg['training']['targets']
    tasks = [t for t in SINGLE_IMAGE_TASKS if len([tt for tt in cfg['training']['targets'] if t in tt]) > 0]
    context = []
    losses = { x:[] for x in cfg['training']['loss_list'] }
    log_steps = []
    log_interval = cfg['saving']['log_interval']
    log_interval = int(log_interval) if log_interval >= 1 else log_interval
    if log_interval < 1 and train:
        num_logs_per_epoch = int(1 // log_interval)
        log_steps = [i * int(len(dataloader)/num_logs_per_epoch) for i in range(1, num_logs_per_epoch)]

    if cfg['training']['post_aggregation_transform_fn'] is not None:
        post_agg_transform = eval(cfg['training']['post_aggregation_transform_fn'])

    with torch.set_grad_enabled(train):
        for i, batch_tuple in enumerate(tqdm(dataloader, desc=f"Epoch {epoch} ({phase})")):
            cache = {'phase': phase, 'sources': sources, 'targets': targets, 'tasks': tasks}  # reset cache every pass

            if cfg['training']['post_aggregation_transform_fn'] is not None:
                batch_tuple = post_agg_transform(batch_tuple, **cfg['training']['post_aggregation_transform_fn_kwargs'])

            # Decompose batch
            x, label, masks = process_batch_tuple(batch_tuple, None, cfg)


            # Forward model
            try:
                prediction = model(x, cache=cache)
            except:
                prediction = model(x)

            if cfg['training']['algo'] == 'student':
                pass
            elif cfg['training']['algo'] == 'zero':
                prediction += label

            # Compute loss
            loss_dict = loss_fn(prediction, label, weight=masks, cache=cache)

            # Backward
            if train:
                optimizer.zero_grad()
                loss_dict['total'].backward()
                if cfg['learner']['max_grad_norm'] is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['learner']['max_grad_norm'])
                optimizer.step()

            # Write to log
            for loss in cfg['training']['loss_list']:
                assert loss in loss_dict.keys(), f'Promised to report loss {loss}, but missing from loss_dict'
                mlog.update_meter(loss_dict[loss].detach().item(), meters={f'losses/{loss}'}, phase=phase)

            if targets[0] == 'action':
                add_imitation_specific_logging(prediction, label, mlog, phase)

            if 'class_object' in tasks or 'class_scene' in tasks:
                add_classification_specific_logging(cache, mlog, phase)

            # TODO fix alpha thing
            try:
                if hasattr(model, 'module'):
                    alpha = [param for name, param in model.module.named_parameters() if 'alpha' in name][0]
                else:
                    alpha = [param for name, param in model.named_parameters() if 'alpha' in name][0]
                mlog.update_meter(torch.sigmoid(alpha).detach().item(), meters={f'introspect/alpha'}, phase=phase)
            except IndexError:
                pass
            except AttributeError:
                pass

            # Log
            if i in log_steps:  # for super long epochs where we want some information
                step = epoch + i / len(dataloader)
                step = int(np.floor(step * cfg['saving']['ticks_per_epoch']))
                for loss in cfg['training']['loss_list']:
                    losses[loss].append(mlog.peek_meter(phase=phase)[f'losses/{loss}'])
                context += log_image(mlog, flog, step, cfg, x, label, prediction, masks, cache)
                context += write_logs(mlog, flog, None, step, cfg, cache, to_print=False)

    for loss in cfg['training']['loss_list']:
        losses[loss].append(mlog.peek_meter(phase=phase)[f'losses/{loss}'].item())

    if log_interval <= 1 or epoch % log_interval == log_interval - 1 or epoch == 0:
        step = epoch + (len(dataloader) - 1) / len(dataloader)
        step = int(np.floor(step * cfg['saving']['ticks_per_epoch']))
        context += log_image(mlog, flog, step, cfg, x, label, prediction, masks, cache)
        context += write_logs(mlog, flog, None, step, cfg, cache, to_print=True)

    assert len(losses['total']) > 0, 'Need to report loss'
    for k in losses.keys():
        losses[k] = sum(losses[k]) / len(losses[k])

    loss_str = ''.join([' | ' + k + ' loss: {0:.6f} '.format(v if isinstance(v, float) else v.item()) for k, v in losses.items()])
    duration = int(time.time() - start_time)
    logger.info(f'End of epoch {epoch} ({phase}) ({duration//60}m {duration%60}s) {loss_str}')  # this is cumulative from previous train epochs in the same log_interval
    return context, losses


def log_image(mlog, flog, step, cfg, x, label, prediction, masks=None, cache={})->list:
    # Use this version of log_image as opposed to the one in logging_helpers
    tasks = cache['tasks']

    targets = cache['targets']
    phase = cache['phase']
    encoding_only = all(['encoding' in t for t in targets]) and not 'perceptual' in cfg['training']['loss_fn']
    if len(label.shape) == 4 and not encoding_only:
        if not isinstance(x, torch.Tensor):
            x = x[0]

    # make class histogram and confusion matrix
    if len(prediction.shape) == 2:
        if len(label.shape) == 2:
            label = torch.argmax(label, dim=1).cpu()
        log(mlog, 'diagnostics/class_histogram', label, phase=phase)

        pred_top = torch.argmax(prediction, dim=1).cpu()
        cm = confusion_matrix(y_true=label.cpu(), y_pred=pred_top.cpu())
        cm = (cm / np.max(cm) * 255).astype(np.uint8)
        cm = torch.ByteTensor(np.array(Image.fromarray(cm).resize((84,84), resample=Image.BOX)))
        cm = torch.stack([cm, cm, cm])
        mlog.update_meter(cm, meters={'diagnostics/confusion_matrix'}, phase=phase)

    # if any(['encoding' in t for t in targets]):  # there should have been something to do this earlier, where'd it go?
    if encoding_only:
        prediction = cache['inputs_decoded']
        if 'targets_decoded' in cache:
            label = cache['targets_decoded']

    # if len(label.shape) == 2:
    if 'class_object' in tasks or 'class_scene' in tasks:
        if not isinstance(x, torch.Tensor):
            x = x[0]
        _, _, img_size, _ = x.shape
        label = index_to_image(cache['top5_label'].cpu(), synset_arr, img_size).cuda()
        prediction = index_to_image(cache['top5_pred'].cpu(), synset_arr, img_size).cuda()

    if prediction.shape[1] == 8 and 'predictions' in cache:  # handle encodings
        prediction = cache['predictions']
        label = cache['labels']

    if prediction.shape[1] == 2:  # handle 2 channels
        zero_layer = torch.zeros_like(prediction)[:,:1,:,:]
        prediction = torch.cat((prediction, zero_layer), dim=1)
        label = torch.cat((label, zero_layer), dim=1)

    if len(label.shape) == 4 and not encoding_only:
        # Unnormalize
        x = x.cpu()
        masks = masks.cpu() if masks is not None else masks
        max_dim = multidim_apply(x, dims=[0, 2, 3], fn=torch.max)
        min_dim = multidim_apply(x, dims=[0, 2, 3], fn=torch.min)
        x_out = (x - min_dim) / (max_dim - min_dim)
        x_out = x_out * 2.0 - 1.0

        im_samples = pack_images(x_out.cpu(), prediction.cpu(), label.cpu(), mask=masks)
        log(mlog, f'decoded_image', im_samples, phase=phase)

    if isinstance(x, dict):
        log_input_images(x, mlog, cfg['training']['dataloader_fn_kwargs']['num_frames'],
                         key_names=['rgb_filled', 'map'], meter_name='input_image', reset_meter=False, phase=phase)
    elif len(x.shape) == 4:
        log(mlog, f'input_image', x[0], phase=phase)
    return []

def save_checkpoint(model, optimizer, epoch, checkpoint_dir, use_thread=False):
    dict_to_save = {
        'state_dict': model.state_dict(),
        'epoch': epoch,
        'model': model,
        'optimizer': optimizer,
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
    
