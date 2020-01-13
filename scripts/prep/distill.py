import copy
import functools
import json
import logging
import numpy as np
import os
import random
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
from torchsummary import summary
from tqdm import tqdm as tqdm

from tlkit.get_reprs import need_to_save
from evkit.models.taskonomy_network import TaskonomyDecoder, TaskonomyNetwork
from tlkit.models.student_models import AtariStudentNet, AtariStudentSmallNet
from tlkit.utils import update, var_to_numpy, save_checkpoint, log, reset_log
import tlkit.utils
from tlkit.data.taskonomy_dataset import get_dataloaders
import tlkit.data.splits as splits
from tlkit.utils import LIST_OF_TASKS, TASKS_TO_CHANNELS

from evkit.saving.observers import FileStorageObserverWithExUuid
import evkit.saving.checkpoints as checkpoints
from evkit.utils.profiler import Profiler
from evkit.utils.random import set_seed

from evkit.models.taskonomy_network import TaskonomyEncoder

# This has been giving me a lot of headaches... just try both
try:
    import tnt.torchnet as tnt
    from tnt.torchnet.logger import FileLogger
except ModuleNotFoundError:
    import torchnet as tnt
    from torchnet.logger import FileLogger


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

ex = Experiment(name="Student distillation")
LOG_DIR = sys.argv[1].strip()
sys.argv.pop(1)

def get_logger(cfg, uuid):
    if cfg['saving']['logging_type'] == 'visdom':
        mlog = tnt.logger.VisdomMeterLogger(
            title=uuid, env=uuid, server=cfg['saving']['visdom_server'],
            port=cfg['saving']['visdom_port'],
            log_to_filename=cfg['saving']['visdom_log_file']
        )
    elif cfg['saving']['logging_type'] == 'tensorboard':
        mlog = tnt.logger.TensorboardMeterLogger(
            env=uuid,
            log_dir=cfg['saving']['log_dir'],
            plotstylecombined=True,
            train_only=cfg['training']['train_only']
        )
    else:
        assert False, 'no proper logger!'
    return mlog


# This is a cool bit of code--but seems like a recipe for bugs
# def load_experiment_config(experiment_path):
#     experiment_metadata_path = [os.path.join(experiment_path, f) for f in os.listdir(experiment_path) if f.endswith('metadata')][0]
#     experiment_config_path = os.path.join(experiment_metadata_path, 'config.json')
#     with open(experiment_config_path) as f:
#         config = json.load(f)  # keys are ['cfg', 'uuid', 'seed']
#     cfg = update(cfg, config['cfg'])
#     uuid = config['uuid']
#     return cfg, uuid

def maybe_bake_decodings(cfg,logger):
    task = cfg['training']['taskonomy_encoder']
    need_encodings = (cfg['training']['baked_encoding'] \
                   and not os.path.isdir(os.path.join(cfg['training']['data_dir'], f'{task}_encoding')))
    need_decodings = (cfg['training']['baked_decoding'] \
                   and not os.path.isdir(os.path.join(cfg['training']['data_dir'], f'{task}_decoding')))

    split_to_use = eval(cfg['training']['split_to_use'])
    folders_to_convert=set(split_to_use['train'] + split_to_use['val'] + split_to_use['test'])
    if not (need_decodings or need_encodings or need_to_save(
        task=task,
        folders_to_convert=folders_to_convert,
        data_dir=cfg['training']['data_dir'],
        save_dir=cfg['training']['data_dir'],
        store_representation=cfg['training']['baked_encoding'],
        store_prediction=cfg['training']['baked_decoding'],
    )):
        return

    logger.info(f"Requiring at least one of baked encodings ({need_encodings}) or decodings ({need_decodings}). Baking...")
    from tlkit.get_reprs import save_reprs as base_decodings
    base_decodings(cfg['training']['taskonomy_encoder'],
            model_base_path=cfg['training']['encoding_base_path'],
            folders_to_convert=folders_to_convert,
            split_to_convert=None,
            data_dir=cfg['training']['data_dir'],
            save_dir=cfg['training']['data_dir'],
            store_representation=cfg['training']['baked_encoding'],
            store_prediction=cfg['training']['baked_decoding'], 
            n_dataloader_workers=cfg['training']['num_workers'],
            batch_size=cfg['training']['batch_size_val'],
            skip_done_folders=True)


@ex.main
def train(cfg, uuid):

    logger.setLevel(logging.INFO)
    logger.info(cfg)
    logger.debug(f'Loaded Torch version: {torch.__version__}')
    logger.debug(f'Using device: {device}')

    task = cfg['training']['taskonomy_encoder']
    start_epoch = 0
    
    logger.debug(f'Starting data loaders')
    maybe_bake_decodings(cfg, logger)
    set_seed(cfg['training']['seed'])
    data_subfolders = ['rgb']
    if cfg['training']['baked_encoding']:
        data_subfolders.append(f'{task}_encoding')
    if cfg['training']['baked_decoding']:
        data_subfolders.append(f'{task}_decoding')
    dataloaders = get_dataloaders(cfg['training']['data_dir'],
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
    
        
    logger.debug(f'Setting up student model')
    set_seed(cfg['training']['seed'])
    student = eval(cfg['learner']['model'])(**cfg['learner']['model_kwargs'])
    if cfg['training']['resume_from_checkpoint_path'] is not None:
        ckpt_fpath = cfg['training']['resume_from_checkpoint_path']
        checkpoint = torch.load(ckpt_fpath)
        student.load_state_dict(checkpoint['state_dict'])        
        start_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
        logger.info(f"Loaded student (epoch {start_epoch if 'epoch' in checkpoint else 'unknown'}) from {ckpt_fpath}")
    student.to(device)


    logger.debug(f'Setting up teacher')
    set_seed(cfg['training']['seed'])
    out_channels = TASKS_TO_CHANNELS[task] if task in TASKS_TO_CHANNELS else None
    teacher = TaskonomyNetwork(out_channels=out_channels)
    if cfg['training']['baked_encoding']:
        teacher.encoder = None
    else:
        teacher.load_encoder(os.path.join(cfg['training']['encoding_base_path'], f'{task}_encoder.dat'))
    if cfg['training']['baked_decoding'] and not (cfg['training']['loss_type'].upper() == 'PERCEPTION'):
        teacher.decoder = None  # No need for this
    else:
        assert out_channels is not None, f"Decoder needed for config, but unknown decoder format for task {task}."
        teacher.load_decoder(os.path.join(cfg['training']['encoding_base_path'], f'{task}_decoder.dat'))
    teacher.eval()        
    teacher.to(device)

    
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs!")
        teacher.encoder = torch.nn.DataParallel(teacher.encoder)
        teacher.decoder = torch.nn.DataParallel(teacher.decoder)
        student = torch.nn.DataParallel(student)

    if cfg['training']['loss_fn'] == 'L2':
        loss_fn = nn.MSELoss()
    elif cfg['training']['loss_fn'] == 'L1':
        loss_fn = nn.L1Loss()
    else:
        logger.warning('Using default L2/MSE loss')
        loss_fn = nn.MSELoss()


    flog = tnt.logger.FileLogger(cfg['saving']['results_log_file'], overwrite=True)
    mlog = get_logger(cfg, uuid)
    mlog.add_meter('decoded_image', tnt.meter.ValueSummaryMeter(), ptype='image')

    if cfg['training']['train']:
        mlog.add_meter('teacher_histogram', tnt.meter.ValueSummaryMeter(), ptype='histogram')
        mlog.add_meter('student_histogram', tnt.meter.ValueSummaryMeter(), ptype='histogram')
        mlog.add_meter('loss', tnt.meter.ValueSummaryMeter())
        mlog.add_meter('image', tnt.meter.ValueSummaryMeter(), ptype='image')
        optimizer = optim.Adam(student.parameters(), lr=cfg['learner']['lr'])
        scheduler = None
        if cfg['learner']['lr_scheduler_method'] is not None:
            scheduler = eval(cfg['learner']['lr_scheduler_method'])(optimizer, **cfg['learner']['lr_scheduler_method_kwargs'])
        logger.info("Starting training...")
        train_model(cfg, student, teacher, dataloaders, loss_fn, optimizer, start_epoch=start_epoch, 
                                      num_epochs=cfg['training']['num_epochs'], save_epochs=cfg['saving']['save_interval'],
                                      scheduler=scheduler, mlog=mlog, flog=flog)  

    if cfg['training']['test']:
        NotImplementedError()

    

def train_model(cfg, student, teacher, dataloaders, loss_fn, optimizer, start_epoch=0, num_epochs=250, save_epochs=25, scheduler=None, mlog=None, flog=None):
    checkpoint_dir = os.path.join(cfg['saving']['log_dir'], cfg['saving']['save_dir'])
    run_kwargs = {
        'baked_encoding': cfg['training']['baked_encoding'],
        'baked_decoding': cfg['training']['baked_decoding'],
        'mlog': mlog,
        'flog': flog,
        'optimizer': optimizer,
        'loss_type': cfg['training']['loss_type'],
        'loss_fn': loss_fn,
        'student':student,
        'teacher': teacher,
        'decoder': teacher.decoder,
        'cfg': cfg
    }

    loss = 99999
    for epoch in range(start_epoch, start_epoch + num_epochs):
        if epoch % save_epochs == save_epochs - 1 or epoch == 0:
            checkpoints.save_checkpoint({
                'state_dict': student.state_dict(),
                'epoch': epoch
            }, directory=checkpoint_dir, step_num=epoch)

        _ = run_one_epoch(dataloader=dataloaders['train'], epoch=epoch, train=True, **run_kwargs)

        if scheduler is not None:
            try:
                scheduler.step(loss)
            except:
                scheduler.step()
    
        loss = run_one_epoch(dataloader=dataloaders['val'], epoch=epoch, train=False, **run_kwargs)
    checkpoints.save_checkpoint({
        'state_dict': student.state_dict(),
        'epoch': epoch
        }, directory=checkpoint_dir, step_num=epoch)
    return student


def run_one_epoch(student, teacher, decoder, dataloader, loss_fn, loss_type, optimizer, epoch, baked_encoding, baked_decoding, mlog, flog, train, cfg):
    student.train(train)
    phase = 'train' if train else 'val'
    with torch.set_grad_enabled(train), Profiler(f"Epoch {epoch} ({'train' if train else 'val'})", logger) as prof:
        running_loss = 0.0
        for batch_tuple in tqdm(dataloader):

            # Decompose batch
            batch_tuple = [x.to(device, non_blocking=True) for x in batch_tuple]
            x = batch_tuple[0]
            if baked_encoding:
                encoding_label = batch_tuple[1]
            if baked_decoding:
                decoding_label = batch_tuple[-1]

            if train:
                optimizer.zero_grad()
            
            student_encoding = student(x)
            if loss_type.upper() == 'PERCEPTION':
                if not baked_decoding:
                    encoding_label = teacher.encoder(x)
                if not baked_decoding:
                    # logger.warning("Decoder is being used for forward pass. This is valid, though you may see speedups from using pre-baked representations")
                    with torch.no_grad():
                        decoding_label = decoder(encoding_label)
                prediction, label = decoder(student_encoding), decoding_label
            else:
                prediction, label = student_encoding, encoding_label

            loss = loss_fn(prediction, label)

            # backward + optimize only if in training phase
            if train:
                loss.backward()
                if cfg['learner']['max_grad_norm'] is not None:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), cfg['learner']['max_grad_norm'])
                optimizer.step()
            mlog.update_meter(loss.detach().item(), meters={'loss'}, phase=phase)

    if decoder is not None: # outputs images
        if len(label.shape) == 4 and label.shape[1] == 2:
            zeros = torch.zeros(label.shape[0], 1, label.shape[2], label.shape[3]).to(label.device)
            label = torch.cat([label, zeros], dim=1)
            prediction = torch.cat([prediction, zeros], dim=1)
        im_samples = torch.cat([x, prediction.expand_as(x), label.expand_as(x)], dim=3)
        im_samples = tvutils.make_grid(im_samples.detach().cpu(), nrow=1, padding=2)
        log(mlog, f'decoded_image', im_samples, phase=phase)
    logs = mlog.peek_meter(phase=phase)
    logger.info(phase + ' loss: {0:.6f}'.format(logs['loss'].item()))
    tlkit.utils.log(mlog, f'image', var_to_numpy(x[0]), phase=phase)
    tlkit.utils.reset_log(mlog, flog=flog, epoch=epoch, phase=phase)
    return logs['loss'].item()


@ex.config
def cfg_base():
    uuid = 'basic'
    cfg = {}
    cfg['learner'] = {
        'model': 'atari_residual',
        'model_kwargs' : {},
        'eps': 1e-5,                 # Small epsilon to prevent divide-by-zero
        'lr': 1e-3,                  # Learning rate for algorithm
        'lr_scheduler_method': None,
        'lr_scheduler_method_kwargs': {},
        'max_grad_norm': 1,        # Clip grads
        'test':False,
        'scheduler': 'plateau',     # Automatically reduce LR once loss plateaus (plateau, step)
    }
    cfg['training'] = {
        'baked_encoding': False,
        'baked_decoding': True,
        'batch_size': 64,
        'batch_size_val': 64,
        'cuda': True,
        'data_dir': '/mnt/hdd2/taskonomy_reps', # for docker
        'epochs': 100,
        'encoding_base_path': '/root/tlkit/taskonomy/taskbank/pytorch',
        'loss_fn': 'L1',
        'loss_type': 'perception',  # If distance in encoding space, use ''. If distance in decoding space, use 'PERCEPTION'
        'load_to_mem': False,  # if dataset small enough, can load activations to memory
        'num_workers': 8,
        'num_epochs': 1,
        'pin_memory': True,
        'resume_from_checkpoint_path': None,
        'seed': random.randint(0,1000),
        'split_to_use': 'splits.taskonomy_no_midlevel["debug"]',
        'taskonomy_encoder': 'autoencoding',
        'train': True,
        'train_only': False,
        'test': False,
    }
    cfg['saving'] = {
        'log_dir': LOG_DIR,
        'log_interval': 1,
        'logging_type': 'tensorboard',
        'results_log_file': os.path.join(LOG_DIR, 'result_log.pkl'),
        'reward_log_file': os.path.join(LOG_DIR, 'rewards.pkl'),
        'save_interval': 1,
        'save_dir': 'checkpoints',
        'visdom_log_file': os.path.join(LOG_DIR, 'visdom_logs.json'),
        'visdom_server': 'r2d2.eecs.berkeley.edu',
        'visdom_port': '8097',
    }
    

##################
# Students
##################
@ex.named_config
def model_fcn5():
    cfg = {'learner': {
        'model': 'FCN5Residual',
        'model_kwargs': {
            'num_groups': 2,
            'use_residual': False,
            'normalize_output': False,
        } } }

@ex.named_config
def model_fcn5_residual():
    cfg = {'learner': {
        'model': 'FCN5Residual',
        'model_kwargs': {
            'num_groups': 2,
            'use_residual': True,
            'normalize_output': False,
        } } }

@ex.named_config
def model_fcn3():
    cfg = { 'learner': {
        'model': 'FCN3',
        'model_kwargs': {
            'num_groups': 2,
            'normalize_output': False,
        } } }

@ex.named_config
def student_taskonomy_encoder_penultimate():
    cfg = { 'learner': {
            'model': 'TaskonomyEncoder',
            'model_kwargs': {
                'train': True,
                'eval_only': False,
          } } }

@ex.named_config
def student_taskonomy_encoder():
    cfg = { 'learner': {
            'model': 'TaskonomyEncoder',
            'model_kwargs': {
                'train_penultimate': True,
                'eval_only': False,
          } } }


    
    
##################
# Learning Rates
##################
@ex.named_config
def scheduler_reduce_on_plateau():
    cfg = { 'learner': {
            'lr_scheduler_method': 'lr_scheduler.ReduceLROnPlateau',
            'lr_scheduler_method_kwargs': {
                'factor': 0.1,
                'patience': 5
          } } }



@ex.named_config
def scheduler_step_lr():
    cfg = { 'learner': {
            'lr_scheduler_method': 'lr_scheduler.StepLR',
            'lr_scheduler_method_kwargs': {
                'lr_decay_epochs': 30,       # number of epochs before the LR drops by factor of 10
                'gamma': 0.1
          } } }



@ex.named_config
def cfg_eval():
    uuid = 'eval'
    cfg = {}
    cfg['learner'] = {
        'model': 'FCN5',
        'test': True,
    }
    cfg['training'] = {
        'train': False,
    }

            
if __name__ == '__main__':
    assert LOG_DIR, 'log dir cannot be empty'
    os.makedirs(LOG_DIR, exist_ok=True)
    # subprocess.call("rm -rf {}/*".format(LOG_DIR), shell=True)  # need this to clean
    ex.observers.append(FileStorageObserverWithExUuid.create(LOG_DIR))
    ex.run_commandline()
else:
    print(__name__)
    
