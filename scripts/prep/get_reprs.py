# Runs encoder on raw images and bakes/saves encodings
# We want to run encoder and decoder
# Example usage:
#     python -m tlkit.get_reprs run_cfg with split_to_convert='splits.taskonomy["debug"]'

import copy
import cv2
from functools import partial
import logging
from multiprocessing import Pool
import multiprocessing
import numpy as np
import os
from sacred import Experiment
import torch
import torchvision.transforms as transforms
from tqdm import tqdm as tqdm

from evkit.models.taskonomy_network import TaskonomyDecoder, TaskonomyNetwork

from tlkit.utils import get_parent_dirname, LIST_OF_TASKS, SINGLE_IMAGE_TASKS, TASKS_TO_CHANNELS, FEED_FORWARD_TASKS
from tlkit.data.taskonomy_dataset import get_dataloaders, TRAIN_BUILDINGS, VAL_BUILDINGS, TEST_BUILDINGS
import tlkit.data.splits as splits
from evkit.models.taskonomy_network import TaskonomyEncoder

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ex = Experiment(name="Save activations")

def save_as_png(file_path, decoding):
    decoding = 0.5 * decoding + 0.5
    decoding *= (2 ** 16 - 1)
    decoding = decoding.astype(np.uint16)
    if decoding.shape[0] == 2:
        zeros = np.zeros((1, decoding.shape[1], decoding.shape[2]), dtype=np.uint16)
        decoding = np.vstack((decoding, zeros))
        # This is fine but need to parse out the empty channel afterwords
    decoding = np.transpose(decoding, (1,2,0))
    if decoding.shape[2] > 1:
        cv2.imwrite(file_path, cv2.cvtColor(decoding, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(file_path, decoding)
    return
  
def save_to_file(arr, original_image_fname, new_root, subfolder, filetype='.npy'):
    abspath = os.path.abspath(original_image_fname)
    base_name = os.path.basename(abspath).replace('.png', filetype)
    parent_name = get_parent_dirname(abspath)
    file_path = os.path.join(new_root, subfolder, parent_name, base_name)
    os.makedirs(os.path.join(new_root, subfolder, parent_name), exist_ok=True)
    if filetype == '.npy':
        np.save(file_path, arr)
    elif filetype == '.npz':
        np.savez_compressed(file_path, arr)
    elif filetype == '.png':
        save_as_png(file_path, arr)
    else:
        raise NotImplementedError("Cannot save {}. Unrecognized filetype {}.".format(file_path, filetype))

def save_mappable(x):
    return save_to_file(*x)

def remove_done_folders(task, folders_to_convert, data_dir, save_dir, store_prediction, store_representation):
    rgb_dir = os.path.join(data_dir, 'rgb')
    encoding_dir = os.path.join(save_dir, f'{task}_encoding')
    decoding_dir = os.path.join(save_dir, f'{task}_decoding')
    folders_to_use = set()
    for folder in folders_to_convert:
        rgb_folder = os.path.join(rgb_dir, folder)
        decoding_folder = os.path.join(decoding_dir, folder)
        encoding_folder = os.path.join(encoding_dir, folder)

        # if rgb folder does not exist we cannot get the repr for it
        if not os.path.exists(rgb_folder):
            print(f'Skipping {folder} because no rgb folder (but is that true? This is probably caused by a bug somewhere)')
            continue

        # only keep if it does not exist or num of files do not match
        if store_representation and not os.path.exists(encoding_folder):
            folders_to_use.add(folder)
        elif store_representation and len(os.listdir(encoding_folder)) != len(os.listdir(rgb_folder)):
            folders_to_use.add(folder)

        if store_prediction and not os.path.exists(decoding_folder):
            folders_to_use.add(folder)
        elif store_prediction and len(os.listdir(decoding_folder)) != len(os.listdir(rgb_folder)):
            folders_to_use.add(folder)
    return list(folders_to_use)

def need_to_save(task, folders_to_convert, data_dir, save_dir, store_prediction, store_representation):
    folders_to_convert = remove_done_folders(task, folders_to_convert, data_dir, save_dir, store_prediction, store_representation)
    return len(folders_to_convert) != 0

def save_reprs(task,
            model_base_path,
            folders_to_convert,
            split_to_convert,
            data_dir,
            save_dir,
            store_representation=True,
            store_prediction=True, 
            n_dataloader_workers=4,
            batch_size=64,
            skip_done_folders=True):
    logger.info(f'Setting up model of {task} with {model_base_path}')
    out_channels = TASKS_TO_CHANNELS[task] if task in TASKS_TO_CHANNELS else None
    feed_forward = task in FEED_FORWARD_TASKS
    model = TaskonomyNetwork(out_channels=out_channels, feed_forward=feed_forward)
    model.load_encoder(os.path.join(model_base_path, f'{task}_encoder.dat'))
    if store_prediction:
        if out_channels is None:
            NotImplementedError(f"Unknown decoder format for task {task}")
        model.load_decoder(os.path.join(model_base_path, f'{task}_decoder.dat'))
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs!")
        model.encoder = torch.nn.DataParallel(model.encoder)
        model.decoder = torch.nn.DataParallel(model.decoder) if store_prediction else model.decoder

    model.eval()        
    model.to(device)
    

    if folders_to_convert is None and split_to_convert is not None:
        split_to_convert = eval(split_to_convert)
        logger.info(f'Converting from split {split_to_convert}')
        folders_to_convert = sorted(list(set(split_to_convert['train'] + split_to_convert['val'] + split_to_convert['test'])))
    assert folders_to_convert is not None, 'No folders to convert. Aborting'

    if skip_done_folders:
        folders_to_convert = remove_done_folders(task, folders_to_convert, data_dir, save_dir, store_prediction, store_representation)

    logger.info(f'Converting folders {str(folders_to_convert)}')

    if task not in SINGLE_IMAGE_TASKS:
        raise NotImplementedError(f'Distillation is currently implemented only for single-image-input tasks.')
    dataloader = get_dataloaders(
        data_path=data_dir, tasks='rgb', 
        batch_size=batch_size, batch_size_val=batch_size,
        num_workers=n_dataloader_workers,
        train_folders=None,
        val_folders=folders_to_convert,
        test_folders=None,
        zip_file_name=True
    )['val']

    pred_format = '.npy' if feed_forward else '.png'
    pool = Pool(n_dataloader_workers)
    for fpaths, x in tqdm(dataloader):
        dirname = get_parent_dirname(fpaths[0])
        x = x.to(device)
        with torch.no_grad():
            encodings = model.encoder(x)
            if store_representation:
                encodings_np = encodings.cpu().detach().numpy()
                pool.map(save_mappable, zip(encodings_np, fpaths,
                                            [save_dir]*batch_size, [f'{task}_encoding']*batch_size,
                                            ['.npy']*batch_size))
#                 [save_to_file(arr, fpath, save_dir, f'{task}_encoding') for arr, fpath in zip(encodings_np, fpaths)]
            if store_prediction:
                decodings = model.decoder(encodings)
                decodings_np = decodings.cpu().detach().numpy()
                pool.map(save_mappable, zip(decodings_np, fpaths,
                                            [save_dir]*batch_size, [f'{task}_decoding']*batch_size,
                                            [pred_format]*batch_size))
#                 [save_to_file(arr, fpath, save_dir, f'{task}_decoding') for arr, fpath in zip(decodings_np, fpaths)]


@ex.main
def run_cfg(cfg):
    save_reprs(
        task=cfg['task'],
        model_base_path=cfg['model_base_path'],
        folders_to_convert=cfg['folders_to_convert'],
        split_to_convert=cfg['split_to_convert'],
        data_dir=cfg['data_dir'],
        save_dir=cfg['save_dir'],
        store_representation=cfg['store_representation'],
        store_prediction=cfg['store_prediction'],
        n_dataloader_workers=cfg['n_dataloader_workers'],
        batch_size=cfg['batch_size']
    )




@ex.config
def cfg_base():
    task = 'autoencoding'
    model_base_path = '/mnt/models/'
    store_representation = True
    store_prediction = True
    folders_to_convert = None
    split_to_convert = None
    batch_size = 64
    n_dataloader_workers = 8
    data_dir = '/mnt/data'
    save_dir = '/mnt/data'


@ex.named_config
def cfg_docker():
    cfg = {
        'task': 'keypoints3d',
        'model_base_path': '/mnt/models/',
        'store_representation': False,
        'store_prediction': True,
        'split_to_convert': 'splits.taskonomy_no_midlevel["fullplus"]',
        'data_dir': '/mnt/data',
        'save_dir': '/mnt/data',
        'folders_to_convert': None,
        'batch_size': 64,
        'n_dataloader_workers': 8,
    }


if __name__ == "__main__":
    ex.run_commandline()
