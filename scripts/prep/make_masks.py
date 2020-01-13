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
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm as tqdm

from evkit.models.taskonomy_network import TaskonomyDecoder, TaskonomyNetwork
from tlkit.utils import get_parent_dirname, LIST_OF_TASKS, SINGLE_IMAGE_TASKS, TASKS_TO_CHANNELS
from tlkit.data.datasets.taskonomy_dataset import get_dataloaders, TRAIN_BUILDINGS, VAL_BUILDINGS, TEST_BUILDINGS
import tlkit.data.splits as splits
from evkit.models.taskonomy_network import TaskonomyEncoder

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ex = Experiment(name="Save activations")

SOURCE_TASK = 'depth_zbuffer'

def save_as_png(file_path, decoding):
    decoding = 0.5 * decoding + 0.5
    decoding *= (2 ** 16 - 1)
    decoding = decoding.astype(np.uint16)
        # This is fine but need to parse out the empty channel afterwords
    decoding = np.transpose(decoding, (1,2,0))
    if decoding.shape[2] > 1:
        cv2.imwrite(file_path, cv2.cvtColor(decoding, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(file_path, decoding.astype(np.uint8))
    return
  
def save_to_file(arr, original_image_fname, new_root, subfolder, filetype='.npy'):
    abspath = os.path.abspath(original_image_fname)
    base_name = os.path.basename(abspath).replace('.png', filetype)
    parent_name = get_parent_dirname(abspath).replace(SOURCE_TASK, "mask_valid")
    file_path = os.path.join(new_root, subfolder, parent_name, base_name)
    os.makedirs(os.path.join(new_root, subfolder, parent_name), exist_ok=True)
    if filetype == '.npy':
        np.save(file_path, arr)
    elif filetype == '.npz':
        np.savez_compressed(file_path, arr)
    elif filetype == '.png':
        cv2.imwrite(file_path, np.uint8(arr[0]))
    else:
        raise NotImplementedError("Cannot save {}. Unrecognized filetype {}.".format(file_path, filetype))

def save_mappable(x):
    return save_to_file(*x)


def build_mask(target, val=65000):
    mask = (target >= val)
#     mask = F.conv2d(mask.float(), torch.ones(1, 1, 5, 5, device=mask.device), padding=2, stride=2) != 0
#     mask = F.conv2d(mask.float(), torch.ones(1, 1, 5, 5, device=mask.device), padding=2, stride=2) != 0
#     mask2 = F.max_pool2d(mask.float(), 5, padding=2, stride=1) == 0
#     mask = mask * 127 + mask2*127

    mask = F.max_pool2d(mask.float(), 5, padding=2, stride=2) == 0
    return(mask)*255

@ex.main
def make_mask(folders_to_convert,
            split_to_convert,
            data_dir,
            save_dir,
            n_dataloader_workers=4,
            batch_size=64):

    if folders_to_convert is None and split_to_convert is not None:
        split_to_convert = eval(split_to_convert)
        logger.info(f'Converting from split {split_to_convert}')
        folders_to_convert = sorted(list(set(split_to_convert['train'] + split_to_convert['val'] + split_to_convert['test'])))

    if folders_to_convert is None:
        logger.info(f'Converting all folders in {data_dir}')
    else:
        logger.info(f'Converting folders {str(folders_to_convert)}')
        
    dataloader = get_dataloaders(
        data_path=data_dir, tasks=SOURCE_TASK, 
        batch_size=batch_size, batch_size_val=batch_size,
        num_workers=n_dataloader_workers,
        train_folders=None,
        val_folders=folders_to_convert,
        test_folders=None,
        zip_file_name=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )['val']
        
    pool = Pool(n_dataloader_workers)
    for fpaths, x in tqdm(dataloader):
        dirname = get_parent_dirname(fpaths[0])
        with torch.no_grad():
            x = build_mask(x)
            pool.map(save_mappable, zip(x, fpaths,
                                        [save_dir]*batch_size, ['mask_valid']*batch_size,
                                        ['.png']*batch_size))
#             return



@ex.config
def cfg_base():
    folders_to_convert = None
    split_to_convert = None
    batch_size = 64
    n_dataloader_workers = 8
    data_dir = '/mnt/data'
    save_dir = '/mnt/data'


if __name__ == "__main__":
    ex.run_commandline()
