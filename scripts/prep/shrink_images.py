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
import time

from evkit.models.taskonomy_network import TaskonomyDecoder, TaskonomyNetwork
from tlkit.utils import get_parent_dirname, LIST_OF_TASKS, SINGLE_IMAGE_TASKS, TASKS_TO_CHANNELS
from tlkit.data.taskonomy_dataset import get_dataloaders, TRAIN_BUILDINGS, VAL_BUILDINGS, TEST_BUILDINGS
import tlkit.data.splits as splits
from evkit.models.taskonomy_network import TaskonomyEncoder
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ex = Experiment(name="Make bitmasks")

SOURCE_TASK = 'rgb'

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
    parent_name = get_parent_dirname(abspath) #.replace(SOURCE_TASK, "mask_valid")
    file_path = os.path.join(new_root, subfolder, parent_name, base_name)
    os.makedirs(os.path.join(new_root, subfolder, parent_name), exist_ok=True)
    if filetype == '.npy':
        np.save(file_path, arr)
    elif filetype == '.npz':
        np.savez_compressed(file_path, arr)
    elif filetype == '.png':
        cv2.imwrite(file_path, cv2.cvtColor(np.uint8(arr[0]), cv2.COLOR_RGB2BGR))
    else:
        raise NotImplementedError("Cannot save {}. Unrecognized filetype {}.".format(file_path, filetype))

def shrink_file(original_fpath, new_fpath):
    with open(original_fpath, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
    img = transforms.Resize((256,256), Image.BICUBIC)(img)    
    with open(new_fpath, 'wb') as f:
        img.save(f)
    
def save_mappable(x):
    return shrink_file(*x)


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
    assert folders_to_convert is not None, 'No folders to convert. Aborting'
        
    logger.info(f'Converting folders {str(folders_to_convert)}')

    assert len(folders_to_convert) == 1
    pool = Pool(n_dataloader_workers)
    for fpath in tqdm(os.listdir(os.path.join(data_dir, SOURCE_TASK, folders_to_convert[0], SOURCE_TASK))):
        dirname = get_parent_dirname(fpath)
        source_path = os.path.join(data_dir, SOURCE_TASK, folders_to_convert[0], SOURCE_TASK, fpath)
        target_path = os.path.join(save_dir, SOURCE_TASK, folders_to_convert[0], fpath)
        pool.apply_async(shrink_file, args=(source_path,target_path))
    pool.close()
    pool.join()
    print(target_path)
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
