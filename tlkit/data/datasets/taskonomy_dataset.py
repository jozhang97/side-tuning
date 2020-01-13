from collections import namedtuple, Counter, defaultdict
from tlkit.data.sequential_tasks_dataloaders import ConcatenatedDataLoader, CyclingDataLoader, ErrorPassingConcatenatedDataLoader, ErrorPassingCyclingDataLoader
from tlkit.utils import SINGLE_IMAGE_TASKS, TASKS_TO_CHANNELS
import torch
import torch.utils.data as utils
import torchvision.transforms as transforms
import torchvision.datasets as ds
import torch.utils.data as data
from tqdm import tqdm
from PIL import Image, ImageFile
import numpy as np
import os
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import warnings

from tlkit.data.img_transforms import default_loader, get_transform
from tlkit.data.splits import SPLIT_TO_NUM_IMAGES, taskonomy_no_midlevel as split_taskonomy_no_midlevel

TRAIN_BUILDINGS = split_taskonomy_no_midlevel['fullplus']['train']
VAL_BUILDINGS = split_taskonomy_no_midlevel['fullplus']['val']
TEST_BUILDINGS = split_taskonomy_no_midlevel['fullplus']['test']


ImageFile.LOAD_TRUNCATED_IMAGES = True # TODO Test this


class TaskonomyData(data.Dataset):
    '''
        Loads data for the Taskonomy dataset.
        This expects that the data is structured
        
            /path/to/data/
                rgb/
                    modelk/
                        point_i_view_j.png
                        ...                        
                depth_euclidean/
                ... (other tasks)
                
        If one would like to use pretrained representations, then they can be added into the directory as:
            /path/to/data/
                rgb_encoding/
                    modelk/
                        point_i_view_j.npy
                ...
        
        Basically, any other folder name will work as long as it is named the same way.
    '''
    def __init__(self, data_path,
                 tasks,
                 buildings,
                 transform=None,
                 load_to_mem=False,
                 zip_file_name=False,
                 max_images=None):
        '''
            data_path: Path to data
            tasks: Which tasks to load. Any subfolder will work as long as data is named accordingly
            buildings: Which models to include. See `splits.taskonomy`
            transform: one transform per task.
            
            Note: This assumes that all images are present in all (used) subfolders
        '''
        self.return_tuple = True
        if isinstance(tasks, str):
            tasks = [tasks]
            transform = [transform]            
            self.return_tuple = False
 
        self.buildings = buildings
        self.cached_data = {}
        self.data_path = data_path
        self.load_to_mem = load_to_mem
        self.tasks = tasks
        self.zip_file_name = zip_file_name

        self.urls = {task: make_dataset(os.path.join(data_path, task), buildings, max_images)
                     for task in tasks}

        # Validate number of images
        n_images_task = [(len(obs), task) for task, obs in self.urls.items()]
        print("\t" + "  |  ".join(["{}: {}".format(k, task) for task, k in n_images_task]))
        if max(n_images_task)[0] != min(n_images_task)[0]:
            print("Each task must have the same number of images. However, the max != min ({} != {}). Number of images per task is: \n\t{}".format(
                max(n_images_task)[0], min(n_images_task)[0], "\n\t".join([str(t) for t in n_images_task])))

            # count number of frames per building per task
            all_buildings = defaultdict(dict)
            for task, obs in self.urls.items():
                c = Counter([url.split("/")[-2] for url in obs])
                for building in c:
                    all_buildings[building][task] = c[building]

            # find where the number of distinct counts is more than 1
            print('Removing data from the following buildings')
            buildings_to_remove = []
            for b, count in all_buildings.items():
                if len(set(list(count.values()))) > 1:
                    print(f"\t{b}:", count)
                    buildings_to_remove.append(b)
            # [(len(obs), task) for task, obs in self.urls.items()]

            # redo the loading with fewer buildings
            buildings_redo = [b for b in buildings if b not in buildings_to_remove]
            self.urls = {task: make_dataset(os.path.join(data_path, task), buildings_redo)
                        for task in tasks}
            n_images_task = [(len(obs), task) for task, obs in self.urls.items()]
            print("\t" + "  |  ".join(["{}: {}".format(k, task) for task, k in n_images_task]))
        assert max(n_images_task)[0] == min(n_images_task)[0], \
                "Each task must have the same number of images. However, the max != min ({} != {}). Number of images per task is: \n\t{}".format(
                max(n_images_task)[0], min(n_images_task)[0], "\n\t".join([str(t) for t in n_images_task]))
        self.size = max(n_images_task)[0]

        # Perhaps load some things into main memory
        if load_to_mem:
            print('Writing activations to memory')
            for t, task in zip(transform, tasks):
                self.cached_data[task] = [None] * len(self)
                for i, url in enumerate(self.urls[task]):
                    self.cached_data[task][i] = t(default_loader(url))
                self.cached_data[task] = torch.stack(self.cached_data[task])
#             self.cached_data = torch.stack(self.cached_data)
            print('Finished writing some activations to memory')
            
        self.transform = transform


    def __len__(self):
        return self.size

    def __getitem__(self, index):
        fpaths = [self.urls[task][index] for task in self.tasks]
        
        if self.load_to_mem:
            result = tuple([self.cached_data[task][index] for task in self.tasks])
        else:
            result = [default_loader(path) for path in fpaths]
            if self.transform is not None:
                # result = [transform(tensor) for transform, tensor in zip(self.transform, result)]
                result_post = []
                for i, (transform, tensor) in enumerate(zip(self.transform, result)):
                    try:
                        result_post.append(transform(tensor))
                    except Exception as e:
                        print(self.tasks[i], transform, tensor)
                        raise e
                result = result_post

        # handle 2 channel outputs
        for i in range(len(self.tasks)):
            task = self.tasks[i]
            base_task = [t for t in SINGLE_IMAGE_TASKS if t in task]
            if len(base_task) == 0:
                continue
            else:
                base_task = base_task[0]
            num_channels = TASKS_TO_CHANNELS[base_task]
            if 'decoding' in task and result[i].shape[0] != num_channels:
                assert torch.sum(result[i][num_channels:,:,:]) < 1e-5, 'unused channels should be 0.'
                result[i] = result[i][:num_channels,:,:]

        if self.zip_file_name:
            result = tuple(zip(fpaths, result))

        if self.return_tuple:
            return result
        else:
            return result[0]

            

def make_dataset(dir, folders=None, max_images=None):
    #  folders are building names. If None, get all the images (from both building folders and dir)
    has_reached_capacity = lambda images, max_images: not max_images is None and len(images) >= max_images
    images = []
    dir = os.path.expanduser(dir)
    if not os.path.isdir(dir):
        assert "bad directory"

    for subfolder in sorted(os.listdir(dir)):
        subfolder_path = os.path.join(dir, subfolder)
        if os.path.isdir(subfolder_path) and (folders is None or subfolder in folders):
            for fname in sorted(os.listdir(subfolder_path)):
                path = os.path.join(subfolder_path, fname)
                if not has_reached_capacity(images, max_images):
                    images.append(path)

        # If folders/buildings are not specified, use images in dir
        if folders is None and os.path.isfile(subfolder_path) and not has_reached_capacity(images, max_images):
            images.append(subfolder_path)

    return images


def get_dataloaders(data_path,
                    tasks,
                    batch_size=64,
                    batch_size_val=4,
                    zip_file_name=False,
                    train_folders=TRAIN_BUILDINGS,
                    val_folders=VAL_BUILDINGS,
                    test_folders=TEST_BUILDINGS,
                    transform=None,
                    num_workers=0,
                    load_to_mem=False,
                    pin_memory=False,
                    max_images=None):
    """
    :param data_path: directory that data is stored at
    :param tasks: names of subdirectories to return observations from
    :param batch_size:
    :param zip_file_name: when returning an observation, this will zip the fpath to it. E.g. (/path/to/img.png, OBS)
    :param train_folders: in a big data dir, which subfolders contain our training data
    :param val_folders: in a big data dir, which subfolders contain our val data
    :param max_images: maximum number of images in any dataset
    :return: dictionary of dataloaders
    """

    if transform is None:
        if isinstance(tasks, str):
            transform = get_transform(tasks)
        else:
            transform = [get_transform(task) if len(task.split(' ')) == 1 else get_transform(*task.split(' ')) for task in tasks]
            tasks = [t.split(' ')[0] for t in tasks]  # handle special data operations

    if isinstance(train_folders, str):
        train_folders = split_taskonomy_no_midlevel[train_folders]['train']
    if isinstance(val_folders, str):
        val_folders = split_taskonomy_no_midlevel[val_folders]['val']
    if isinstance(test_folders, str):
        test_folders = split_taskonomy_no_midlevel[test_folders]['test']


    dataloaders = {}
    print(f"Taskonomy dataset TRAIN folders: {train_folders}")
    dataset = TaskonomyData(data_path, tasks, buildings=train_folders,
                                transform=transform, zip_file_name=zip_file_name,
                                load_to_mem=load_to_mem, max_images=max_images)
    if len(dataset) == 0:
        print(f'\tNO IMAGES FOUND for tasks {tasks} at path {data_path}')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    dataloaders['train'] = dataloader

    print(f"Taskonomy dataset VAL folders: {val_folders}")
    dataset = TaskonomyData(data_path, tasks, buildings=val_folders,
        transform=transform, zip_file_name=zip_file_name, load_to_mem=load_to_mem, max_images=max_images)

    if len(dataset) == 0:
        print(f'\tNO IMAGES FOUND for tasks {tasks} at path {data_path}')
    dataloader = DataLoader(dataset, batch_size=batch_size_val, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    dataloaders['val'] = dataloader

    print(f"Taskonomy dataset TEST folders: {test_folders}")
    dataset = TaskonomyData(data_path, tasks, buildings=test_folders,
                            transform=transform, zip_file_name=zip_file_name, load_to_mem=load_to_mem, max_images=max_images)
    if len(dataset) == 0:
        print(f'\tNO IMAGES FOUND for tasks {tasks} at path {data_path}')
    dataloader = DataLoader(dataset, batch_size=batch_size_val, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    dataloaders['test'] = dataloader
    return dataloaders


def get_lifelong_dataloaders(data_path,
                             sources,
                             targets,
                             masks,
                             epochs_per_task=5,
                             epochs_until_cycle=0,
                             split='fullplus',
                             batch_size=64,
                             batch_size_val=4,
                             transform=None,
                             num_workers=0,
                             load_to_mem=False,
                             pin_memory=False,
                             speedup_no_rigidity=False,
                             max_images_per_task=None):

    phases = ['train', 'val', 'test']
    dataloaders = {phase: [] for phase in phases}

    if isinstance(masks, bool):
        masks = [masks] * len(sources)

    masks = [['mask_valid'] if mask else [] for mask in masks]

    for i, (source, target, mask) in enumerate(zip(sources, targets, masks)):
        print(f'# Task {i} dataloader: {source} -> {target}')
        tasks = source + target + mask
        dl = get_dataloaders(
                    data_path,
                    tasks,
                    batch_size=batch_size,
                    batch_size_val=batch_size_val,
                    train_folders=split,
                    val_folders=split,
                    test_folders=split,
                    transform=transform,
                    num_workers=num_workers,
                    load_to_mem=load_to_mem,
                    pin_memory=pin_memory,
                    max_images=max_images_per_task,
        )
        for phase in phases:
            dataloaders[phase].append(dl[phase])

    if speedup_no_rigidity:
        # For methods that do not forget (no intransigence) by construction.
        # In validation, we only compute task performance for just-trained task and next-to-be-trained task
        epoch_lengths = [len(dl.dataset) for dl in dataloaders['val']]
        epoch_length = min(epoch_lengths) if min(epoch_lengths) == max(epoch_lengths) else None

        dl_just_trained = CyclingDataLoader(dataloaders['val'], epochs_until_cycle=1, start_dl=0,
                                            epoch_length_per_dl=epoch_length)
        dl_next_to_be_trained = CyclingDataLoader(dataloaders['val'], epochs_until_cycle=0, start_dl=0,
                                                  epoch_length_per_dl=epoch_length)
        dataloaders['val'] = ErrorPassingConcatenatedDataLoader([dl_just_trained, dl_next_to_be_trained], zip_idx=False)
    else:
        dataloaders['val'] = ErrorPassingConcatenatedDataLoader(dataloaders['val'])

    train_epoch_length = SPLIT_TO_NUM_IMAGES[split] if split is not None else min([len(dl.dataset) for dl in dataloaders['train']])
    dataloaders['train'] = ErrorPassingCyclingDataLoader(dataloaders['train'], epoch_length_per_dl=epochs_per_task * train_epoch_length, epochs_until_cycle=epochs_until_cycle)
    dataloaders['test'] = ErrorPassingConcatenatedDataLoader(dataloaders['test'])
    return dataloaders




