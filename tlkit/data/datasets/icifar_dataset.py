from itertools import chain, cycle
import torch
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np
import threading
import math
from tqdm import tqdm
import warnings

from tlkit.data.sequential_tasks_dataloaders import KthDataLoader, CyclingDataLoader, ConcatenatedDataLoader

class iCIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, root, class_idxs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train, transform, target_transform, download)
        self.class_idxs = list(class_idxs)
        self.old_targets = self.targets
        is_valid = np.isin(self.targets, self.class_idxs)
        self.data    = self.data[is_valid]
        self.targets = np.int32(self.targets)[is_valid]
        self.new_to_old_class_idx = np.sort(np.unique(self.targets))
        self.old_to_new_class_idx = np.full((np.max(self.targets) + 1,), -1, dtype=np.int32)
        self.old_to_new_class_idx[self.new_to_old_class_idx] = np.arange(len(self.new_to_old_class_idx))
        self.targets = self.old_to_new_class_idx[self.targets]
        self.targets = torch.LongTensor(self.targets)
        self.classes = [c for c in self.classes if self.class_to_idx[c] in self.class_idxs]
        # print(self.classes)

        # self.data    = self.data[:5]
        # self.targets = self.targets[:5]
        # print(len(self.data))
                # return
    # def __getitem__(self, index):
    #     """
    #     Args:
    #         index (int): Index

    #     Returns:
    #         tuple: (image, target) where target is index of the target class.
    #     """
    #     img, target = self.data[index], self.targets[index]

    #     # doing this so that it is consistent with all other datasets
    #     # to return a PIL Image
    #     img = Image.fromarray(img)

    #     if self.transform is not None:
    #         img = self.transform(img)

    #     if self.target_transform is not None:
    #         target = self.target_transform(target)

    #     return img, target



def get_dataloaders(data_path,
                    targets,
                    sources=None, # Ignored
                    masks=None,   # Ignored
                    tasks=None,   # Ignored
                    epochlength=20000,
                    epochs_until_cycle=1,
                    batch_size=64,
                    batch_size_val=4,
                    transform=None,
                    num_workers=0,
                    load_to_mem=False,
                    pin_memory=False, 
                    imsize=256):
    '''
        Targets can either be of the form [iterable1, iterable2]
            or of the form 'cifarXX-YY'
    '''

    if transform is None:      
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        ])
#         transform = transforms.Compose([
#                 transforms.CenterCrop(imsize),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 normalize,
#             ])
    dataloaders = {}
    train_dataloaders = []
    classes = []
    for target in targets:
        if isinstance(target[0], str):
            start, end = [int(i) for i in target[0].lower().replace('cifar', '').split('-')]
            classes.append(np.arange(start, end + 1))
        else: 
            classes.append(target)

    for i, task in enumerate(tqdm(classes, 'Loading training data')):
        should_dl = int(i)==0
        dataset = iCIFAR100(data_path, task, train=True, transform=transform, target_transform=None, download=should_dl)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        train_dataloaders.append(dataloader)
    dataloaders['train'] = CyclingDataLoader(train_dataloaders, epochlength, epochs_until_cycle=epochs_until_cycle)

    val_dataloaders = []
    for task in tqdm(classes, 'Loading validation data'):
        dataset = iCIFAR100(data_path, task, train=False, transform=transform, target_transform=None, download=False)
        dataloader = DataLoader(dataset, batch_size=batch_size_val, shuffle=False, num_workers=num_workers, pin_memory=pin_memory) #,

        val_dataloaders.append(dataloader)
    dataloaders['val'] = ConcatenatedDataLoader(val_dataloaders)

    # dataset = iCIFAR100(data_path, train=False, transform=transform, target_transform=None, download=False)
    # dataloader = DataLoader(dataset, batch_size=batch_size_val, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    dataloaders['test'] = []
    return dataloaders



def get_limited_dataloaders(data_path,
                    sources, # Ignored
                    targets,
                    masks,   # Ignored
                    tasks=None,   # Ignored
                    epochlength=20000,
                    batch_size=64,
                    batch_size_val=4,
                    transform=None,
                    num_workers=0,
                    load_to_mem=False,
                    pin_memory=False, 
                    imsize=256):
    '''
        Targets can either be of the form [iterable1, iterable2]
            or of the form 'cifarXX-YY'
    '''

    if transform is None:      
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        ])

    dataloaders = {}
    train_dataloaders = []
    classes = []
    for target in targets:
        if isinstance(target[0], str):
            start, end = [int(i) for i in target[0].lower().replace('cifar', '').split('-')]
            classes.append(np.arange(start, end + 1))
        else: 
            classes.append(target)

    for i, task in enumerate(tqdm(classes, 'Loading training data')):
        should_dl = int(i)==0
        dataset = iCIFAR100(data_path, task, train=True, transform=transform, target_transform=None, download=should_dl)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        train_dataloaders.append(dataloader)
    dataloaders['train'] = KthDataLoader(train_dataloaders, k=0, epochlength=1000)

    val_dataloaders = []
    for task in tqdm(classes, 'Loading validation data'):
        dataset = iCIFAR100(data_path, task, train=False, transform=transform, target_transform=None, download=False)
        dataloader = DataLoader(dataset, batch_size=batch_size_val, shuffle=False, num_workers=num_workers, pin_memory=pin_memory) #,
        val_dataloaders.append(dataloader)
    dataloaders['val'] = KthDataLoader(val_dataloaders, k=0)

    dataloaders['test'] = []
    return dataloaders



def get_cifar_dataloaders(data_path,
                    sources, # Ignored
                    targets,
                    masks,   # Ignored
                    tasks=None,   # Ignored
                    epochlength=20000,
                    batch_size=64,
                    batch_size_val=4,
                    transform=None,
                    num_workers=0,
                    load_to_mem=False,
                    pin_memory=False, 
                    imsize=256):
    '''
        Targets can either be of the form [iterable1, iterable2]
            or of the form 'cifarXX-YY'
    '''

    if transform is None:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        transform_val = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
    else:
        transform_train = transform
        transform_val = transform

    dataloaders = {}

    dataset = torchvision.datasets.CIFAR10(data_path, train=True, transform=transform_train, target_transform=None, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    dataloaders['train'] = dataloader

    dataset = torchvision.datasets.CIFAR10(data_path, train=False, transform=transform_val, target_transform=None, download=False)
    dataloader = DataLoader(dataset, batch_size=batch_size_val, shuffle=False, num_workers=num_workers, pin_memory=pin_memory) #,
    dataloaders['val'] = dataloader
    dataloaders['test'] = []
    return dataloaders
