import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np
import threading
import math


def crop(img,tw,th):
        n, c, w, h = img.shape
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img[:, :, x1:x1+tw,y1:y1+th].clone().detach()



class RotateBatch(object):

    def __init__(self, size=256):
        self.ANGLE = 0.0
        self.size = 256


    def __call__(self, batch):
        print(self.ANGLE)
        self.ANGLE += 10
        theta = self.ANGLE / 360 * 2 * math.pi
        with torch.no_grad():
            rot = torch.tensor([[math.cos(theta), -math.sin(theta), 0],
                                [math.sin(theta), math.cos(theta), 0],
                                ])
            rot = torch.stack([rot] * batch[0].shape[0], dim=0)
            ff = F.affine_grid(rot, batch[0].shape)
            img = F.grid_sample(batch[0], ff)
            img = crop(img, self.size, self.size)
            batch = [img, torch.tensor(self.ANGLE)] +  batch[1:]
        return batch

def imagenet_rotation_transform(angle, imsize):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
                transforms.Resize(int(np.ceil(imsize * np.sqrt(2)))),
                # lambda x: TF.rotate(x, angle),
                transforms.CenterCrop(int(np.ceil(imsize * np.sqrt(2)))),
                transforms.ToTensor(),
                normalize,
            ])
    return transform

def get_dataloaders(data_path,
                    tasks,
                    batch_size=64,
                    batch_size_val=4,
                    transform=None,
                    num_workers=0,
                    load_to_mem=False,
                    pin_memory=False, 
                    imsize=256):


    if transform is None:
        transform = imagenet_rotation_transform(30.0, imsize)
#         normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])

#         transform = transforms.Compose([
#                 transforms.CenterCrop(imsize),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 normalize,
#             ])

    dataloaders = {}
    dataset = torchvision.datasets.ImageNet(data_path, split='train', transform=transform, target_transform=None, download=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    dataloaders['train'] = dataloader

    dataset = torchvision.datasets.ImageNet(data_path, split='val', transform=transform, target_transform=None, download=False)
    dataloader = DataLoader(dataset, batch_size=batch_size_val, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    dataloaders['val'] = dataloader

    dataset = torchvision.datasets.ImageNet(data_path, split='val', transform=transform, target_transform=None, download=False)
    dataloader = DataLoader(dataset, batch_size=batch_size_val, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    dataloaders['test'] = dataloader
    return dataloaders


import os
import shutil
import torch
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import check_integrity, download_url

ARCHIVE_DICT = {
    'train': {
        'url': 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar',
        'md5': '1d675b47d978889d74fa0da5fadfb00e',
    },
    'val': {
        'url': 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar',
        'md5': '29b22e2961454d5413ddabcf34fc5622',
    },
    'devkit': {
        'url': 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_devkit_t12.tar.gz',
        'md5': 'fa75699e90414af021442c21a62c3abf',
    }
}


class ImageNet(ImageFolder):
    """`ImageNet <http://image-net.org/>`_ 2012 Classification Dataset.

    Args:
        root (string): Root directory of the ImageNet Dataset.
        split (string, optional): The dataset split, supports ``train``, or ``val``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
        imgs (list): List of (image path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, split='train', download=False, **kwargs):
        root = self.root = os.path.expanduser(root)
        self.split = self._verify_split(split)

        if download:
            self.download()
        wnid_to_classes = self._load_meta_file()[0]

        super(ImageNet, self).__init__(self.split_folder, **kwargs)
        self.root = root

        idcs = [idx for _, idx in self.imgs]
        self.wnids = self.classes
        self.wnid_to_idx = {wnid: idx for idx, wnid in zip(idcs, self.wnids)}
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx
                             for clss, idx in zip(self.classes, idcs)
                             for cls in clss}

    def download(self):
        if not check_integrity(self.meta_file):
            tmpdir = os.path.join(self.root, 'tmp')

            archive_dict = ARCHIVE_DICT['devkit']
            download_and_extract_tar(archive_dict['url'], self.root,
                                     extract_root=tmpdir,
                                     md5=archive_dict['md5'])
            devkit_folder = _splitexts(os.path.basename(archive_dict['url']))[0]
            meta = parse_devkit(os.path.join(tmpdir, devkit_folder))
            self._save_meta_file(*meta)

            shutil.rmtree(tmpdir)

        if True: #not os.path.isdir(self.split_folder):
            archive_dict = ARCHIVE_DICT[self.split]
#             download_and_extract_tar(archive_dict['url'], self.root,
#                                      extract_root=self.split_folder,
#                                      md5=archive_dict['md5'])

            if self.split == 'train':
                prepare_train_folder(self.split_folder)
            elif self.split == 'val':
                val_wnids = self._load_meta_file()[1]
                prepare_val_folder(self.split_folder, val_wnids)
        else:
            msg = ("You set download=True, but a folder '{}' already exist in "
                   "the root directory. If you want to re-download or re-extract the "
                   "archive, delete the folder.")
            print(msg.format(self.split))

    @property
    def meta_file(self):
        return os.path.join(self.root, 'meta.bin')

    def _load_meta_file(self):
        if check_integrity(self.meta_file):
            return torch.load(self.meta_file)
        else:
            raise RuntimeError("Meta file not found or corrupted.",
                               "You can use download=True to create it.")

    def _save_meta_file(self, wnid_to_class, val_wnids):
        torch.save((wnid_to_class, val_wnids), self.meta_file)

    def _verify_split(self, split):
        if split not in self.valid_splits:
            msg = "Unknown split {} .".format(split)
            msg += "Valid splits are {{}}.".format(", ".join(self.valid_splits))
            raise ValueError(msg)
        return split

    @property
    def valid_splits(self):
        return 'train', 'val'

    @property
    def split_folder(self):
        return os.path.join(self.root, self.split)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)



def extract_tar(src, dest=None, gzip=None, delete=False):
    import tarfile

    if dest is None:
        dest = os.path.dirname(src)
    if gzip is None:
        gzip = src.lower().endswith('.gz')

    mode = 'r:gz' if gzip else 'r'
    with tarfile.open(src, mode) as tarfh:
        tarfh.extractall(path=dest)

    if delete:
        os.remove(src)


def download_and_extract_tar(url, download_root, extract_root=None, filename=None,
                             md5=None, **kwargs):
    return 
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if filename is None:
        filename = os.path.basename(url)

    if not check_integrity(os.path.join(download_root, filename), md5):
        download_url(url, download_root, filename=filename, md5=md5)

    extract_tar(os.path.join(download_root, filename), extract_root, **kwargs)


def parse_devkit(root):
    idx_to_wnid, wnid_to_classes = parse_meta(root)
    val_idcs = parse_val_groundtruth(root)
    val_wnids = [idx_to_wnid[idx] for idx in val_idcs]
    return wnid_to_classes, val_wnids


def parse_meta(devkit_root, path='data', filename='meta.mat'):
    import scipy.io as sio

    metafile = os.path.join(devkit_root, path, filename)
    meta = sio.loadmat(metafile, squeeze_me=True)['synsets']
    nums_children = list(zip(*meta))[4]
    meta = [meta[idx] for idx, num_children in enumerate(nums_children)
            if num_children == 0]
    idcs, wnids, classes = list(zip(*meta))[:3]
    classes = [tuple(clss.split(', ')) for clss in classes]
    idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
    wnid_to_classes = {wnid: clss for wnid, clss in zip(wnids, classes)}
    return idx_to_wnid, wnid_to_classes


def parse_val_groundtruth(devkit_root, path='data',
                          filename='ILSVRC2012_validation_ground_truth.txt'):
    with open(os.path.join(devkit_root, path, filename), 'r') as txtfh:
        val_idcs = txtfh.readlines()
    return [int(val_idx) for val_idx in val_idcs]


def prepare_train_folder(folder):
    for archive in [os.path.join(folder, archive) for archive in os.listdir(folder)]:
        extract_tar(archive, os.path.splitext(archive)[0], delete=True)


def prepare_val_folder(folder, wnids):
    img_files = sorted([os.path.join(folder, file) for file in os.listdir(folder)])

    for wnid in set(wnids):
        os.mkdir(os.path.join(folder, wnid))

    for wnid, img_file in zip(wnids, img_files):
        shutil.move(img_file, os.path.join(folder, wnid, os.path.basename(img_file)))


def _splitexts(root):
    exts = []
    ext = '.'
    while ext:
        root, ext = os.path.splitext(root)
        exts.append(ext)
    return root, ''.join(reversed(exts))