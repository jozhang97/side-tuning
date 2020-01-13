from PIL import Image
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from tlkit.utils import TASKS_TO_CHANNELS, FEED_FORWARD_TASKS

MAKE_RESCALE_0_1_NEG1_POS1 = lambda n_chan: transforms.Normalize([0.5]*n_chan, [0.5]*n_chan)
RESCALE_0_1_NEG1_POS1 = transforms.Normalize([0.5], [0.5])  # This needs to be different depending on num out chans
MAKE_RESCALE_0_MAX_NEG1_POS1 = lambda maxx: transforms.Normalize([maxx / 2.], [maxx * 1.0])
RESCALE_0_255_NEG1_POS1 = transforms.Normalize([127.5,127.5,127.5], [255, 255, 255])
shrinker = nn.Upsample(scale_factor=0.125, mode='nearest')

def get_transform(task, special=None):
    if task in ['rgb', 'normal', 'reshading']:
        if special is None:
            return transform_8bit
        elif special == 'compressed':
            return lambda x: downsample_group_stack(transform_8bit(x))
    elif task in ['mask_valid']:
        return transforms.ToTensor()
    elif task in ['keypoints2d', 'keypoints3d', 'depth_euclidean', 'depth_zbuffer', 'edge_texture', 'edge_occlusion']:
#         return transform_16bit_int
        return transform_16bit_single_channel
    elif task in ['principal_curvature', 'curvature']:
        if special is None:
            return transform_8bit_n_channel(2)
        elif special == 'compressed':
            return lambda x: downsample_group_stack(transform_8bit_n_channel(2)(x))
    elif task in ['segment_semantic']:  # this is stored as 1 channel image (H,W) where each pixel value is a different class
        return transform_dense_labels
    elif len([t for t in FEED_FORWARD_TASKS if t in task]) > 0:
        return torch.Tensor
    elif 'decoding' in task:
        return transform_16bit_n_channel(TASKS_TO_CHANNELS[task.replace('_decoding', '')])
    elif 'encoding' in task:
        return torch.Tensor
    else:
        raise NotImplementedError("Unknown transform for task {}".format(task))

def downsample_group_stack(img):
    # (k, 256, 256) -> (k, 32, 32) -> (4*k, 16, 16)
    no_batch = False
    if len(img.shape) == 3:
        no_batch = True
        img = img.unsqueeze(dim=0)
    assert len(img.shape) == 4
    img = shrinker(img)
    assert img.shape[2] == img.shape[3] == 32
    img = F.unfold(img, kernel_size=2, stride=2).view(img.shape[0],-1,16,16)
    #     img = F.unfold(img, kernel_size=2, stride=2).view(1,-1,8,8)
    img = img[:,:8,:,:]  # keep only first 8 channels
    if no_batch:
        img = img.squeeze()
    return img

transform_dense_labels = lambda img: torch.Tensor(np.array(img)).long()  # avoids normalizing

transform_8bit = transforms.Compose([
        transforms.ToTensor(),
        MAKE_RESCALE_0_1_NEG1_POS1(3),
    ])
    
def transform_8bit_n_channel(n_channel=1):
    crop_channels = lambda x: x[:n_channel] if x.shape[0] > n_channel else x
    return transforms.Compose([
            transforms.ToTensor(),
            crop_channels,
            MAKE_RESCALE_0_1_NEG1_POS1(n_channel),
        ])

def transform_16bit_single_channel(im):
    im = transforms.ToTensor()(im)
    im = im.float() / (2 ** 16 - 1.0) 
    return RESCALE_0_1_NEG1_POS1(im)

def transform_16bit_n_channel(n_channel=1):
    if n_channel == 1:
        return transform_16bit_single_channel # PyTorch handles these differently
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            MAKE_RESCALE_0_1_NEG1_POS1(n_channel),
        ])


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return  accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    if '.npy' in path:
        return np.load(path)
    elif '.json' in path:
        raise NotImplementedError("Not sure how to load files of type: {}".format(os.path.basename(path)))
    else:
        from torchvision import get_image_backend
        if get_image_backend() == 'accimage':
            im = accimage_loader(path)
        else:
            im = pil_loader(path)
        return im

def pil_loader(path):
  # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
  with open(path, 'rb') as f:
    img = Image.open(f)
    return img.convert(img.mode)

