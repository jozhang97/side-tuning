import os, sys
import collections
import pickle
import gc
from PIL import Image, ImageDraw
import gc
import torch
import torch.nn as nn
import subprocess
import re
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

task_mapping = {
 'autoencoder': 'autoencoding',
 'colorization': 'colorization',
 'curvature': 'curvature',
 'denoise': 'denoising',
 'edge2d':'edge_texture',
 'edge3d': 'edge_occlusion',
 'ego_motion': 'egomotion', 
 'fix_pose': 'fixated_pose', 
 'jigsaw': 'jigsaw',
 'keypoint2d': 'keypoints2d',
 'keypoint3d': 'keypoints3d',
 'non_fixated_pose': 'nonfixated_pose',
 'point_match': 'point_matching', 
 'reshade': 'reshading',
 'rgb2depth': 'depth_zbuffer',
 'rgb2mist': 'depth_euclidean',
 'rgb2sfnorm': 'normal',
 'room_layout': 'room_layout',
 'segment25d': 'segment_unsup25d',
 'segment2d': 'segment_unsup2d',
 'segmentsemantic': 'segment_semantic',
 'class_1000': 'class_object',
 'class_places': 'class_scene',
 'inpainting_whole': 'inpainting',
 'vanishing_point': 'vanishing_point'
}


CHANNELS_TO_TASKS = {
    1: ['colorization', 'edge_texture', 'edge_occlusion',  'keypoints3d', 'keypoints2d', 'reshading', 'depth_zbuffer', 'depth_euclidean', ],
    2: ['curvature', 'principal_curvature'],
    3: ['autoencoding', 'denoising', 'normal', 'inpainting', 'rgb', 'normals'],
    63: ['class_scene'],
    128: ['segment_unsup2d', 'segment_unsup25d'],
    1000: ['class_object'],
    None: ['segment_semantic']
}

PIX_TO_PIX_TASKS = ['colorization', 'edge_texture', 'edge_occlusion',  'keypoints3d', 'keypoints2d', 'reshading', 'depth_zbuffer', 'depth_euclidean', 'curvature', 'autoencoding', 'denoising', 'normal', 'inpainting', 'segment_unsup2d', 'segment_unsup25d', 'segment_semantic', ]
FEED_FORWARD_TASKS = ['class_object', 'class_scene', 'room_layout', 'vanishing_point']
SINGLE_IMAGE_TASKS = PIX_TO_PIX_TASKS + FEED_FORWARD_TASKS
SIAMESE_TASKS = ['fix_pose', 'jigsaw', 'ego_motion', 'point_match', 'non_fixated_pose']


TASKS_TO_CHANNELS = {}
for n, tasks in CHANNELS_TO_TASKS.items():
    for task in tasks:
        TASKS_TO_CHANNELS[task] = n

LIST_OF_OLD_TASKS = sorted(list(task_mapping.keys()))
LIST_OF_TASKS = sorted(list(task_mapping.values()))

def get_output_sizes():  # really only need to run this once to populate the lists above
    base_path = '/root/tlkit/tlkit/taskonomy_data/'
    decoder_paths = [os.path.join(base_path, f'{task}_decoder.dat') for task in LIST_OF_TASKS]
    decoder_state_dicts = [torch.load(path) for path in decoder_paths]
    output_sizes = [decoder['state_dict']['decoder_output.0.bias'].numpy().size for decoder in decoder_state_dicts]
    print(list(zip(LIST_OF_TASKS, output_sizes)))

# get_output_sizes()

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def update(d, u):  # we need a deep dictionary update
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def flatten(d, parent_key='', sep='.'):  # flattens dictionary
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def var_to_numpy(encoding):
    encoding = encoding.detach().cpu().numpy()
    return encoding


def checkpoint_name(checkpoint_dir, epoch='latest'):
    return os.path.join(checkpoint_dir, 'ckpt-{}.dat'.format(epoch))


def save_checkpoint(obj, directory, step_num):
    os.makedirs(directory, exist_ok=True)
    torch.save(obj, checkpoint_name(directory))
    subprocess.call('cp {} {} &'.format(
        checkpoint_name(directory),
        checkpoint_name(directory, step_num)),
        shell=True)


def get_parent_dirname(path):
    return os.path.basename(os.path.dirname(path))

def get_subdir(training_directory, subdir_name):
    """
    look through all files/directories in training_directory
    return all files/subdirectories whose basename have subdir_name
    if 0, return none
    if 1, return it
    if more, return list of them

    e.g. training_directory: '/path/to/exp'
         subdir_name: 'checkpoints' (directory)
         subdir_name: 'rewards' (files)
    """
    training_directory = training_directory.strip()
    subdirectories = os.listdir(training_directory)
    special_subdirs = []

    for subdir in subdirectories:
        if subdir_name in subdir:
            special_subdir = os.path.join(training_directory, subdir)
            special_subdirs.append(special_subdir)

    if len(special_subdirs) == 0:
        return None
    elif len(special_subdirs) == 1:
        return special_subdirs[0]
    return special_subdirs

def read_pkl(pkl_name):
    with open(pkl_name, 'rb') as f:
        data = pickle.load(f)
    return data

def get_number(name):
    """
    use regex to get the first integer in the name
    if none exists, return -1
    """
    try:
        num = int(re.findall("[0-9]+", name)[0])
    except:
        num = -1
    return num


def unused_dir_name(output_dir):
    """
    Returns a unique (not taken) output_directory name with similar structure to existing one
    Specifically,
    if dir is not taken, return itself
    if dir is taken, return a new name where
        if dir = base + number, then newdir = base + {number+1}
        ow: newdir = base1
    e.g. if output_dir = '/eval/'
         if empty: return '/eval/'
         if '/eval/' exists: return '/eval1/'
         if '/eval/' and '/eval1/' exists, return '/eval2/'

    """
    existing_output_paths = []
    if os.path.exists(output_dir):
        if os.path.basename(output_dir) == '':
            output_dir = os.path.dirname(output_dir)  # get rid of end slash
        dirname = os.path.dirname(output_dir)
        base_name_prefix = re.sub('\d+$', '', os.path.basename(output_dir))

        existing_output_paths = get_subdir(dirname, base_name_prefix)
        assert existing_output_paths is not None, f'Bug, cannot find output_dir {output_dir}'
        if not isinstance(existing_output_paths, list):
            existing_output_paths = [existing_output_paths]
        numbers = [get_number(os.path.basename(path)[-5:]) for path in existing_output_paths]
        eval_num = max(max(numbers), 0) + 1

        output_dir = os.path.join(dirname, f'{base_name_prefix}{eval_num}', '')
        print('New output dir', output_dir)

    return output_dir, existing_output_paths

def index_to_image(idxs: torch.Tensor, dictionary: np.ndarray, img_size):
    # for object classification, converts labels to corresponding labels
    imgs = []
    for inst_top5 in dictionary[idxs]:
        inst_top5 = [w.split(' ', 1)[1] for w in inst_top5]
        to_print = 'Top 5 predictions: \n ' + ' '.join([f'{w} \n' for w in inst_top5])
        img = Image.new('RGB', (img_size, img_size), (255, 255, 255))
        d = ImageDraw.Draw(img)
        d.text((20, 5), to_print, fill=(255, 0, 0))
        imgs.append(np.array(img))

    # # this method will do entire batch at once but its hard to match label to image
    # # above method is slower but if it is done once per epoch it cannot be too bad
    # words = dictionary[idxs].flatten()
    # to_print = ' '.join([f"{t.split(' ', 1)[1].split(',')[0]} \n " for t in words])
    # img = Image.new('RGB', (img_size, img_size * batch_size), (255, 255, 255))
    # d = ImageDraw.Draw(img)
    # d.text((20, 5), to_print, fill=(255, 0, 0))
    # img_np = np.array(img)
    # imgs = [img_np[img_size * i: img_size * (i + 1), : , : ] for i in range(batch_size)]
    ret = np.transpose(np.stack(imgs), (0,3,1,2)).astype(np.float32)
    ret -= 127.5
    ret /= 127.5
    return torch.Tensor(ret)

# util to help move between datatypes 
def pil_to_np(img): # np is dtype=uint8
    img_arr = np.frombuffer(img.tobytes(), dtype=np.uint8)
    img_arr = img_arr.reshape((img.size[1], img.size[0], 3))
    return img_arr

def np_to_pil(img_arr):
    return Image.fromarray(img_arr.astype(np.uint8))

def count_open():
    tensor_count = {}
    var_count = {}
    np_count = {}
    for obj in gc.get_objects():
        try:
            if isinstance(obj, np.ndarray):
                if obj.shape in np_count:
                    np_count[obj.shape] += 1
                else:
                    np_count[obj.shape] = 1

            if torch.is_tensor(obj):
                if obj.size() in tensor_count:
                    tensor_count[obj.size()] += 1
                else:
                    tensor_count[obj.size()] = 1

            if hasattr(obj, 'data') and torch.is_tensor(obj.data):
                if obj.size() in tensor_count:
                    var_count[obj.size()] += 1
                else:
                    var_count[obj.size()] = 1
        except:
            pass
    biggest_hitters = sorted(list(tensor_count.items()), key=lambda x: x[1])[-3:]
    biggest_hitters = biggest_hitters[::-1]
    print('Most frequent tensor shape:', biggest_hitters)

    biggest_hitters = sorted(list(np_count.items()), key=lambda x: x[1])[-3:]
    biggest_hitters = biggest_hitters[::-1]
    print('Most frequent numpy array shape:', biggest_hitters)
    return biggest_hitters


def process_batch_tuple(batch_tuple, task_idx, cfg):
    batch_tuple = [x.to(device, non_blocking=True) for x in batch_tuple]
    if task_idx is None:
        sources = cfg['training']['sources']
        targets = cfg['training']['targets']
    else:
        sources = cfg['training']['sources'][task_idx]
        targets = cfg['training']['targets'][task_idx]
    x = batch_tuple[:len(sources)]
    if len(sources) == 1:
        x = x[0]

    if cfg['training']['sources_as_dict']:
        x = dict(zip(sources, x))
    if cfg['training']['suppress_target_and_use_annotator']:
        labels = [ cfg['training']['annotator'](x) ]
    else:
        labels = batch_tuple[len(sources):len(sources)+len(targets)]

    if (isinstance(cfg['training']['use_masks'], list) and cfg['training']['use_masks'][task_idx]) or \
       (isinstance(cfg['training']['use_masks'], bool) and cfg['training']['use_masks']):
        masks = batch_tuple[-1]
    else:
        masks = None

    assert len(targets) == 1, "Transferring is only supported for one target task"
    label = labels[0]
    return x, label, masks

# from tlkit.models.superposition import HashBasicBlock
def forward_sequential(x, layers, task_idx):
    # if isinstance(layers, HashBasicBlock):
    #     x = layers(x, task_idx)
    if isinstance(layers, nn.Sequential) or isinstance(layers, list) or isinstance(layers, nn.ModuleList):
        for layer in layers:
            try:
                x = layer(x, task_idx)
            except TypeError:
                x = layer(x)
    else:
        try:
            x = layers(x, task_idx)
        except TypeError:
            x = layers(x)
    return x

def load_state_dict_from_path(model, path):
    checkpoint = torch.load(path)
    if 'state_dict' in checkpoint.keys():
        if any(['module' in k for k in checkpoint['state_dict']]):
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items()}
        else:
            state_dict = checkpoint['state_dict']

        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print(f'{e}, reloaded with strict=False \n')
            incompatible = model.load_state_dict(state_dict, strict=False)
            if incompatible is not None:
                  print(f'Num matches: {len([k for k in model.state_dict() if k in state_dict])}\n'
                        f'Num missing: {len(incompatible.missing_keys)} \n'
                        f'Num unexpected: {len(incompatible.unexpected_keys)}')
    else:
        model.load_state_dict(checkpoint)
    return model, checkpoint

if __name__ == '__main__':
    print("See this")
    with HiddenPrints():
        print('donot see this')
    print("See this")
