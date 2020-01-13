# imitation learning, visual navigation
from evkit.models.taskonomy_network import task_mapping
import os
from PIL import Image
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tlkit.data.img_transforms import MAKE_RESCALE_0_1_NEG1_POS1
import numpy as np
from tqdm import tqdm

KEYS =['rgb_filled', 'taskonomy', 'map', 'target', 'global_pos', 'action']
KEYS.extend([f'taskonomy_{task}' for task in task_mapping.values()])


class ExpertData(data.Dataset):
    def __init__(self, data_path, keys, num_frames, split='train', transform: dict = {}, load_to_mem=False,
                 remove_last_step_in_traj=True, removed_actions=[]):
        """
        data expected format
            /path/to/data/
                scenek/
                    trajj/
                        rgb_1.png
                        map_1.npz
                        action_1.npz
                        ...
                        rgb_24.png
                        map_24.npz
                        action_24.npz
        """
        if not os.path.isdir(data_path):
            assert "bad directory"

        # write down all the data paths
        self.keys = keys
        self.urls = {k: [] for k in self.keys}
        print(f'Loading {split} data')
        for scene in tqdm(sorted(os.listdir(os.path.join(data_path, split)))):
            for traj in sorted(os.listdir(os.path.join(data_path, split, scene))):
                for step in sorted(os.listdir(os.path.join(data_path, split, scene, traj))):
                    path = os.path.join(data_path, split, scene, traj, step)
                    key = [k for k in self.keys if k in path]
                    if len(key) != 1:
                        continue
                    self.urls[key[0]].append(path)
                if remove_last_step_in_traj:
                    for k in self.keys:
                        self.urls[k].pop()  # remove the stop action, which is the last one in the sequence

        lens = [len(v) for k, v in self.urls.items()]
        assert max(lens) == min(lens), f'should have same number of each key: {keys} with len f{lens}'

        self.load_to_mem = load_to_mem
        if self.load_to_mem:
            print('Loading trajectories to memory')
            self.cached_data = {}
            for k, objs in self.urls.items():
                if 'rgb' in k:
                    self.cached_data[k] = [np.asarray(Image.open(obj)) for obj in objs]
                else:
                    self.cached_data[k] = [np.load(obj) for obj in objs]

        self.num_frames = num_frames
        self.transform = transform
        for k in self.transform.keys():
            assert k in self.keys, f'transform {k} not in keys {self.keys}'
        self.removed_actions = removed_actions

    def __len__(self):
        return len(self.urls[self.keys[0]])

    def __getitem__(self, index):
        episode_num = self._episode_num(index)
        ret = [[] for _ in self.keys]

        # stack previously seen frames. ret = [o_{t}, ..., o_{t-N}]
        for i in range(self.num_frames):
            if episode_num == self._episode_num(index - i):
                for key_idx, data in enumerate(self._get_index(index - i)):
                    ret[key_idx].append(data)
            else:
                for key_idx in range(len(self.keys)):
                    ret[key_idx].append(np.zeros_like(ret[key_idx][0]))

        for i in range(len(self.keys)):
            if i == self.keys.index('action'):
                ret[i] = ret[i][0]  # pick only the last action - do not frame stack
                if isinstance(ret[i], list) or (isinstance(ret[i], np.ndarray) and len(ret[i].shape) > 0):
                    num_acts = len(ret[i])
                    while np.argmax(ret[i]) in self.removed_actions:  # we do not want to include some actions
                        rand_act = np.zeros(num_acts, dtype=np.uint8)
                        rand_act[np.random.randint(num_acts)] = 1
                        ret[i] = rand_act
                    keep_indices = [i for i in range(num_acts) if i not in self.removed_actions]
                    ret[i] = ret[i][keep_indices]
                else:
                    if ret[i] in self.removed_actions:
                        ret[i] = np.array(np.random.randint(min(self.removed_actions)))  # resample
            else:
                ret[i] = np.concatenate(ret[i][::-1], axis=0)
        return ret

    def _episode_num(self, index):
        return self.urls[self.keys[0]][index].split('/')[-2]

    def _get_index(self, index):
        if self.load_to_mem:
            ret = [self.cached_data[k][index] for k in self.keys()]
        else:
            ret = []
            for k in self.keys:
                path = self.urls[k][index]
                if 'rgb' in k:
                    with open(path, 'rb') as f:
                        img = Image.open(f)
                        img.convert(img.mode)
                        ret.append(img)
                else:
                    ret.append(np.load(path)['arr_0'])

        for k, t in self.transform.items():
            idx = self.keys.index(k)
            ret[idx] = t(ret[idx])
        return ret

def get_dataloaders(data_path,
                    tasks,
                    num_frames,
                    batch_size=64,
                    batch_size_val=4,
                    transform={},
                    num_workers=0,
                    load_to_mem=False,
                    pin_memory=False,
                    remove_last_step_in_traj=True,
                    removed_actions=[]):

    if 'rgb_filled' in tasks:
        transform['rgb_filled'] = transforms.Compose([
            transforms.CenterCrop([256, 256]),
            transforms.Resize(256),
            transforms.ToTensor(),
            MAKE_RESCALE_0_1_NEG1_POS1(3),
        ])
    keys = [t for t in tasks if t in KEYS]
    assert len(keys) == len(tasks), f'unrecognized task in {tasks} not in {KEYS}! cannot be added to Dataset'

    dataloaders = {}
    dataset = ExpertData(data_path, keys=keys, num_frames=num_frames, split='train', transform=transform, load_to_mem=load_to_mem, remove_last_step_in_traj=remove_last_step_in_traj, removed_actions=removed_actions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    dataloaders['train'] = dataloader

    dataset = ExpertData(data_path, keys=keys, num_frames=num_frames, split='val', transform=transform, load_to_mem=load_to_mem, remove_last_step_in_traj=remove_last_step_in_traj, removed_actions=removed_actions)
    dataloader = DataLoader(dataset, batch_size=batch_size_val, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    dataloaders['val'] = dataloader

    dataset = ExpertData(data_path, keys=keys, num_frames=num_frames, split='test', transform=transform, load_to_mem=load_to_mem, remove_last_step_in_traj=remove_last_step_in_traj, removed_actions=removed_actions)
    dataloader = DataLoader(dataset, batch_size=batch_size_val, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    dataloaders['test'] = dataloader
    return dataloaders

if __name__ == '__main__':
    data_path = '/mnt/data/expert_trajs/largeplus'
    keys = ['rgb_filled', 'taskonomy_denoising', 'map', 'target', 'action']
    dataset = ExpertData(data_path, keys=keys, num_frames=4, split='train', transform={}, remove_last_step_in_traj=False)
