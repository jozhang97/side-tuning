import numpy as np
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from tlkit.data.img_transforms import MAKE_RESCALE_0_1_NEG1_POS1

SPLITS = ['train', 'val', 'test']
t_to_np = lambda x: (np.transpose(x.cpu().numpy(), (1,2,0)) * 255).astype(np.uint8)
show = lambda x: Image.fromarray(t_to_np(x))

rgb_t = transforms.Compose([
    transforms.CenterCrop([256, 256]),
    transforms.Resize(256),
    transforms.ToTensor(),
    MAKE_RESCALE_0_1_NEG1_POS1(3),
])
class Expert():
    def __init__(self, data_dir, compare_with_saved_trajs=False, follower=None):
        self.data_dir = data_dir
        self.compare_with_saved_trajs = compare_with_saved_trajs  # this lets us compare all observations with saved
        self.traj_dir = None
        self.action_idx = 0
        self.same_as_il = True
        self.follower = None
        if follower is not None:
            checkpoint_obj = torch.load(follower)
            start_epoch = checkpoint_obj['epoch']
            print('Loaded imitator (epoch {}) from {}'.format(start_epoch, follower))
            self.follower = checkpoint_obj['model']
            self.follower.eval()


    def reset(self, envs):
        scene_id = envs.env.env.env._env.current_episode.scene_id
        scene_id = scene_id.split('/')[-1].split('.')[0]
        episode_id = envs.env.env.env._env.current_episode.episode_id
        self.traj_dir = self._find_traj_dir(scene_id, episode_id)
        self.action_idx = 0
        self.same_as_il = True
        if self.follower is not None:
            self.follower.reset()

    def _find_traj_dir(self, scene_id, episode_id):
        for split in SPLITS:
            for building in os.listdir(os.path.join(self.data_dir, split)):
                if building == scene_id:
                    for episode in os.listdir(os.path.join(self.data_dir, split, building)):
                        if str(episode_id) in episode:
                            return os.path.join(self.data_dir, split, building, episode)
        assert False, f'Could not find scene {scene_id}, episode {episode_id} in {self.data_dir}'

    def _load_npz(self, fn):
        path = os.path.join(self.traj_dir, fn)
        return np.load(path)['arr_0']

    def _cmp_with_il(self, k, observations, img=False, save_png=False, printer=True):
        if k not in observations:
            if printer:
                print(f'Key {k}: cannot find')
            return 0
        if img:
            il_obj = Image.open(os.path.join(self.traj_dir, f'{k}_{self.action_idx:03d}.png'))
            il_obj = rgb_t(il_obj).cuda()
        else:
            il_obj = torch.Tensor(self._load_npz(f'{k}_{self.action_idx:03d}.npz')).cuda()
        num_channels = observations[k].shape[1] // 4
        rl_obj = observations[k][0][-num_channels:]
        if save_png:
            debug_dir = os.path.join('/mnt/data/debug/', str(self.action_idx))
            os.makedirs(debug_dir, exist_ok=True)
            show(il_obj).save(os.path.join(debug_dir, f'il_{k}.png'))
            show(rl_obj).save(os.path.join(debug_dir, f'rl_{k}.png'))
        diff = torch.sum(il_obj - rl_obj)
        if printer:
            print(f'Key {k}: {diff}')
        return diff

    def _debug(self, observations):
        self._cmp_with_il('map', observations, save_png=self.same_as_il, printer=self.same_as_il)
        self._cmp_with_il('target', observations, printer=self.same_as_il)
        self._cmp_with_il('rgb_filled', observations, img=True, save_png=self.same_as_il, printer=self.same_as_il)
        self._cmp_with_il('taskonomy', observations, printer=self.same_as_il)

    def act(self, observations, states, mask_done, deterministic=True):
        action = self._load_npz(f'action_{self.action_idx:03d}.npz')
        if len(action.shape) == 0:
            action = int(action)
        else:
            action = np.argmax(action)

        if self.compare_with_saved_trajs:
            mapd = self._cmp_with_il('map', observations, printer=False)
            taskonomyd = self._cmp_with_il('taskonomy', observations, printer=False)
            targetd = self._cmp_with_il('target', observations, printer=False)
            if abs(mapd) > 0.1 or abs(taskonomyd) > 0.1 or abs(targetd) > 1e-6:
                print('-' * 50)
                print(f'Running {self.traj_dir} on step {self.action_idx}. expert: {action}, targetd {targetd} mapdif {mapd}, taskdif {taskonomyd}')
                self._debug(observations)
                self.same_as_il = False

        if self.follower is not None:
            _, follower_action, _, _ = self.follower.act(observations, states, mask_done, True)
            follower_action = follower_action.squeeze(1).cpu().numpy()[0]
            if follower_action != action and action != 3:
                print('-' * 50)
                print(f'Running {self.traj_dir} on step {self.action_idx}. expert: {action}, follower: {follower_action}, mapdif {mapd}, taskdif {taskonomyd}')
                self._debug(observations)
                self.same_as_il = False

        if action == 3:
            action = 2
        self.action_idx += 1
        return 0, torch.Tensor([action]).unsqueeze(1), 0, states

    def cuda(self, *inputs, **kwargs):
        pass

    def eval(self, *inputs, **kwargs):
        pass


