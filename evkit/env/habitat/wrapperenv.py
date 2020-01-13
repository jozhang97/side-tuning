import copy
import numpy as np
import habitat
import torch

class HabitatPreprocessVectorEnv(habitat.VectorEnv):
    def __init__(
        self,
        make_env_fn,
        env_fn_args,
        preprocessing_fn=None,
        auto_reset_done: bool = True,
        multiprocessing_start_method: str = "forkserver",
    ):
        super().__init__(make_env_fn, env_fn_args, auto_reset_done, multiprocessing_start_method)
        obs_space = self.observation_spaces[0]

        # Preprocessing
        self.transform = None
        if preprocessing_fn is not None:
            #             preprocessing_fn = eval(preprocessing_fn)
            self.transform, obs_space = preprocessing_fn(obs_space)

        for i in range(self.num_envs):
            self.observation_spaces[i] = obs_space

        self.collate_obs_before_transform = False
        self.keys = []
        shapes, dtypes = {}, {}

        for key, box in obs_space.spaces.items():
            shapes[key] = box.shape
            dtypes[key] = box.dtype
            self.keys.append(key)

        self.buf_obs = { k: np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k]) for k in self.keys }
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_rews  = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]

    def reset(self):
        observation_list = super().reset()
        if self.collate_obs_before_transform:
            self._save_init_obs(observation_list)
            if self.transform is not None:
                obs = self.transform(self.buf_init_obs)
            self._save_all_obs(obs)
        else:
            for e, obs in enumerate(observation_list):
                if self.transform is not None:
                    obs = self.transform(obs)
                self._save_obs(e, obs)
        return self._obs_from_buf()

    def step(self, action):
        results_list = super().step(action)
        for e, result in enumerate(results_list):
            self.buf_rews[e] = result[1]
            self.buf_dones[e] = result[2]
            self.buf_infos[e] = result[3]

        if self.collate_obs_before_transform:
            self._save_init_obs([r[0] for r in results_list])
            if self.transform is not None:
                obs = self.transform(self.buf_init_obs)
            self._save_all_obs(obs)
        else:
            for e, (obs, _, _, _) in enumerate(results_list):
                if self.transform is not None:
                    obs = self.transform(obs)
                self._save_obs(e, obs)

        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), self.buf_infos.copy())

    def _save_init_obs(self, all_obs):
        self.buf_init_obs = {}
        for k in all_obs[0].keys():
            if k is None:
                self.buf_init_obs[k] = torch.stack([torch.Tensor(o) for o in all_obs])
            else:
                self.buf_init_obs[k] = torch.stack([torch.Tensor(o[k]) for o in all_obs])

    def _save_obs(self, e, obs):
        try:
            for k in self.keys:
                if k is None:
                    self.buf_obs[k][e] = obs
                else:
                    self.buf_obs[k][e] = obs[k]
        except Exception as e:
            print(k, e)
            raise e

    def _save_all_obs(self, obs):
        for k in self.keys:
            if k is None:
                self.buf_obs[k] = obs
            else:
                self.buf_obs[k] = obs[k]

    def _obs_from_buf(self):
        if self.keys==[None]:
            return self.buf_obs[None]
        else:
            return self.buf_obs



class PreprocessEnv(habitat.RLEnv):
    # single env, mostly for debugging
    def __init__(self, env, preprocessing_fn=None):
        self.env = env

        # Preprocessing
        self.transform = None
        self.observation_space = self.env.observation_space
        if preprocessing_fn is not None:
            self.transform, self.observation_space = preprocessing_fn(self.env.observation_space)

    def reset(self):
        self.done = False
        obs = self.env.reset()
        obs = copy.deepcopy(obs)  # when no loop episodes, the action is stored and returned
        if self.transform is not None:
            obs = self.transform(obs)
        return self.wrap(obs)

    def step(self, action):
        action = action[0] if isinstance(action, list) else action
        obs, reward, self.done, info = self.env.step(action)
        obs = copy.deepcopy(obs)  # when no loop episodes, the action is stored and returned
        if self.transform is not None:
            obs = self.transform(obs)
        return self.wrap(obs), np.array([reward], dtype=np.float32), np.array([self.done]), [info]

    def wrap(self, x):
        assert isinstance(x, dict)
        for k, v in x.items():
            if isinstance(v, torch.Tensor):
                x[k] = v.unsqueeze(0)
            elif isinstance(v, np.ndarray):
                x[k] = np.expand_dims(v, axis=0)
            elif isinstance(v, list):
                x[k] = [x[k]]
            else:
                print(f'Habitat Single Env Wrapper: not wrapping {k}')
        return x

    def close(self):
        self.env.close()