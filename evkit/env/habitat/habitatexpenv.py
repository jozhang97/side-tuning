from gym import spaces
import math
import numpy as np
import habitat
from evkit.env.util.occupancy_map import pos_to_map, rotate
from evkit.env.habitat.utils import draw_top_down_map, gray_to_rgb
from PIL import Image
import torch

class ExplorationRLEnv(habitat.RLEnv):
    # This env has a global map to calculate reward and an agent map to pass as a sensor
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, config_env, config_baseline, dataset, map_kwargs={}, reward_kwargs={},
                 loop_episodes=True, scenario_kwargs={}):
        config_env.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")  # for top down view
        config_env.TASK.SENSORS.append("HEADING_SENSOR") # for top down view
        config_env.SIMULATOR.AGENT_0.SENSORS.append("DEPTH_SENSOR")
        config_env.defrost()
        config_env.ENVIRONMENT.MAX_EPISODE_STEPS = scenario_kwargs['max_episode_steps']
        config_env.freeze()
        self._config_env = config_env.TASK
        self.map_size = map_kwargs['map_size']
        super().__init__(config_env, dataset)

        self.image_dim = 256
        self.loop_episodes = loop_episodes
        self.n_episodes_completed = 0

        # Map
        self.depth_scale = 1.0
        if config_env.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH:
            self.depth_scale = 1.0 * config_env.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
        self.fov = map_kwargs['fov']
        self.min_depth = map_kwargs['min_depth']
        self.max_depth = map_kwargs['max_depth']
        self.pre_map_x_range = np.array(map_kwargs['map_x_range'])
        self.pre_map_y_range = np.array(map_kwargs['map_y_range'])
        self.map_kwargs = map_kwargs
        if not self.map_kwargs['relative_range']:
            self.map_x_range = self.pre_map_x_range
            self.map_y_range = self.pre_map_y_range

        self.max_building_size = max(self.pre_map_y_range[1] - self.pre_map_y_range[0], self.pre_map_x_range[1] - self.pre_map_x_range[0])
        self.cell_size = self.max_building_size / self.map_size

        self.observation_space = spaces.Dict({
            "taskonomy": spaces.Box(low=0, high=255, shape=(self.image_dim, self.image_dim, 3), dtype=np.uint8),
            "rgb_filled": spaces.Box(low=0., high=255., shape=(self.image_dim, self.image_dim, 3), dtype=np.uint8),
            "depth": spaces.Box(low=0., high=config_env.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH / self.depth_scale, shape=(self.image_dim, self.image_dim, 1), dtype=np.float32),
            "map": spaces.Box(low=0., high=255., shape=(self.map_size, self.map_size, 1), dtype=np.uint8),
            "global_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,) , dtype=np.float32),
        })
        self.reward_kwargs = reward_kwargs


    def _transform_observations(self, observations):
        new_obs = observations
        new_obs["rgb_filled"] = observations["rgb"]
        new_obs["taskonomy"] = observations["rgb"]
        del new_obs['rgb']
        self.obs['global_pos'] = self._agent_global_position
        self.obs['map'] = self._construct_agent_map()
        return new_obs

    def reset(self):
        self.obs = super().reset()
        self.info = None

        if self.map_kwargs['relative_range']:
            self.map_x_range = self.pre_map_x_range + int(self._agent_global_position[0])  # puts initial agent at center of map, not sure if good idea
            self.map_y_range = self.pre_map_y_range + int(self._agent_global_position[1])

        self.seen_history = np.array([])  # history of cells seen in global coordinates (you do not see where you are!)
        self.visit_history = np.array([self._agent_global_position[:2]])  # history of cells visited in global coordinates
        self.orig_found = 0
        self.global_map = np.zeros((self.map_size, self.map_size, 1), dtype=np.uint8)  # binary laser vision map

        self._update_global_map(self.obs['depth'])
        self.obs = self._transform_observations(self.obs)
        return self.obs

    def step(self, action):
        if self.n_episodes_completed >= len(self.episodes) and not self.loop_episodes:
            return self.obs, 0.0, False, self.info  # noop forever

        self.obs, reward, done, self.info = super().step(action)
        self.visit_history = np.vstack((self.visit_history, self._agent_global_position[:2]))

        self.orig_found = np.sum(self.global_map)
        self._update_global_map(self.obs['depth'])
        self.obs = self._transform_observations(self.obs)

        if done:
            self.n_episodes_completed += 1
        return self.obs, reward, done, self.info

    def get_reward(self, observations):
        return 0.1 * (np.sum(self.global_map) - self.orig_found) + self.reward_kwargs['slack_reward']

    @property
    def _agent_global_position(self):
        x, z, y = self._env._sim.get_agent_state().position
        return np.array([x, y, self.obs['heading']])

    def _construct_agent_map(self):
        # agent recognizes what cells it has seen
        if self.seen_history.size == 0 and self.visit_history.size == 0:
            return np.zeros((self.map_size, self.map_size, 1), dtype=np.uint8)

        history = np.concatenate((self.seen_history, self.visit_history), axis=0)
        agent_coords = history - self._agent_global_position[:2]
        agent_coords = rotate(agent_coords, -1 * self.obs['heading'])

        visitation_cells = pos_to_map(agent_coords + self.max_building_size / 2, cell_size=self.cell_size)
        omap = torch.full((1, self.map_size, self.map_size), fill_value=128, dtype=torch.uint8, device=None, requires_grad=False)  # Avoid multiplies, stack, and copying to torch
        omap[0][visitation_cells[:, 0], visitation_cells[:, 1]] = 255  # Agent visitation
        omap = omap.permute(1, 2, 0).cpu().numpy()
        assert omap.dtype == np.uint8, f'Omap needs to be uint8, currently {omap.dtype}'
        return omap


    def _update_global_map(self, depth):
        depth = depth * self.depth_scale
        clipped_depth_image = np.clip(depth, self.min_depth, self.max_depth)
        if self.map_kwargs['fullvision']:
            xyz = self._reproject_depth_map(depth.squeeze(2))
            xyz_seen = xyz
        else:
            xyz = self._reproject_depth_map(clipped_depth_image.squeeze(2))
            xyz_seen = xyz[self.image_dim//2:, : , :][:, self.image_dim//2-2 : self.image_dim//2+2, :]
        if len(xyz_seen.shape) == 3:
            xyz_seen = xyz_seen.reshape(xyz_seen.shape[0] * xyz_seen.shape[1], 3)
        xx, yy = self._rotate_origin_only(xyz_seen, np.pi + self.obs['heading'])
        xx = np.append(xx, 0)  # see agent cur location
        yy = np.append(yy, 0)
        xx += self._agent_global_position[0]
        yy += self._agent_global_position[1]
        pts = np.stack((xx, yy)).T
        self.seen_history = np.vstack((self.seen_history, pts)) if self.seen_history.size != 0 else pts
        xx = np.clip((xx - self.map_x_range[0]) / self.cell_size, 0, self.map_size-1).astype(np.uint8)
        yy = np.clip((yy - self.map_y_range[0]) / self.cell_size, 0, self.map_size-1).astype(np.uint8)
        if np.any(xx < 0) or np.any(yy < 0):
            raise ValueError("Trying to set occupancy in negative grid cell")
        self.global_map[xx, yy, 0] = 1

    def _reproject_depth_map(self, depth, unit_scale=1.0):
        rows, cols = depth.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
        y = depth * unit_scale
        x = y * ((c - self.image_dim // 2) / self.fov / (self.image_dim // 2))
        z = y * ((r - self.image_dim // 2) / self.fov / (self.image_dim // 2))
        return np.dstack((x,y,z))

    def _rotate_origin_only(self, xy, radians):  # clockwise
        x, y = xy[:,:2].T
        xx = x * math.cos(radians) + y * math.sin(radians)
        yy = -x * math.sin(radians) + y * math.cos(radians)
        return xx, yy

    def get_reward_range(self):
        return (0, self.map_size * self.map_size)

    def get_done(self, observations):
        return self._env.episode_over

    def _prepare_map_for_render(self, map: np.ndarray) -> np.ndarray:
        gray = len(map.shape) == 3 and map.shape[2] == 1 or len(map.shape) == 2
        if len(map.shape) == 3 and map.shape[2] == 1:
            map = map.squeeze(2)
        map = np.array(Image.fromarray(map * 255).resize((self.image_dim, self.image_dim)))
        if gray:
            map = gray_to_rgb(map)
        return map

    def render(self, mode='human'):
        if mode == 'rgb_array':
            im = self.obs["rgb_filled"]

            # agent map with agent (in the middle)
            debug_layer = np.ones((self.map_size, self.map_size, 1), dtype=np.uint8) * 128 # binary laser vision map
            debug_layer[int(self.map_size/2), int(self.map_size/2)] = 255
            agent_render_map = np.concatenate((self.obs['map'], self.obs['map'], debug_layer), axis=2)
            agent_render_map = self._prepare_map_for_render(agent_render_map)

            # global map with agent
            debug_layer_current = np.zeros((self.map_size, self.map_size, 1), dtype=np.uint8)  # current agent location
            idx_x = np.clip(int((self._agent_global_position[0] - self.map_x_range[0]) / self.cell_size), 0, self.map_size-1)
            idx_y = np.clip(int((self._agent_global_position[1] - self.map_y_range[0]) / self.cell_size), 0, self.map_size-1)
            debug_layer_current[idx_x, idx_y, 0] = 1
            debug_layer_history = np.zeros((self.map_size, self.map_size, 1), dtype=np.uint8)  # visited map (blue by itself, pink if also seen)
            idx_x = np.clip((self.visit_history[:,0] - self.map_x_range[0]) // self.cell_size, 0, self.map_size-1).astype(np.uint8)
            idx_y = np.clip((self.visit_history[:,1] - self.map_y_range[0]) // self.cell_size, 0, self.map_size-1).astype(np.uint8)
            debug_layer_history[idx_x, idx_y, 0] = 1
            # global_map is red
            global_render_map = np.concatenate((self.global_map, debug_layer_current, debug_layer_history), axis=2)
            global_render_map = self._prepare_map_for_render(global_render_map)

            # Get the birds eye view of the agent
            if self.info is None:
                top_down_map = np.zeros((256, 256, 3), dtype=np.uint8)
            else:
                top_down_map = draw_top_down_map(self.info, self.obs["heading"], im.shape[0])
                top_down_map = np.array(Image.fromarray(top_down_map).resize((256,256)))

            to_concat = [im, agent_render_map, global_render_map, top_down_map]
            if 'depth' in self.obs:
                clipped_depth_image = np.clip(self.obs['depth'], self.min_depth / self.depth_scale, self.max_depth / self.depth_scale)  # here we want to keep it normalized
                depth = gray_to_rgb(clipped_depth_image * 255).astype(np.uint8)
                depth[self.image_dim//2:, : , :][:, self.image_dim//2-2 : self.image_dim//2+2, 0] = 255  # points that robot "unlocks"
                to_concat.append(depth)

            output_im = np.concatenate(to_concat, axis=1)
            return output_im
        else:
            super().render(mode=mode)

    def get_info(self, observations):
        info = self.habitat_env.get_metrics()
        if self.get_done(observations):
            info['episode_info'] = {
                'scene_id': self._env.current_episode.scene_id,
                'episode_id': self._env.current_episode.episode_id,
            }
        return info

