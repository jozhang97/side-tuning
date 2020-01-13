from evkit.env.habitat.utils import draw_top_down_map
from evkit.env.util.occupancy_map import OccupancyMap
from evkit.env.habitat.utils import gray_to_rgb
from gym import spaces
import habitat
from habitat.sims.habitat_simulator import SimulatorActions
import numpy as np
from PIL import Image

def transform_target(target):
    r = target[0]
    theta = target[1]
    return np.array([np.cos(theta), np.sin(theta), r])

def transform_observations(observations, target_dim=16, omap=None):
    new_obs = observations
    new_obs["rgb_filled"] = observations["rgb"]
    new_obs["taskonomy"] = observations["rgb"]
    new_obs["target"] = np.moveaxis(np.tile(transform_target(observations["pointgoal"]), (target_dim,target_dim,1)), -1, 0)
    if omap is not None:
        new_obs['map'] = omap.construct_occupancy_map()
        new_obs['global_pos'] = omap.get_current_global_xy_pos()
    del new_obs['rgb']
    return new_obs

def get_obs_space(image_dim=256, target_dim=16, map_dim=None, use_depth=False):
    prep_dict = {
        "taskonomy": spaces.Box(low=0, high=255, shape=(image_dim, image_dim, 3), dtype=np.uint8),
        "rgb_filled": spaces.Box(low=0., high=255., shape=(image_dim, image_dim, 3), dtype=np.uint8),
        "target": spaces.Box(low=-np.inf, high=np.inf, shape=(3, target_dim, target_dim), dtype=np.float32),
        'pointgoal': spaces.Box(low=-np.inf, high=np.inf, shape=(2,) , dtype=np.float32),
    }
    if map_dim is not None:
        prep_dict['map'] = spaces.Box(low=0., high=255., shape=(map_dim, map_dim, 3), dtype=np.uint8)
        prep_dict['global_pos'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,) , dtype=np.float32)

    if use_depth:
        prep_dict['depth'] = spaces.Box(low=0., high=1.0, shape=(image_dim, image_dim, 3), dtype=np.float32)

    return spaces.Dict(prep_dict)


class NavRLEnv(habitat.RLEnv):
    def __init__(self, config_env, config_baseline, dataset):
        config_env.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")  # for top down view
        config_env.TASK.SENSORS.append("HEADING_SENSOR") # for top down view
        self._config_env = config_env.TASK
        self._config_baseline = config_baseline
        self._previous_target_distance = None
        self._previous_action = None
        self._episode_distance_covered = None
        super().__init__(config_env, dataset)

    def reset(self):
        self._previous_action = None

        observations = super().reset()

        self._previous_target_distance = self.habitat_env.current_episode.info[
            "geodesic_distance"
        ]
        return observations

    def step(self, action):
        if self._distance_target() < self._config_env.SUCCESS_DISTANCE:
            action = SimulatorActions.STOP.value

        self._previous_action = action
        obs = super().step(action)
        return obs

    def get_reward_range(self):
        return (
            self._config_baseline.BASELINE.RL.SLACK_REWARD - 1.0,
            self._config_baseline.BASELINE.RL.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = self._config_baseline.BASELINE.RL.SLACK_REWARD

        current_target_distance = self._distance_target()
        reward += self._previous_target_distance - current_target_distance
        self._previous_target_distance = current_target_distance

        if self._episode_success():
            reward += self._config_baseline.BASELINE.RL.SUCCESS_REWARD

        return reward

    def _distance_target(self):
        current_position = self._env.sim.get_agent_state().position.tolist()
        target_position = self._env.current_episode.goals[0].position
        distance = self._env.sim.geodesic_distance(
            current_position, target_position
        )
        return distance

    def _episode_success(self):
        if (
            self._previous_action == SimulatorActions.STOP.value
            and self._distance_target() < self._config_env.SUCCESS_DISTANCE
        ):
            return True
        return False

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        info = self.habitat_env.get_metrics()
        if self.get_done(observations):
            info['success'] = np.ceil(info['spl'])
            info['episode_info'] = {
                'geodesic_distance': self._env.current_episode.info['geodesic_distance'],
                'scene_id': self._env.current_episode.scene_id,
                'episode_id': self._env.current_episode.episode_id,
            }
        #             info["spl"] = self.habitat_env.get_metrics()["spl"]
        return info

class MidlevelNavRLEnv(NavRLEnv):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, config_env, config_baseline, dataset, target_dim=7, map_kwargs={}, reward_kwargs={},
                 loop_episodes=True, scenario_kwargs={}):
        if scenario_kwargs['use_depth']:
            config_env.SIMULATOR.AGENT_0.SENSORS.append("DEPTH_SENSOR")
        super().__init__(config_env, config_baseline, dataset)
        self.target_dim = target_dim
        self.image_dim = 256

        self.use_map = map_kwargs['map_building_size'] > 0
        self.map_dim = 84 if self.use_map else None
        self.map_kwargs = map_kwargs
        self.reward_kwargs = reward_kwargs
        self.scenario_kwargs = scenario_kwargs
        self.last_map = None  # TODO unused

        self.observation_space = get_obs_space(self.image_dim, self.target_dim, self.map_dim, scenario_kwargs['use_depth'])

        self.omap = None
        if self.use_map:
            self.omap = OccupancyMap(map_kwargs=map_kwargs)  # this one is not used

        self.loop_episodes = loop_episodes
        self.n_episodes_completed = 0

    def get_reward(self, observations):
        reward = self.reward_kwargs['slack_reward']

        if not self.reward_kwargs['sparse']:
            current_target_distance = self._distance_target()
            reward += (self._previous_target_distance - current_target_distance) * self.reward_kwargs['dist_coef']
            self._previous_target_distance = current_target_distance

            if self.reward_kwargs['use_visit_penalty'] and len(self.omap.history) > 5:
                reward += self.reward_kwargs['visit_penalty_coef'] * self.omap.compute_eps_ball_ratio(self.reward_kwargs['penalty_eps'])

        if self._episode_success():
            reward += self.reward_kwargs['success_reward']

        return reward

    def reset(self):
        self.obs = self._reset()
        return self.obs

    def _reset(self):
        self.info = None
        self.obs = super().reset()
        if self.use_map:
            self.omap = OccupancyMap(initial_pg=self.obs['pointgoal'], map_kwargs=self.map_kwargs)
        self.obs = transform_observations(self.obs, target_dim=self.target_dim, omap=self.omap)
        if 'map' in self.obs:
            self.last_map = self.obs['map']
        return self.obs

    def step(self, action):

        if self.n_episodes_completed >= len(self.episodes) and not self.loop_episodes:
            return self.obs, 0.0, False, self.info  # noop forever

        self.obs, reward, done, self.info = super().step(action)
        if self.use_map:
            self.omap.add_pointgoal(self.obs['pointgoal'])  # s_{t+1}
            self.omap.step(action)  # a_t our forward model needs to see how the env changed due to the action (via the pg)
        self.obs = transform_observations(self.obs, target_dim=self.target_dim, omap=self.omap)
        if 'map' in self.obs:
            self.last_map = self.obs['map']

        if done:
            self.n_episodes_completed += 1

        return self.obs, reward, done, self.info

    def render(self, mode='human'):
        if mode == 'rgb_array':
            im = self.obs["rgb_filled"]
            to_concat = [im]


            if 'depth' in self.obs:
                depth_im = gray_to_rgb(self.obs['depth'] * 255).astype(np.uint8)
                to_concat.append(depth_im)

            # Get the birds eye view of the agent
            if self.info is not None:
                top_down_map = draw_top_down_map(
                    self.info, self.obs["heading"], im.shape[0]
                )
                top_down_map = np.array(Image.fromarray(top_down_map).resize((256,256)))
            else:
                top_down_map = np.zeros((256,256,3), dtype=np.uint8)
            to_concat.append(top_down_map)

            if 'map' in self.obs:
                occupancy_map = np.copy(self.obs['map'])  # NEED TO COPY OR THIS IS PASS BY REFERENCE
                h,w,_ = occupancy_map.shape
                occupancy_map[int(h//2), int(w//2), 2] = 255   # for debugging
                occupancy_map = np.array(Image.fromarray(occupancy_map).resize((256,256)))
                to_concat.append(occupancy_map)

            output_im = np.concatenate(to_concat, axis=1)
            return output_im
        else:
            super().render(mode=mode)

