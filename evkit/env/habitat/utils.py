import cv2
import numpy as np
import torch
import random
from habitat.sims.habitat_simulator import SimulatorActions
from habitat.utils.visualizations import maps

try:
    from habitat.sims.habitat_simulator import SIM_NAME_TO_ACTION  # backwards support
except:
    pass


# TODO these are action values. Make sure to add the word "action" into the name
FORWARD_VALUE = SimulatorActions.FORWARD.value
FORWARD_VALUE = FORWARD_VALUE if isinstance(FORWARD_VALUE, int) else SIM_NAME_TO_ACTION[FORWARD_VALUE]

STOP_VALUE = SimulatorActions.STOP.value
STOP_VALUE = STOP_VALUE if isinstance(STOP_VALUE, int) else SIM_NAME_TO_ACTION[STOP_VALUE]

LEFT_VALUE = SimulatorActions.LEFT.value
LEFT_VALUE = LEFT_VALUE if isinstance(LEFT_VALUE, int) else SIM_NAME_TO_ACTION[LEFT_VALUE]

RIGHT_VALUE = SimulatorActions.RIGHT.value
RIGHT_VALUE = RIGHT_VALUE if isinstance(RIGHT_VALUE, int) else SIM_NAME_TO_ACTION[RIGHT_VALUE]


TAKEOVER1 = [LEFT_VALUE] * 4 + [FORWARD_VALUE] * 4
TAKEOVER2 = [RIGHT_VALUE] * 4 + [FORWARD_VALUE] * 4
TAKEOVER3 = [LEFT_VALUE] * 6 + [FORWARD_VALUE] * 2
TAKEOVER4 = [RIGHT_VALUE] * 6 + [FORWARD_VALUE] * 2
# TAKEOVER5 = [LEFT_VALUE] * 8  # rotation only seems not to break out of bad behavior
# TAKEOVER6 = [RIGHT_VALUE] * 8
TAKEOVER_ACTION_SEQUENCES = [TAKEOVER1, TAKEOVER2, TAKEOVER3, TAKEOVER4]
TAKEOVER_ACTION_SEQUENCES = [torch.Tensor(t).long() for t in TAKEOVER_ACTION_SEQUENCES]

DEFAULT_TAKEOVER_ACTIONS = torch.Tensor([LEFT_VALUE, LEFT_VALUE, LEFT_VALUE, LEFT_VALUE, FORWARD_VALUE, FORWARD_VALUE]).long()
NON_STOP_VALUES = torch.Tensor([FORWARD_VALUE, LEFT_VALUE, RIGHT_VALUE]).long()


flatten = lambda l: [item for sublist in l for item in sublist]

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def shuffle_episodes(env, swap_every_k=10):
    episodes = env.episodes
    #     buildings_for_epidodes = [e.scene_id for e in episodes]
    episodes = env.episodes = random.sample([c for c in chunks(episodes, swap_every_k)], len(episodes) // swap_every_k)
    env.episodes = flatten(episodes)
    return env.episodes


def draw_top_down_map(info, heading, output_size):
    if info is None:
        return
    top_down_map = maps.colorize_topdown_map(info["top_down_map"]["map"])
    original_map_size = top_down_map.shape[:2]
    map_scale = np.array(
        (1, original_map_size[1] * 1.0 / original_map_size[0])
    )
    new_map_size = np.round(output_size * map_scale).astype(np.int32)
    # OpenCV expects w, h but map size is in h, w
    top_down_map = cv2.resize(top_down_map, (new_map_size[1], new_map_size[0]))

    map_agent_pos = info["top_down_map"]["agent_map_coord"]
    map_agent_pos = np.round(
        map_agent_pos * new_map_size / original_map_size
    ).astype(np.int32)
    top_down_map = maps.draw_agent(
        top_down_map,
        map_agent_pos,
        heading - np.pi / 2,
        agent_radius_px=top_down_map.shape[0] / 40,
        )
    return top_down_map

def gray_to_rgb(img_arr):
    # Input: (H,W,1) or (H,W) or (H,W,3)
    # Output: (H,W,3)
    if len(img_arr.shape) == 3 and img_arr.shape[2] == 3:  # (H,W,3)
        return img_arr
    if len(img_arr.shape) == 3:
        img_arr = img_arr.squeeze(2)  # (H,W,1)
    return np.dstack((img_arr, img_arr, img_arr))