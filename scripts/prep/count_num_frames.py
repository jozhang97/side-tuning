# count_num_frames in a expert trajectory directory (needs to be /train)
import os
import shutil
import sys
from tqdm import tqdm
import re

try:
    buildings_dir = sys.argv[1]
except:
    buildings_dir = '/mnt/data/expert_trajs/debug/train'

total_frames = 0
n_episodes = 0
for building in tqdm(os.listdir(buildings_dir)):
    episodes_dir = os.path.join(buildings_dir, building)
    for episode in sorted(os.listdir(episodes_dir)):
        episode_pth = os.path.join(episodes_dir, episode)
        last = sorted(os.listdir(episode_pth))[-1]
        num_frames = int(re.findall("[0-9]+", last)[0]) + 1
        total_frames += num_frames
        n_episodes += 1
print(f'In {buildings_dir}, we have a total of {total_frames} frames in {n_episodes} episodes, averaging {total_frames//n_episodes} frames per episode')