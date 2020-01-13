# sample from DATA_SOURCE every SUBSAMPLE_RATE episode and store under /mnt/data/expert_trajs/DATA_SIZE
# take all val episodes

import os
import shutil
import sys
from tqdm import tqdm

DATA_SIZE = sys.argv[1] if len(sys.argv) >= 4 else 'small'
SUBSAMPLE_RATE = eval(sys.argv[2]) if len(sys.argv) >= 4 else 100
DATA_SOURCE = sys.argv[3] if len(sys.argv) >= 4 else 'large'

BASE_DIR = '/mnt/data/expert_trajs'
SOURCE_DIR = os.path.join(BASE_DIR, DATA_SOURCE)
TARGET_DIR = os.path.join(BASE_DIR, DATA_SIZE)
buildings_dir = os.path.join(SOURCE_DIR, 'train')

os.makedirs(os.path.join(BASE_DIR, DATA_SIZE, 'train'), exist_ok=True)

# handle train split
counter = 0
for building in tqdm(os.listdir(buildings_dir)):
    episodes_dir = os.path.join(buildings_dir, building)
    for episode in sorted(os.listdir(episodes_dir)):
        episode_pth = os.path.join(episodes_dir, episode)
        counter += 1
        if counter % SUBSAMPLE_RATE == 0:
            copy_loc = episode_pth.replace(DATA_SOURCE, DATA_SIZE)
            shutil.copytree(episode_pth, copy_loc)

print(f'Copied over {counter//SUBSAMPLE_RATE} training episodes from {DATA_SOURCE} to {DATA_SIZE}')

# handle val/test splits
os.symlink(os.path.join(SOURCE_DIR, 'val'), os.path.join(TARGET_DIR, 'val'))
os.symlink(os.path.join(SOURCE_DIR, 'test'), os.path.join(TARGET_DIR, 'test'))
print(f'Transfered from {counter//SUBSAMPLE_RATE} training episodes from {DATA_SOURCE} to {DATA_SIZE}')
