# Puts only models (buildings) from the specified split and task into a new directory

import csv
import os
import subprocess
import sys

tar_dir = '/mnt/barn/data/taskonomy_small'

task = ''  # empty string does all
SPLIT = 'tiny'
new_dir = f'/mnt/jeff_data/data/{SPLIT}'
dry_run = False
split_csv = f'/root/perception_module/tlkit/data/splits_taskonomy/train_val_test_{SPLIT}.csv'

with open(split_csv) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    models = [row[0] for row in readCSV][1:]

for model in models:
    source = f'{tar_dir}/{model}_{task}*'
    os.makedirs(new_dir, exist_ok=True)
    cmd = f'rsync -chavP --ignore-existing {source} {new_dir}'
    if dry_run:
        print(cmd)
    else:
        subprocess.call(cmd, shell=True)
