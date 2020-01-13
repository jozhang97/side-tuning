from multiprocessing import Pool
import numpy as np
import os
from PIL import Image
import time
from tqdm import tqdm
import warnings

from tlkit.data.splits import taskonomy_no_midlevel as split_taskonomy_no_midlevel
from tlkit.utils import np_to_pil, pil_to_np
from tnt.torchnet.meter.valuesummarymeter import ValueSummaryMeter 
from tnt.torchnet.meter.medianimagemeter import MedianImageMeter 

BASE_DIR = '/mnt/data'
task = 'normal'  # TODO only works for pix tasks, impl for logits (e.g. class_object)
#building_dir = os.path.join(BASE_DIR, task, 'allensville')

# Get image paths for all images for all splits 
split_to_images = {}
for split, split_data in split_taskonomy_no_midlevel.items():
    data_paths = [os.path.join(BASE_DIR, task, building) for building in split_data['train']]
    img_paths = []
    for data_dir in data_paths:
        if os.path.exists(data_dir):
            img_paths.extend([os.path.join(data_dir, fn) for fn in os.listdir(data_dir)])
        else:
            warnings.warn(f'{data_dir} is missing.')
    split_to_images[split] = img_paths
    print(split, len(split_data['train']), len(img_paths))


chunk = lambda l, n: [l[i:i+n] for i in range(0, len(l), max(1,n))]
def compute_optimal_imgs(img_paths, use_pool=False):
    median_time, mean_time, pil_time = 0, 0, 0
    img_paths = [path for path in img_paths if '.png' in path]
    mean_meter = ValueSummaryMeter()
    median_meter = MedianImageMeter(bit_depth=8, im_shape=(256, 256, 3), device='cuda')
    p = Pool(6)
    for img_paths_chunk in tqdm(chunk(img_paths, 64)):
        t0 = time.time()
        if use_pool:
            imgs = p.map(Image.open, img_paths_chunk)
        else:
            imgs = [Image.open(img_path) for img_path in img_paths_chunk]
        t1 = time.time()
        for img in imgs:
            median_meter.add(pil_to_np(img)) # keep at uint8 - median wants discrete numbers
        t2 = time.time()
        for img in imgs:
            mean_meter.add(pil_to_np(img).astype(np.float32))  # convert to float - mean requires compute
            img.close()
        t3 = time.time()
        median_time += t2 - t1
        mean_time += t3 - t2
        pil_time += t1 - t0
    p.close()
    print('median', median_time, 'mean', mean_time, 'pil', pil_time)
    return np_to_pil(mean_meter.value()[0]), np_to_pil(median_meter.value())

for split, img_paths in split_to_images.items():
    print(f'starting {split}')
    mean, median = compute_optimal_imgs(img_paths, use_pool=True)
    mean.save(os.path.join(BASE_DIR, task, f'mean_{split}.png'))
    median.save(os.path.join(BASE_DIR, task, f'median_{split}.png'))
