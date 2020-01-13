import numpy as np
import time
from tqdm import tqdm

import torch
import torch.nn.functional as F

from tlkit.models.lifelong_framework import LifelongSidetuneNetwork
from tlkit.data.datasets.icifar_dataset import get_dataloaders
import tnt.torchnet as tnt

# Determine what device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Training on device:', device)

# Set up Model
model = LifelongSidetuneNetwork(
    base_class='ResnetiCifar44NoLinear',
    base_kwargs={ 'eval_only': True },
    base_weights_path='../side-tuning/assets/pytorch/resnet44-cifar.pth',
    side_class='FCN4Reshaped',
    side_kwargs={ 'eval_only': False },
    side_weights_path='../side-tuning/assets/pytorch/distillation/fcn4-cifar.pth',
    transfer_class='nn.Linear',
    transfer_kwargs={
        'in_features': 64,
        'out_features': 10,
    },
    merge_method='merge_operators.MLP2',  # Try 'merge_operators.Alpha', 'merge_operators.MLP'
    dataset='icifar',
)
model.to(device)


# Prepare Dataloaders
tasks = ['cifar0-9',   'cifar10-19', 'cifar20-29', 'cifar30-39', 'cifar40-49',
         'cifar50-59', 'cifar60-69', 'cifar70-79', 'cifar80-89', 'cifar90-99']
dataloaders = get_dataloaders(
    targets=[[t] for t in tasks],
    data_path='/tmp/icifar_demo/data',
    epochs_until_cycle=0,
    batch_size=128,
)
dl_train, dl_val = dataloaders['train'], dataloaders['val']


# Set up optimizer (in general, make sure to set weight decay for alpha to 0!)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)


# Set up logging
mlog = tnt.logger.TensorboardMeterLogger(
    env='demo',
    log_dir='/tmp/icifar_demo/tensorboards',
    plotstylecombined=True
)
for task in range(len(tasks)):
    mlog.add_meter(f'losses/task_{task}', tnt.meter.ValueSummaryMeter())
    mlog.add_meter(f'accuracy_top1/task_{task}', tnt.meter.ClassErrorMeter(topk=[1], accuracy=True))


# Training loop
print('Starting training')
model.train(True)
start_time = time.time()
for epoch in range(len(tasks)):
    seen = set()
    for task_idx, (x, label) in tqdm(dl_train, desc="Epoch " + str(epoch) + " (Train)"):
        # Prepare for new task
        old_size = len(seen)
        seen.add(task_idx)
        if len(seen) > old_size:
            model.start_task(task_idx, train=True)  # important to stop gradients from flowing to other tasks

        # Forward
        x, label = x.to(device), label.to(device)
        pred = model(x, task_idx=task_idx)
        loss = F.cross_entropy(pred, label)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
        optimizer.step()

        # Log
        mlog.update_meter(loss.item()/x.shape[0], meters={f'losses/task_{task_idx}'}, phase='train')
        mlog.update_meter(pred, target=label, meters={f'accuracy_top1/task_{task_idx}'}, phase='train')

meter_dict = mlog.peek_meter('train')
print('Finished training in:', time.time() - start_time, 'seconds. \nResults:')
for task_idx in range(len(tasks)):
    print(f'\t Task {task_idx} Train Loss:', meter_dict[f'losses/task_{task_idx}'].item())
    print(f'\t Task {task_idx} Train Accuracy: ', meter_dict[f'accuracy_top1/task_{task_idx}'])


# Validate model on held-out data
model.to(device)
model.train(False)
for task_idx, (x, label) in tqdm(dl_val, desc="Validation"):
    # Forward
    x, label = x.to(device), label.to(device)
    pred = model(x, task_idx=task_idx)
    loss = F.cross_entropy(pred, label)

    # Log
    mlog.update_meter(loss.item()/x.shape[0], meters={f'losses/task_{task_idx}'}, phase='val')
    mlog.update_meter(pred, target=label, meters={f'accuracy_top1/task_{task_idx}'}, phase='val')

meter_dict = mlog.peek_meter('val')
print('Results (Validation): ')
for task_idx in range(len(tasks)):
    print(f'\t Task {task_idx} Val Loss:', meter_dict[f'losses/task_{task_idx}'].item())
    print(f'\t Task {task_idx} Val Accuracy:', meter_dict[f'accuracy_top1/task_{task_idx}'])

avg_acc = np.mean([meter_dict[f'accuracy_top1/task_{task_idx}'] for task_idx in range(len(tasks))])
print('Average Val Accuracy', avg_acc)


# Save checkpoint
print('Saving model to /tmp/icifar_demo/model.pth')
torch.save(model.state_dict(), '/tmp/icifar_demo/model.pth')
