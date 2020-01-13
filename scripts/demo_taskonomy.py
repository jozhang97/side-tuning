import time
from tqdm import tqdm

import torch
import torch.nn.functional as F

from evkit.utils.losses import weighted_l1_loss, weighted_l2_loss, softmax_cross_entropy, dense_cross_entropy
from tlkit.models.lifelong_framework import LifelongSidetuneNetwork
from tlkit.data.datasets.taskonomy_dataset import get_lifelong_dataloaders
import tnt.torchnet as tnt

# Determine what device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Training on device:', device)

# Define what tasks to consider
tasks = ['principal_curvature',  # Note: In the rest of the codebase, this is refered to as 'curvature'
         'segment_semantic', 'reshading', 'keypoints3d', 'keypoints2d',
         'edge_texture', 'edge_occlusion', 'depth_zbuffer',
         'depth_euclidean', 'normal', 'class_object', 'rgb']
task_specific_transfer_kwargs = [
    {'out_channels': 2,    'is_decoder_mlp': False},  # curvature
    {'out_channels': 18,   'is_decoder_mlp': False},  # segment_semantic
    {'out_channels': 1,    'is_decoder_mlp': False},  # reshading
    {'out_channels': 1,    'is_decoder_mlp': False},  # keypoints3d
    {'out_channels': 1,    'is_decoder_mlp': False},  # keypoints2d
    {'out_channels': 1,    'is_decoder_mlp': False},  # edge_texture
    {'out_channels': 1,    'is_decoder_mlp': False},  # edge_occlusion
    {'out_channels': 1,    'is_decoder_mlp': False},  # depth_zbuffer
    {'out_channels': 1,    'is_decoder_mlp': False},  # depth_euclidean
    {'out_channels': 3,    'is_decoder_mlp': False},  # normal
    {'out_channels': 1000, 'is_decoder_mlp': True},   # class_object
    {'out_channels': 3,    'is_decoder_mlp': False},  # rgb
]
loss_fns = [weighted_l2_loss, softmax_cross_entropy, weighted_l1_loss,
            weighted_l1_loss, weighted_l1_loss, weighted_l1_loss,  weighted_l1_loss,  weighted_l1_loss,
            weighted_l1_loss, weighted_l1_loss, dense_cross_entropy, weighted_l1_loss]

# Set up Model
model = LifelongSidetuneNetwork(
    base_class='TaskonomyEncoder',
    base_kwargs={ 'eval_only': True, 'normalize_outputs': False },
    base_weights_path='../side-tuning/side-tuning/assets/pytorch/curvature_encoder.dat',
    side_class='FCN5',
    side_kwargs={ 'eval_only': False, 'normalize_outputs': False },
    side_weights_path='../side-tuning/side-tuning/assets/pytorch/distillation/curvature-distilled.pth',
    task_specific_transfer_kwargs=task_specific_transfer_kwargs,
    transfer_class='PreTransferedDecoder',
    transfer_kwargs={
        'transfer_class': 'TransferConv3',
        'transfer_weights_path': None,
        'transfer_kwargs': {'n_channels': 8, 'residual': True},
        'decoder_class': 'TaskonomyDecoder',
        'decoder_weights_path': None,
        'decoder_kwargs': {'eval_only': False},
    },
    merge_method='merge_operators.Alpha',
    dataset='taskonmy',
    tasks=range(len(tasks)),
)
model.to(device)


# Prepare Dataloaders
dataloaders = get_lifelong_dataloaders(
    data_path='../side-tuning/taskonomy-sample-model-1',
    sources = [['rgb']] * len(tasks),
    targets=[[t] for t in tasks],
    masks=[False] * len(tasks),
    epochs_per_task=3,
    split=None,
    batch_size=16,
    batch_size_val=16,
    num_workers=8,
    max_images_per_task=100,
)
dl_train, dl_val = dataloaders['train'], dataloaders['val']


# Set up optimizer (in general, make sure to set weight decay for alpha to 0!)
optimizer = torch.optim.Adam(
    [
        {'params': [param for name, param in model.named_parameters() if 'merge_operator' in name or 'context' in name or 'alpha' in name], 'weight_decay': 0.0},
        {'params': [param for name, param in model.named_parameters() if 'merge_operator' not in name and 'context' not in name and 'alpha' not in name]},
    ],
    lr=1e-4, weight_decay=2e-6
)


# Set up logging
mlog = tnt.logger.TensorboardMeterLogger(
    env='demo',
    log_dir='/tmp/taskonomy_demo/tensorboards',
    plotstylecombined=True
)
for task in range(len(tasks)):
    mlog.add_meter(f'losses/task_{task}', tnt.meter.ValueSummaryMeter())


# Training loop
print('Starting training')
model.train(True)
start_time = time.time()
seen = set()
for epoch in range(len(tasks)):
    for task_idx, batch_tuple in tqdm(dl_train, desc="Epoch " + str(epoch) + " (Train)"):
        # Prepare for new task
        old_size = len(seen)
        seen.add(task_idx)
        if len(seen) > old_size:
            model.start_task(task_idx, train=True)  # important to stop gradients from flowing to other tasks

        # Process Batch
        batch_tuple = [elm.to(device) for elm in batch_tuple]
        x, label = batch_tuple[0], batch_tuple[1]
        x = F.interpolate(x, 256)
        label = F.interpolate(label, 256) if len(label.shape) == 4 else label

        # Forward
        pred = model(x, task_idx=task_idx)
        label = label[:,:pred.shape[1],:,:] if len(label.shape) == 4 else label   # data may be duplicated
        loss = loss_fns[task_idx](pred, label)['total']

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
        optimizer.step()

        # Log
        mlog.update_meter(loss.item()/x.shape[0], meters={f'losses/task_{task_idx}'}, phase='train')

meter_dict = mlog.peek_meter('train')
print('Finished training in:', time.time() - start_time, 'seconds. \nResults:')
for task_idx in range(len(tasks)):
    print(f'\t Task {task_idx} Train Loss:', meter_dict[f'losses/task_{task_idx}'].item())


# Validate model on held-out data
model.to(device)
model.train(False)
for task_idx, batch_tuple in tqdm(dl_val, desc="Validation"):
    # Process Batch
    batch_tuple = [elm.to(device) for elm in batch_tuple]
    x, label = batch_tuple[0], batch_tuple[1]
    x = F.interpolate(x, 256)
    label = F.interpolate(label, 256) if len(label.shape) == 4 else label

    # Forward
    pred = model(x, task_idx=task_idx)
    label = label[:,:pred.shape[1],:,:] if len(label.shape) == 4 else label   # data may be duplicated
    loss = loss_fns[task_idx](pred, label)['total']

    # Log
    mlog.update_meter(loss.item()/x.shape[0], meters={f'losses/task_{task_idx}'}, phase='val')

meter_dict = mlog.peek_meter('val')
print('Results (Validation): ')
for task_idx in range(len(tasks)):
    print(f'\t Task {task_idx} Val Loss:', meter_dict[f'losses/task_{task_idx}'].item())

# Save checkpoint
print('Saving model to /tmp/taskonomy_demo/model.pth')
torch.save(model.state_dict(), '/tmp/taskonomy_demo/model.pth')