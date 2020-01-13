import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F

from evkit.utils.viz.core import pack_images
import tnt.torchnet as tnt

import warnings

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

unorm = UnNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

def log(mlog, key, val, phase):
    if mlog is not None:
        # logger.add_scalar(tag=key, scalar_value=val, global_step=t)
        if 'image' in key:
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val)
            val = unorm(val)
        elif 'histogram' in key:
            if isinstance(val, torch.Tensor):
                val = val.cpu().numpy()
            outlier_bound = 10.
            while np.count_nonzero(val > outlier_bound) > val.size * 0.01:  # high number of outliers
                val = val[ val < outlier_bound ]
                outlier_bound -= 0.5
        mlog.update_meter(val, meters={key}, phase=phase)


def reset_log(mlog, flog, epoch, phase, use_thread=False):
    if use_thread:
        warnings.warn('use_threads set to True, but done synchronously still')
    if not mlog:
        return
    results = mlog.peek_meter(phase=phase)  # need to be run before reset
    mlog.reset_meter(epoch, mode=phase)
    # Log to file
    results_to_log = {}
    results['step_num'] = epoch
    for k in results.keys():
        if 'input/task' in k or 'output/task' in k:  # these are too big to log
            continue
        else:
            results_to_log[k] = results[k]
    if flog:
        flog.log('all_results', results_to_log)
    return []

def add_classification_specific_logging(cache, mlog, task=None, phase='train'):
    ''' Adds in top1 and top5
    '''
    prediction = cache['predictions']
    label = cache['labels']
    is_one_hot = len(label.shape) == 1
    if is_one_hot: # one-hot labels:
        top5_label = torch.stack([label] * 5, dim=-1)
    else:
        top5_label = torch.argsort(label, dim=1, descending=True)[:, :5]

    meter_suffix = f'/task_{task}' if task is not None else ''

    top1_label = top5_label[:,:1]
    mlog.update_meter(prediction, target=top1_label, meters={f'accuracy_top1{meter_suffix}'}, phase=phase)
    mlog.update_meter(prediction, target=top1_label, meters={f'accuracy_top5{meter_suffix}'}, phase=phase)
    top5_pred = torch.argsort(prediction, dim=1, descending=True)[:, :5]
    top1_pred = top5_pred[:,:1]

    entropy_pred = -1 * torch.sum(torch.softmax(prediction, dim=1) * F.log_softmax(prediction, dim=1)) / prediction.shape[0]
    perplexity_pred = torch.exp(entropy_pred).cpu()
    mlog.update_meter(perplexity_pred, meters={f'perplexity_pred{meter_suffix}'}, phase=phase)

    if is_one_hot:
        perplexity_label = 1
    else:
        if 0 <= torch.min(label) and torch.max(label) <= 1:  # already probably measure
            entropy_label = -1 * torch.sum(label * torch.log(label)) / label.shape[0]
        else:
            entropy_label = -1 * torch.sum(torch.softmax(label, dim=1) * F.log_softmax(label, dim=1)) / label.shape[0]
        perplexity_label = torch.exp(entropy_label).cpu()
    mlog.update_meter(perplexity_label, meters={f'perplexity_label{meter_suffix}'}, phase=phase)
    cache['top5_label'] = top5_label
    cache['top5_pred'] = top5_pred

def add_imitation_specific_logging(prediction, label, mlog, phase):
    perplexity = torch.mean(torch.exp(Categorical(logits=prediction).entropy()))
    mlog.update_meter(perplexity.cpu(), meters={'diagnostics/perplexity'}, phase=phase)
    if len(label.shape) == 2:
        mlog.update_meter(prediction.cpu(), target=torch.argmax(label.cpu(), dim=1), meters={'diagnostics/accuracy'}, phase=phase)
    elif len(label.shape) == 1:
        mlog.update_meter(prediction.cpu(), target=label.cpu(), meters={'diagnostics/accuracy'}, phase=phase)

MAX_NUM_IMAGES = 64
def log_image(mlog, task, cfg, x, label, prediction, masks=None, cache={}):
    targets = cache['targets']
    phase = cache['phase']
    encoding_only = all(['encoding' in t for t in targets]) and not isinstance(cfg['training']['loss_fn'], list) and not 'perceptual' in cfg['training']['loss_fn']
    masks = masks.cpu() if masks is not None else None
    if len(label.shape) == 4 and not encoding_only:
        if not isinstance(x, torch.Tensor):
            x = x[0]

    if any(['encoding' in t for t in targets]):  # there should have been something to do this earlier, where'd it go?
        prediction = cache['inputs_decoded']
        if 'targets_decoded' in cache:
            label = cache['targets_decoded']

    if 'class_object' in targets[task]:  # handle classification tasks
        warnings.warn('index_to_image will crash the program on k')
        return
        if not isinstance(x, torch.Tensor):
            x = x[0]
        _, _, img_size, _ = x.shape
        label = index_to_image(cache['top5_label'].cpu(), synset_arr, img_size).cuda()
        prediction = index_to_image(cache['top5_pred'].cpu(), synset_arr, img_size).cuda()

    if prediction.shape[1] == 2:  # handle 2 channels
        zero_layer = torch.zeros_like(prediction)[:,:1,:,:]
        prediction = torch.cat((prediction, zero_layer), dim=1)
        label = torch.cat((label, zero_layer), dim=1)

    if len(label.shape) == 4 and not encoding_only:
        # Unnormalize
        x_out = to_human(x.cpu())[:MAX_NUM_IMAGES]
        prediction_out = to_human(prediction.cpu())[:MAX_NUM_IMAGES]
        label_out = to_human(label.cpu())[:MAX_NUM_IMAGES]
        if masks is not None:
            masks_out = to_human(masks.cpu())[:MAX_NUM_IMAGES]
        else:
            masks_out = None
        im_samples = pack_images(x_out, prediction_out, label_out, mask=masks_out)
        log(mlog, f'output/task_{task}', im_samples, phase=phase)

    if isinstance(x, list):  # for more than single inputs (rgb, curv) ... can I do this earlier? not sure...
        x = x[0]

    if len(x.shape) == 4:
        x_out = to_human(x.cpu())
        log(mlog, f'input/task_{task}', x_out[0], phase=phase)

def write_logs(mlog, flog, task, step, cfg, cache={}, to_print=True)->list:
    phase = cache['phase']
    logs = mlog.peek_meter(phase=phase)
    if to_print:
        loss_str = ''
        for loss in cfg['training']['loss_list']:
            loss_name = f'losses/{loss}' if task is None else f'losses/{loss}_{task}'
            loss_value = logs[loss_name] # warning: this will become nan if nothing has been logged
            loss_value = loss_value.item() if not isinstance(loss_value, float) else loss_value
            loss_str += ' | ' + loss + ' loss: {0:.6f} '.format(loss_value)
        print(f'Logging step {step} ({phase}) {loss_str}')
    context = reset_log(mlog, flog, step, phase)
    return context

def get_logger(cfg, uuid):
    if cfg['saving']['logging_type'] == 'visdom':
        mlog = tnt.logger.VisdomMeterLogger(
            title=uuid, env=uuid, server=cfg['saving']['visdom_server'],
            port=cfg['saving']['visdom_port'],
            log_to_filename=cfg['saving']['visdom_log_file']
        )
    elif cfg['saving']['logging_type'] == 'tensorboard':
        mlog = tnt.logger.TensorboardMeterLogger(
            env=uuid,
            log_dir=cfg['saving']['log_dir'],
            plotstylecombined=True
        )
    else:
        assert False, 'no proper logger!'
    return mlog


def multidim_apply(x, dims, fn):
    if len(dims) == 0:
        return x
    else:
        return multidim_apply(fn(x, dim=dims[0], keepdim=True)[0], dims[1:], fn)

def to_human(x):
    # normalizes batch of image to (0,1)
    assert len(x.shape) == 4, 'working with batched images only'
    max_dim = multidim_apply(x, dims=[0, 2, 3], fn=torch.max)
    min_dim = multidim_apply(x, dims=[0, 2, 3], fn=torch.min)
    x_out = (x - min_dim) / (max_dim - min_dim)
    return x_out

