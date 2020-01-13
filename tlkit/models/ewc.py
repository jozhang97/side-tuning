# Based on https://github.com/moskomule/ewc.pytorch/blob/master/utils.py
import copy
import torch
from tqdm import tqdm

from tlkit.utils import process_batch_tuple

class EWC:
    # wraps around a loss_fn and has the same behavior as it (but adds penalties)
    def __init__(self, loss_fn, model, coef=1e-3, avg_tasks=False, n_samples_fisher=1000, **kwargs):
        self.loss_fn = loss_fn
        self.model = model
        self.coef = coef
        self.avg_tasks = avg_tasks  # instead of storing N tasks, store the average of them

        self.weights_anchor_list = []  # copies of base parameters from previous points in time
        self.precision_matrices_list = []
        self.n_samples_fisher = n_samples_fisher
        self.n_tasks = 0

    def __call__(self, *args, **kwargs):
        orig_losses = self.loss_fn(*args, **kwargs)
        regularization_loss = self.compute_penalty(cur_model=self.model)
        orig_losses.update({
            'total': orig_losses['total'] + self.coef * regularization_loss,
            'weight_tying': regularization_loss,
        })
        return orig_losses

    def compute_penalty(self, cur_model):
        loss = torch.tensor(0.).to(next(cur_model.parameters()).device)
        hits = 0
        for weights_anchor, precision_matrices in zip(self.weights_anchor_list, self.precision_matrices_list):
            for name, param in cur_model.base.named_parameters():
                if name in precision_matrices:
                    hits += 1
                    _loss = precision_matrices[name] * (weights_anchor[name] - param) ** 2
                    loss += _loss.sum()
        assert hits != 0 or len(self.weights_anchor_list) == 0, 'No parameters for computing ewc penalty, are you sure the names in model and precision_matrix match?'
        return loss


    def post_training_epoch(self, model, dataloader, post_training_cache, **kwargs):
        # at the end of each task, compute fisher matrices from entire dataset, store weights
        if 'weights_anchor' in post_training_cache:
            weights_anchor = post_training_cache['weights_anchor']
            precision_matrices = post_training_cache['precision_matrices']
        else:
            weights_anchor = copy.deepcopy({n:p.detach() for n,p in model.base.named_parameters() if p.requires_grad})
            precision_matrices = self._diag_fisher(model, dataloader, copy.deepcopy(weights_anchor), kwargs['cfg'])
            post_training_cache['weights_anchor'] = weights_anchor
            post_training_cache['precision_matrices'] = precision_matrices

        if self.avg_tasks:
            self.n_tasks += 1
            if len(self.weights_anchor_list) == 0:
                self.weights_anchor_list.append(weights_anchor)
                self.precision_matrices_list.append(precision_matrices)
            else:
                self.weights_anchor_list[0] = self._compute_running_avg(self.weights_anchor_list[0], weights_anchor, self.n_tasks)
                self.precision_matrices_list[0] = self._compute_running_avg(self.precision_matrices_list[0], precision_matrices, self.n_tasks)
        else:
            self.weights_anchor_list.append(weights_anchor)
            self.precision_matrices_list.append(precision_matrices)

    def _compute_running_avg(self, running_avg, sample, n):
        # running_avg * (n-1)/n + sample * 1/n
        for name in running_avg.keys():
            running_avg[name] = running_avg[name] * (n - 1) / n + sample[name] / n
        return running_avg

    def _diag_fisher(self, model, dataloader, precision_matrices, cfg):
        # F = E [ (grad log p(z)) ^ T (grad log p(z))]
        model.eval()
        for name, param in precision_matrices.items():
            param.data.zero_()

        n_samples = 0
        task_idx, dataloader = dataloader.get_last_dl()
        for batch_tuple in tqdm(dataloader, f'Computing fisher matrix for task {task_idx}'):
            n_samples += len(batch_tuple)
            if n_samples > self.n_samples_fisher:
                break

            x, label, masks = process_batch_tuple(batch_tuple, task_idx, cfg)

            model.zero_grad()
            predictions = model(x, task_idx=task_idx)

            # loss = self(predictions, label, masks)
            # loss['total'].backward()

            # log_p = F.nll_loss(F.log_softmax(predictions, dim=1), label)
            # log_p.backward()  # grad log_p
            log_p = self.loss_fn(predictions, label, masks)
            log_p['total'].backward()  # grad log_p

            for name, param in model.base.named_parameters():
                if name in precision_matrices:
                    precision_matrices[name].data += param.grad.data.detach() ** 2 / len(dataloader)

        return precision_matrices
