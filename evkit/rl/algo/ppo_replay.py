import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

from evkit.rl.policy import BasePolicy
from evkit.utils.radam import RAdam

class PPOReplay(object):
    def __init__(self,
                 actor_critic: BasePolicy,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 on_policy_epoch,
                 off_policy_epoch,
                 num_steps,
                 n_frames,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 amsgrad=True,
                 weight_decay=0.0,
                 gpu_devices=None,
                 loss_kwargs={},
                 cache_kwargs={},
                 optimizer_class='optim.Adam',
                 optimizer_kwargs={},
                 ):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.on_policy_epoch = on_policy_epoch
        self.off_policy_epoch = off_policy_epoch
        self.num_mini_batch = num_mini_batch
        self.num_steps = num_steps
        self.n_frames = n_frames

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.loss_kwargs = loss_kwargs
        self.loss_kwargs['intrinsic_loss_coefs'] = self.loss_kwargs['intrinsic_loss_coefs'] if 'intrinsic_loss_coefs' in loss_kwargs else []
        self.loss_kwargs['intrinsic_loss_types'] = self.loss_kwargs['intrinsic_loss_types'] if 'intrinsic_loss_types' in loss_kwargs else []
        assert len(loss_kwargs['intrinsic_loss_coefs']) == len(loss_kwargs['intrinsic_loss_types']), 'must have same number of losses as loss_coefs'

        self.max_grad_norm = max_grad_norm

        self.optimizer = eval(optimizer_class)(
            [
                {'params': [param for name, param in actor_critic.named_parameters() if 'alpha' in name], 'weight_decay': 0.0},
                {'params': [param for name, param in actor_critic.named_parameters() if 'alpha' not in name]},
            ],
            lr=lr,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            **optimizer_kwargs,
        )
        self.last_grad_norm = None
        self.gpu_devices = gpu_devices

        # self.cache = {}
        # self.cache_kwargs = cache_kwargs


    def update(self, rollouts):
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        max_importance_weight_epoch = 0
        on_policy = [0] * self.on_policy_epoch
        off_policy = [1] * self.off_policy_epoch
        epochs = on_policy + off_policy
        random.shuffle(epochs)
        info = {}
        yield_cuda = not (torch.cuda.device_count() > 1 and (self.gpu_devices is None or len(self.gpu_devices) > 1))
        # put samples on CPU as opposed to CUDA mem if DataParallel so that the first GPU is not memory bottlenecked

        for e in epochs:
            if e == 0:
                data_generator = rollouts.feed_forward_generator(
                        None, self.num_mini_batch, on_policy=True, device=self.gpu_devices[0], yield_cuda=yield_cuda)
            else:
                data_generator = rollouts.feed_forward_generator(
                        None, self.num_mini_batch, on_policy=False, device=self.gpu_devices[0], yield_cuda=yield_cuda)

            for sample in data_generator:
                observations_batch, states_batch, actions_batch, \
                   return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample


                cache = {}

                # Reshape to do in a single forward pass for all steps
                # TODO we call evaluate_action and get_value in this method, which is inefficient
                values, action_log_probs, dist_entropy, states = self.actor_critic.evaluate_actions(
                    observations_batch, states_batch,
                    masks_batch, actions_batch, cache
                )

                intrinsic_loss_dict = self.actor_critic.compute_intrinsic_losses(
                    self.loss_kwargs,
                    observations_batch, states_batch,
                    masks_batch, actions_batch, cache
                )

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch.to(self.gpu_devices[0]))
                surr1 = ratio * adv_targ.to(self.gpu_devices[0])
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                           1.0 + self.clip_param) * adv_targ.to(self.gpu_devices[0])
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, return_batch.to(self.gpu_devices[0]))
                self.optimizer.zero_grad()

                total_loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef
                for iloss, iloss_coef in zip(self.loss_kwargs['intrinsic_loss_types'], self.loss_kwargs['intrinsic_loss_coefs']):
                    total_loss += intrinsic_loss_dict[iloss] * iloss_coef
                total_loss.backward()
                self.last_grad_norm = nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

                for iloss in self.loss_kwargs['intrinsic_loss_types']:
                    if iloss in info:
                        info[iloss] += intrinsic_loss_dict[iloss].item()
                    else:
                        info[iloss] = intrinsic_loss_dict[iloss].item()

                for key in cache:
                    key_flat = torch.cat(cache[key]).view(-1).detach()
                    if key in info:
                        info[key] = torch.cat((info[key], key_flat))
                    else:
                        info[key] = key_flat
                max_importance_weight_epoch = max(torch.max(ratio).item(), max_importance_weight_epoch)

        num_updates = (self.on_policy_epoch + self.off_policy_epoch) * self.num_mini_batch  # twice since on_policy and off_policy
        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        for iloss in self.loss_kwargs['intrinsic_loss_types']:
            info[iloss] /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, max_importance_weight_epoch, info
