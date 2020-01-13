import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import random



class PPOCuriosity(object):
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 optimizer=None,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 amsgrad=True,
                 weight_decay=0.0):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        
        self.forward_loss_coef = 0.2
        self.inverse_loss_coef = 0.8
        self.curiosity_coef = 0.2
        self.original_task_reward_proportion = 1.0
        
        self.max_grad_norm = max_grad_norm
        
        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = optim.Adam(actor_critic.parameters(),
                                        lr=lr,
                                        eps=eps,
                                        weight_decay=weight_decay,
                                        amsgrad=amsgrad)
        self.last_grad_norm = None

    def update(self, rollouts):
        advantages = rollouts.returns * self.original_task_reward_proportion - rollouts.value_preds
#         advantages = (advantages - advantages.mean()) / (
#             advantages.std() + 1e-5)


        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        max_importance_weight_epoch = 0
        self.forward_loss_epoch = 0
        self.inverse_loss_epoch = 0

        for e in range(self.ppo_epoch):
            if hasattr(self.actor_critic.base, 'gru'):
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
                raise NotImplementedError("PPOCuriosity has not implemented for recurrent networks because masking is undefined")
            else:
#                 data_generator = rollouts.feed_forward_generator(
#                     advantages, self.num_mini_batch)
                data_generator = rollouts.feed_forward_generator_with_next_state(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                observations_batch, next_observations_batch, rnn_history_state, actions_batch, \
                   return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample
                
#                 observations_batch, rnn_history_state, actions_batch, \
#                    return_batch, masks_batch, old_action_log_probs_batch, \
#                         adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, next_rnn_history_state, state_features = self.actor_critic.evaluate_actions(
                    observations_batch, rnn_history_state,
                    masks_batch, actions_batch)
#                 import pdb
#                 pdb.set_trace()
                # masks_batch is from state_t but we ned for state_t_plus_1. Bad if recurrent!
                value, next_state_features, _ = self.actor_critic.base(
                    next_observations_batch, next_rnn_history_state, masks_batch) 
                                
                # Curiosity
                # Inverse Loss
                pred_action = self.actor_critic.base.inverse_model(state_features.detach(), next_state_features)
                self.inverse_loss = F.cross_entropy(pred_action, actions_batch.squeeze(1))

                # Forward Loss: Only works for categorical actions
                one_hot_actions = torch.zeros((actions_batch.shape[0], self.actor_critic.dist.num_outputs), device=actions_batch.device)
                one_hot_actions.scatter_(1, actions_batch, 1.0)
                pred_next_state = self.actor_critic.base.forward_model(state_features.detach(), one_hot_actions)
                self.forward_loss = F.mse_loss(pred_next_state, next_state_features.detach())

                # Exploration bonus
                curiosity_bonus = (1.0 - self.original_task_reward_proportion) * self.curiosity_coef * self.forward_loss
                return_batch += curiosity_bonus
                adv_targ += curiosity_bonus
                adv_targ = (adv_targ - adv_targ.mean()) / (adv_targ.std() + 1e-5)
                
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                clipped_ratio = torch.clamp(ratio,
                                            1.0 - self.clip_param,
                                            1.0 + self.clip_param)
                surr1 = ratio * adv_targ
                surr2 = clipped_ratio * adv_targ
                self.action_loss = -torch.min(surr1, surr2).mean()
                # value_loss = torch.mean(clipped_ratio * (values - return_batch) ** 2)
                self.value_loss = F.mse_loss(values, return_batch)
                self.dist_entropy = dist_entropy
                self.optimizer.zero_grad()
                self.get_loss().backward()
                
                nn.utils.clip_grad_norm_(self.forward_loss.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.inverse_loss.parameters(), self.max_grad_norm)
                self.last_grad_norm = nn.utils.clip_grad_norm_(
                                        self.actor_critic.parameters(),
                                        self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += self.value_loss.item()
                action_loss_epoch += self.action_loss.item()
                dist_entropy_epoch += self.dist_entropy.item()
                self.forward_loss_epoch += self.forward_loss.item()
                self.inverse_loss_epoch += self.inverse_loss.item()
                max_importance_weight_epoch = max(torch.max(ratio).item(), max_importance_weight_epoch)

        num_updates = self.ppo_epoch * self.num_mini_batch
        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        self.forward_loss_epoch /= num_updates
        self.inverse_loss_epoch /= num_updates
        self.last_update_max_importance_weight = max_importance_weight_epoch
        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, {}

    def get_loss(self):
        return self.value_loss * self.value_loss_coef \
               + self.action_loss \
               - self.dist_entropy * self.entropy_coef \
               + self.forward_loss * self.forward_loss_coef \
               + self.inverse_loss * self.inverse_loss_coef
    

    

class PPOReplayCuriosity(object):
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 on_policy_epoch,
                 off_policy_epoch,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 amsgrad=True,
                 weight_decay=0.0,
                 curiosity_reward_coef=0.1,
                 forward_loss_coef=0.2,
                 inverse_loss_coef=0.8):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.on_policy_epoch = on_policy_epoch
        self.off_policy_epoch = off_policy_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.forward_loss_coef = forward_loss_coef
        self.inverse_loss_coef = inverse_loss_coef

        self.curiosity_reward_coef = curiosity_reward_coef


        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(actor_critic.parameters(),
                                    lr=lr,
                                    eps=eps,
                                    weight_decay=weight_decay,
                                    amsgrad=amsgrad)
        self.last_grad_norm = None

    def update(self, rollouts):
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        self.forward_loss_epoch = 0
        self.inverse_loss_epoch = 0
        max_importance_weight_epoch = 0
        on_policy = [0] * self.on_policy_epoch
        off_policy = [1] * self.off_policy_epoch
        epochs = on_policy + off_policy
        random.shuffle(epochs)

        for e in epochs:
            if e == 0:
                data_generator = rollouts.feed_forward_generator_with_next_state(
                        None, self.num_mini_batch, on_policy=True)
            else:
                data_generator = rollouts.feed_forward_generator_with_next_state(
                        None, self.num_mini_batch, on_policy=False)

            for sample in data_generator:
                observations_batch, next_observations_batch, states_batch, actions_batch, \
                   return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                actions_batch_long = actions_batch.type(torch.cuda.LongTensor)
                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, next_states_batch = self.actor_critic.evaluate_actions(
                    observations_batch, states_batch, masks_batch, actions_batch)
                
                
                
                # Curiosity
                # Inverse Loss
                state_feats = self.actor_critic.base.perception_unit(observations_batch)
                next_state_feats = self.actor_critic.base.perception_unit(next_observations_batch)
                pred_action = self.actor_critic.base.inverse_model(
                                                    state_feats,
                                                    next_state_feats)
                self.inverse_loss = F.cross_entropy(pred_action,
                                                    actions_batch_long.squeeze(1))

                # Forward Loss: Only works for categorical actions
                one_hot_actions = torch.zeros((actions_batch.shape[0], self.actor_critic.dist.num_outputs),
                                              device=actions_batch.device)
                one_hot_actions.scatter_(1, actions_batch_long, 1.0)
                pred_next_state = self.actor_critic.base.forward_model(
                                              state_feats,
                                              one_hot_actions)
                self.forward_loss = F.mse_loss(pred_next_state, 
                                              next_state_feats)

                curiosity_bonus = self.curiosity_reward_coef * self.forward_loss
                adv_targ += curiosity_bonus.detach()
                adv_targ = (adv_targ - adv_targ.mean()) / (adv_targ.std() + 1e-5)

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                           1.0 + self.clip_param) * adv_targ
                self.action_loss = -torch.min(surr1, surr2).mean()
                self.value_loss = F.mse_loss(values, return_batch)
                self.dist_entropy = dist_entropy
                self.optimizer.zero_grad()
                self.get_loss().backward()
                
                nn.utils.clip_grad_norm_(self.actor_critic.base.forward_model.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.actor_critic.base.inverse_model.parameters(), self.max_grad_norm)
                self.last_grad_norm = nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += self.value_loss.item()
                action_loss_epoch += self.action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                self.forward_loss_epoch += self.forward_loss.item()
                self.inverse_loss_epoch += self.inverse_loss.item()
                max_importance_weight_epoch = max(torch.max(ratio).item(), max_importance_weight_epoch)
                self.last_update_max_importance_weight = max_importance_weight_epoch

        num_updates = self.ppo_epoch * self.num_mini_batch
        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, max_importance_weight_epoch, {}

    def get_loss(self):
        return self.value_loss * self.value_loss_coef \
               + self.action_loss \
               - self.dist_entropy * self.entropy_coef \
               + self.forward_loss * self.forward_loss_coef \
               + self.inverse_loss * self.inverse_loss_coef