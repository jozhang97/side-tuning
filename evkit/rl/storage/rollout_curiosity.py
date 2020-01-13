from collections import defaultdict
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch
import torch.nn as nn
import random
import time

from .segment_tree import SumSegmentTree, MinSegmentTree
from evkit.rl.storage.memory import ReplayMemory
from evkit.sensors import SensorDict


class RolloutSensorDictCuriosityReplayBuffer(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space, state_size, actor_critic, use_gae, gamma, tau, memory_size=10000):
#         assert num_processes == 1
        self.num_steps = num_steps
        self.num_processes = num_processes
        self.state_size = state_size
        self.memory_size = memory_size
        self.obs_shape = obs_shape
        self.sensor_names = set(obs_shape.keys())
        self.observations = SensorDict({
            k: torch.zeros(memory_size, num_processes, *ob_shape)
            for k, ob_shape in obs_shape.items()
        })
        self.states = torch.zeros(memory_size, num_processes, state_size)
        self.rewards = torch.zeros(memory_size, num_processes, 1)
        self.value_preds = torch.zeros(memory_size, num_processes, 1)
        self.returns = torch.zeros(memory_size, num_processes, 1)
        self.action_log_probs = torch.zeros(memory_size, num_processes, 1)
        self.actions = torch.zeros(memory_size, num_processes, 1)
        self.masks = torch.ones(memory_size, num_processes, 1)
        
        self.actor_critic = actor_critic
        self.use_gae = use_gae
        self.gamma = gamma
        self.tau = tau

        self.num_steps = num_steps
        self.step = 0
        self.memory_occupied = 0
    
    def cuda(self):
        self.observations = self.observations.apply(lambda k, v: v.cuda())
        self.states = self.states.cuda()
        self.rewards = self.rewards.cuda()
        self.value_preds = self.value_preds.cuda()
        self.returns = self.returns.cuda()
        self.action_log_probs = self.action_log_probs.cuda()
        self.actions = self.actions.cuda()
        self.masks = self.masks.cuda()
        self.actor_critic = self.actor_critic.cuda()

    def insert(self, current_obs, state, action, action_log_prob, value_pred, reward, mask):
        next_step = (self.step + 1) % self.memory_size
        modules = [self.observations[k][next_step].copy_ for k in self.observations]
        inputs = tuple([(current_obs[k].peek(),) for k in self.observations])
        nn.parallel.parallel_apply(modules, inputs)
        # for k in self.observations:
            # self.observations[k][self.step + 1].copy_(current_obs[k].peek())
        self.states[next_step].copy_(state)
        self.actions[self.step].copy_(action)
        self.action_log_probs[self.step].copy_(action_log_prob)
        self.value_preds[self.step].copy_(value_pred)
        self.rewards[self.step].copy_(reward)
        self.masks[next_step].copy_(mask)

        self.step = (self.step + 1) % self.memory_size
        if self.memory_occupied < self.memory_size:
            self.memory_occupied += 1

    def get_current_observation(self):
        return self.observations.at(self.step)

    def get_current_state(self):
        return self.states[self.step]

    def get_current_mask(self):
        return self.masks[self.step]

    def after_update(self):
        pass



    def feed_forward_generator_with_next_state(self, advantages, num_mini_batch, on_policy=True):
        # Randomly sample a trajectory if off policy
        if on_policy or self.memory_occupied < self.memory_size:
            stop_idx = self.step - 1
            start_idx = (self.step - self.num_steps - 1) % self.memory_size
        else:
        # Make sure the start index is at least n+1 steps behind the current step
            start_idx = (self.step - 1 - np.random.randint(self.num_steps + 1, self.memory_size)) % self.memory_size
            stop_idx = (start_idx + self.num_steps - 1) % self.memory_size

        # Create buffers for the current sample
        # todo: clean this cuda mess up
        observations_sample = SensorDict({k: torch.zeros(self.num_steps + 1, self.num_processes, *ob_shape) for k, ob_shape in self.obs_shape.items()}).apply(lambda k, v: v.cuda())
        next_observations_sample = SensorDict({k: torch.zeros(self.num_steps + 1, self.num_processes, *ob_shape) for k, ob_shape in self.obs_shape.items()}).apply(lambda k, v: v.cuda())
        states_sample = torch.zeros(self.num_steps + 1, self.num_processes, self.state_size).cuda()
        rewards_sample = torch.zeros(self.num_steps, self.num_processes, 1).cuda()
        values_sample = torch.zeros(self.num_steps + 1, self.num_processes, 1).cuda()
        returns_sample = torch.zeros(self.num_steps + 1, self.num_processes, 1).cuda()
        action_log_probs_sample = torch.zeros(self.num_steps, self.num_processes, 1).cuda()
        actions_sample = torch.zeros(self.num_steps, self.num_processes, 1).cuda()
        masks_sample = torch.ones(self.num_steps + 1, self.num_processes, 1).cuda()


        
        # Fill the buffers and get values
        idx = start_idx
        sample_idx = 0
        while idx != (stop_idx % self.memory_size):
            next_idx = (idx + 1) % self.memory_size
            for k in self.observations:
                observations_sample[k][sample_idx] = self.observations[k][idx]
                for j, not_done in enumerate(self.masks[next_idx]):
                    if not_done > 0.5:  # if this timestep is not the end
                        next_observations_sample[k][sample_idx][j] = self.observations[k][next_idx][j]
                    else:
                        next_observations_sample[k][sample_idx][j] = self.observations[k][idx][j]
            states_sample[sample_idx] = self.states[idx]
            try:
                rewards_sample[sample_idx] = self.rewards[idx]
            except:
                print(rewards_sample, self.rewards)
                print(sample_idx, idx, next_idx, start_idx, stop_idx)
                raise
            action_log_probs_sample[sample_idx] = self.action_log_probs[idx]
            actions_sample[sample_idx] = self.actions[idx]
            masks_sample[sample_idx] = self.masks[idx]
            with torch.no_grad():
                next_value = self.actor_critic.get_value(self.observations.at(idx), self.states[idx], self.masks[idx])
                values_sample[sample_idx] = self.actor_critic.get_value(self.observations.at(idx), self.states[idx], self.masks[idx])
            idx = next_idx
            sample_idx += 1

        # we need to compute returns and advantages on the fly, since we now have updated value function predictions
        with torch.no_grad():
            next_value = self.actor_critic.get_value(self.observations.at(stop_idx), self.states[stop_idx], self.masks[stop_idx])
        if self.use_gae:
            values_sample[-1] = next_value
            gae = 0
            for step in reversed(range(rewards_sample.size(0))):
                delta = rewards_sample[step] + self.gamma * values_sample[step + 1] * masks_sample[step + 1] - values_sample[step]
                gae = delta + self.gamma * self.tau * masks_sample[step + 1] * gae
                returns_sample[step] = gae + values_sample[step]
        else:
            returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                returns_sample[step] = returns_sample[step + 1] * self.gamma * masks_batch[step + 1] + rewards_sample[step]

        mini_batch_size = self.num_steps // num_mini_batch
        observations_batch = {}
        next_observations_batch = {}
        sampler = BatchSampler(SubsetRandomSampler(range(self.num_steps)), mini_batch_size, drop_last=False)
        advantages = returns_sample[:-1] - values_sample[:-1]
        for indices in sampler:
            subsequent_indices = [i + 1 for i in indices]
            for k, sensor_ob in observations_sample.items():
                observations_batch[k] = sensor_ob[:-1].view(-1, *sensor_ob.size()[2:])[indices]
                next_observations_batch[k] = next_observations_sample[k][:-1].view(-1, *sensor_ob.size()[2:])[indices]
            states_batch = states_sample[:-1].view(-1, states_sample.size(-1))[indices]
            actions_batch = actions_sample.view(-1, actions_sample.size(-1))[indices]
            return_batch = returns_sample[:-1].view(-1, 1)[indices]
            masks_batch = masks_sample[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = action_log_probs_sample.view(-1, 1)[indices]
            adv_targ = advantages.view(-1, 1)[indices]
            yield observations_batch, next_observations_batch, states_batch, actions_batch, \
                return_batch, masks_batch, old_action_log_probs_batch, adv_targ


            
    def feed_forward_generator(self, advantages, num_mini_batch, on_policy=True):
        # Randomly sample a trajectory if off policy
        if on_policy or self.memory_occupied < self.memory_size:
            stop_idx = self.step
            start_idx = (self.step - self.num_steps) % self.memory_size
        else:
        # Make sure the start index is at least n+1 steps behind the current step
            start_idx = (self.step - np.random.randint(self.num_steps + 1, self.memory_size)) % self.memory_size
            stop_idx = (start_idx + self.num_steps) % self.memory_size

        # Create buffers for the current sample
        # todo: clean this cuda mess up
        observations_sample = SensorDict({k: torch.zeros(self.num_steps + 1, self.num_processes, *ob_shape) for k, ob_shape in self.obs_shape.items()}).apply(lambda k, v: v.cuda())
        states_sample = torch.zeros(self.num_steps + 1, self.num_processes, self.state_size).cuda()
        rewards_sample = torch.zeros(self.num_steps, self.num_processes, 1).cuda()
        values_sample = torch.zeros(self.num_steps + 1, self.num_processes, 1).cuda()
        returns_sample = torch.zeros(self.num_steps + 1, self.num_processes, 1).cuda()
        action_log_probs_sample = torch.zeros(self.num_steps, self.num_processes, 1).cuda()
        actions_sample = torch.zeros(self.num_steps, self.num_processes, 1).cuda()
        masks_sample = torch.ones(self.num_steps + 1, self.num_processes, 1).cuda()


        
        # Fill the buffers and get values
        idx = start_idx
        sample_idx = 0
        while idx != stop_idx:
            for k in self.observations:
                observations_sample[k][sample_idx] = self.observations[k][idx]
            states_sample[sample_idx] = self.states[idx]
            rewards_sample[sample_idx] = self.rewards[idx]
            action_log_probs_sample[sample_idx] = self.action_log_probs[idx]
            actions_sample[sample_idx] = self.actions[idx]
            masks_sample[sample_idx] = self.masks[idx]
            with torch.no_grad():
                next_value = self.actor_critic.get_value(self.observations.at(idx), self.states[idx], self.masks[idx])
                values_sample[sample_idx] = self.actor_critic.get_value(self.observations.at(idx), self.states[idx], self.masks[idx])
            idx = (idx + 1) % self.memory_size
            sample_idx += 1

        # we need to compute returns and advantages on the fly, since we now have updated value function predictions
        with torch.no_grad():
            next_value = self.actor_critic.get_value(self.observations.at(stop_idx), self.states[stop_idx], self.masks[stop_idx])
        if self.use_gae:
            values_sample[-1] = next_value
            gae = 0
            for step in reversed(range(rewards_sample.size(0))):
                delta = rewards_sample[step] + self.gamma * values_sample[step + 1] * masks_sample[step + 1] - values_sample[step]
                gae = delta + self.gamma * self.tau * masks_sample[step + 1] * gae
                returns_sample[step] = gae + values_sample[step]
        else:
            returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                returns_sample[step] = returns_sample[step + 1] * self.gamma * masks_batch[step + 1] + rewards_sample[step]

        mini_batch_size = self.num_steps // num_mini_batch
        observations_batch = {}
        sampler = BatchSampler(SubsetRandomSampler(range(self.num_steps)), mini_batch_size, drop_last=False)
        advantages = returns_sample[:-1] - values_sample[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        for indices in sampler:
            for k, sensor_ob in observations_sample.items():
                observations_batch[k] = sensor_ob[:-1].view(-1, *sensor_ob.size()[2:])[indices]
            states_batch = states_sample[:-1].view(-1, states_sample.size(-1))[indices]
            actions_batch = actions_sample.view(-1, actions_sample.size(-1))[indices]
            return_batch = returns_sample[:-1].view(-1, 1)[indices]
            masks_batch = masks_sample[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = action_log_probs_sample.view(-1, 1)[indices]
            adv_targ = advantages.view(-1, 1)[indices]
            yield observations_batch, states_batch, actions_batch, \
                return_batch, masks_batch, old_action_log_probs_batch, adv_targ


