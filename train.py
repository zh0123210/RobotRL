import torch
import numpy as np
import RobotRL.tricks as tricks
import gym
import time


class Count:
    def __init__(self):
        self.done = 0

    def clean(self):
        self.done = 0

    def add(self):
        self.done += 1

    def output(self):
        return self.done


class Buffer:
    def __init__(self, device, buffer_size=1e6):
        self.device = device
        self.memory = torch.DoubleTensor([]).to(self.device)
        self.size = buffer_size

    def store(self, trans):
        self.memory = torch.cat([self.memory, trans], dim=0)
        if self.size and self.memory.size()[0] > self.size:
            self.memory = self.memory[-self.size:, :]

    def clean(self):
        self.memory = torch.DoubleTensor([]).to(self.device)

    def output(self):
        return self.memory

    def load(self, memory):
        self.memory = memory


def get_logproba(net, state, action):
    mean, log_std = net.forward_actor(state)
    logproba = normal_logproba(action, mean, log_std)
    return logproba

def normal_logproba(x, mean, log_std, std=None):
    if std is None:
        std = torch.exp(log_std)
    std_sq = std.pow(2)
    logproba = - 0.5 * np.log(2 * np.pi) - log_std - (x - mean).pow(2) / (2 * std_sq)
    return logproba.sum(1)


def get_batch(batch, args):
    dimension_sa = args.state_shape + args.action_shape
    batch_states = batch[:, 0:args.state_shape]
    batch_actions = batch[:, args.state_shape:dimension_sa]
    batch_rewards = batch[:, dimension_sa]
    batch_norm_rewards = batch[:, dimension_sa + 1]
    batch_masks = batch[:, dimension_sa + 2]
    batch_size = batch.size()[0]
    batch_steps = int((batch_size - torch.sum(batch_masks, dim=0) / args.gamma).item())
    return batch_size, batch_states, batch_actions, batch_rewards, batch_norm_rewards, batch_masks, batch_steps


def get_gaes(length, rewards, values, masks, args):
    values = values.squeeze(1)
    prev_values = torch.cat((values[1:], torch.DoubleTensor([0]).to(args.device)), dim=0)
    advantages = [0.0] * length
    returns = [0.0] * length
    prev_return = 0.0
    prev_advantage = 0.0
    deltas = rewards + prev_values * masks - values
    for t in reversed(range(len(advantages))):
        returns[t] = rewards[t] + prev_return * masks[t]
        advantages[t] = deltas[t] + args.gae_lambda * prev_advantage * masks[t]
        prev_return = returns[t]
        prev_advantage = advantages[t]
    returns = torch.tensor(returns)
    advantages = torch.tensor(advantages)
    if args.advantage_norm:
        advantages = tricks.advantage_norm(advantages)
    return returns.to(args.device), advantages.to(args.device)


def get_stochastic_action(args, state, networks, random=True):
    mean, log_std = networks.forward_actor(state)
    if random:
        std = torch.exp(log_std)
        action = torch.normal(mean, std)
    else:
        action = mean
    action = torch.squeeze(action)
    action = action.detach().numpy()
    if args.action_shape == 1:
        action = np.array([action])
    return action


def get_deterministic_action(args, state, networks):
    action = networks.forward_actor(state)
    action = torch.squeeze(action)
    action = action.detach().numpy()
    noice = np.random.normal(0, args.sigma_now)
    action += noice
    if args.action_shape == 1:
        action = np.array([action])
    return action


def test(args, running_state, net):
    if "RobotRL" in args.env_name:
        env = gym.make(args.env_name, render=True)
    else:
        env = gym.make(args.env_name)
    rewards = 0
    for test_episode in range(5):
        state = env.reset()
        if args.state_norm:
            state = running_state.estimate(state)
        for t in range(args.max_step_per_round):
            if "RobotRL" not in args.env_name:
                env.render()
            time.sleep(1 / 240)
            state_tensor = torch.DoubleTensor(state)
            if args.policy == 'stochastic':
                action = get_stochastic_action(args, state_tensor, net, random=False)
            elif args.policy == 'deterministic':
                action = get_deterministic_action(args, state_tensor, net)
            next_state, reward, done, _ = env.step(action)
            rewards += reward
            if args.state_norm:
                next_state = running_state.estimate(next_state)
            if done:
                break
            state = next_state
    return rewards / 5


def collect_batch(pipe, args, running_state, running_reward, buffer, count, worker):
    if "RobotRL" in args.env_name:
        env = gym.make(args.env_name, render=args.render)
    else:
        env = gym.make(args.env_name)
    if args.seed is not None:
        env.seed(args.seed + worker)
        torch.manual_seed(args.seed + worker)
    torch.set_default_tensor_type(torch.DoubleTensor)
    while True:
        if args.train:
            net = pipe.recv()
        T = 0
        while T < args.horizon:
            rewards = []
            norm_rewards = []
            states = []
            actions = []
            masks = []
            state = env.reset()
            if args.state_norm:
                state = running_state.estimate(state)
            for t in range(args.max_step_per_round):
                if "RobotRL" not in args.env_name and args.render:
                    env.render()
                state_tensor = torch.DoubleTensor(state)
                states.append(state)
                if args.policy == 'stochastic':
                    action = get_stochastic_action(args, state_tensor, net, random=True)
                elif args.policy == 'deterministic':
                    action = get_deterministic_action(args, state_tensor, net)
                actions.append(action)
                next_state, reward, done, _ = env.step(action)
                if args.state_norm:
                    next_state = running_state.estimate(next_state)
                rewards.append(reward)
                if args.reward_norm:
                    norm_rewards.append(running_reward.estimate(reward))
                else:
                    norm_rewards.append(np.array([reward]))
                mask = 0 if done else args.gamma
                masks.append(mask)
                T += 1
                if done:
                    break
                state = next_state
            states = torch.DoubleTensor(states)
            actions = torch.DoubleTensor(actions)
            rewards = torch.DoubleTensor(rewards).unsqueeze(1)
            norm_rewards = torch.DoubleTensor(norm_rewards)
            masks = torch.DoubleTensor(masks).unsqueeze(1)
            trans = torch.cat([states, actions, rewards, norm_rewards, masks], dim=1)
            buffer.store(trans.to(args.device))
        count.add()
