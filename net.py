import torch
import torch.nn as nn
import RobotRL.tricks as tricks

class StochasticVNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.log_std = nn.Parameter(torch.zeros(1, args.action_shape))
        self.fca1 = nn.Linear(args.state_shape, 64)
        self.fca2 = nn.Linear(64, 64)
        self.fca3 = nn.Linear(64, args.action_shape)
        self.fcc1 = nn.Linear(args.state_shape, 64)
        self.fcc2 = nn.Linear(64, 64)
        self.fcc3 = nn.Linear(64, 1)
        self.tanh = nn.Tanh()

        if args.layer_norm:
            tricks.layer_norm(self.fca1, std=1.0)
            tricks.layer_norm(self.fca2, std=1.0)
            tricks.layer_norm(self.fca3, std=0.01)

            tricks.layer_norm(self.fcc1, std=1.0)
            tricks.layer_norm(self.fcc2, std=1.0)
            tricks.layer_norm(self.fcc3, std=1.0)

    def forward_actor(self, state):
        x = self.fca1(state)
        x = self.tanh(x)
        x = self.fca2(x)
        x = self.tanh(x)
        mean = self.fca3(x)
        log_std = self.log_std
        return mean, log_std

    def forward_critic(self, state):
        x = self.fcc1(state)
        x = self.tanh(x)
        x = self.fcc2(x)
        x = self.tanh(x)
        value = self.fcc3(x)
        return value


class DeterministicQNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fca1 = nn.Linear(args.state_shape, 64)
        self.fca2 = nn.Linear(64, 64)
        self.fca3 = nn.Linear(64, args.action_shape)
        self.fcc1 = nn.Linear(args.state_shape + args.action_shape, 64)
        self.fcc2 = nn.Linear(64, 64)
        self.fcc3 = nn.Linear(64, 1)
        self.tanh = nn.Tanh()

        if args.layer_norm:
            tricks.layer_norm(self.fca1, std=1.0)
            tricks.layer_norm(self.fca2, std=1.0)
            tricks.layer_norm(self.fca3, std=0.01)

            tricks.layer_norm(self.fcc1, std=1.0)
            tricks.layer_norm(self.fcc2, std=1.0)
            tricks.layer_norm(self.fcc3, std=1.0)

    def forward_actor(self, state):
        x = self.fca1(state)
        x = self.tanh(x)
        x = self.fca2(x)
        x = self.tanh(x)
        action = self.fca3(x)
        return action

    def forward_critic(self, state, action):
        x = self.fcc1(torch.cat([state, action]))
        x = self.tanh(x)
        x = self.fcc2(x)
        x = self.tanh(x)
        value = self.fcc3(x)
        return value