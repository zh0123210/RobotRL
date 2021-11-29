import torch
import torch.nn as nn
import numpy as np


def layer_norm(layer, std=1.0, bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)


def lossvalue_norm(values_next, return_):
    return_6std = 6 * return_.std()
    loss_vf = torch.mean((values_next - return_).pow(2)) / return_6std
    return loss_vf


def advantage_norm(advantages):
    normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
    return normalized_advantages

class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """
    def __init__(self, shape, demean=True, discount=1, clip=5.0, update=True):
        self.shape = shape
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)
        self.demean = demean
        self.discount = discount
        self.clip = clip
        self.update = update

    def estimate(self, x):
        if self.update:
            self.push(x)
        mean = self._M
        var = self._S / (self._n - 1) if self._n > 1 else np.square(self._M)
        std = np.sqrt(var)
        if self.demean:
            x = x - mean
        x = x / (std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def push(self, x):
        self._M *= self.discount
        self._S *= self.discount ** 2
        x = np.asarray(x)
        if x.shape == ():
            x = np.asarray([x])
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    def stop_update(self):
        self.update = False

    def output(self):
        return self.shape, self._n, self._M, self._S, self.demean, self.discount, self.clip

    def load(self, shape, _n, _M, _S, demean, discount, clip):
        self.shape, self._n, self._M, self._S, self.demean, self.discount, self.clip = shape, _n, _M, _S, demean, discount, clip