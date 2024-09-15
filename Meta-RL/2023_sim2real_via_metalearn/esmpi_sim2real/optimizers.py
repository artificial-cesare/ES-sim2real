"""
Adapted from https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/optimizers.py
"""

import torch.nn as nn
import numpy as np


class Optimizer(object):
    def __init__(self, network):
        self.network = network
        self.dim = len(nn.utils.parameters_to_vector(self.network.parameters()).cpu().detach())
        self.t = 0

    def update(self, globalg):
        self.t += 1
        step = self._compute_step(globalg)
        theta = nn.utils.parameters_to_vector(self.network.parameters()).cpu().detach()
        ratio = np.linalg.norm(step) / np.linalg.norm(theta)
        nn.utils.vector_to_parameters(theta+step, self.network.parameters())
        return ratio

    def _compute_step(self, globalg):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, pi, lr, momentum=0.9):
        Optimizer.__init__(self, pi)
        self.v = np.zeros(self.dim, dtype=np.float32)
        self.lr, self.momentum = lr, momentum

    def _compute_step(self, globalg):
        self.v = self.momentum * self.v + (1. - self.momentum) * globalg
        step = -self.lr * self.v
        return step


class Adam(Optimizer):
    def __init__(self, pi, lr, beta1=0.9, beta2=0.999, epsilon=1e-08):
        Optimizer.__init__(self, pi)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)

    def _compute_step(self, globalg):
        a = self.lr * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step
