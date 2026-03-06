import numpy as np

class SGD:
    def __init__(self, lr=0.001):
        self.lr = lr

    def step(self, layer):
        layer.W -= self.lr * layer.grad_W
        layer.b -= self.lr * layer.grad_b


class Momentum:
    def __init__(self, lr=0.001, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v_W = {}
        self.v_b = {}

    def step(self, layer, idx):
        if idx not in self.v_W:
            self.v_W[idx] = np.zeros_like(layer.W)
            self.v_b[idx] = np.zeros_like(layer.b)

        self.v_W[idx] = self.beta * self.v_W[idx] + (1 - self.beta) * layer.grad_W
        self.v_b[idx] = self.beta * self.v_b[idx] + (1 - self.beta) * layer.grad_b

        layer.W -= self.lr * self.v_W[idx]
        layer.b -= self.lr * self.v_b[idx]


class NAG:
    def __init__(self, lr=0.001, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v_W = {}
        self.v_b = {}

    def step(self, layer, idx):
        if idx not in self.v_W:
            self.v_W[idx] = np.zeros_like(layer.W)
            self.v_b[idx] = np.zeros_like(layer.b)

        v_prev_W = self.v_W[idx]
        v_prev_b = self.v_b[idx]

        self.v_W[idx] = self.beta * self.v_W[idx] + self.lr * layer.grad_W
        self.v_b[idx] = self.beta * self.v_b[idx] + self.lr * layer.grad_b

        layer.W -= (self.beta * v_prev_W + (1 + self.beta) * self.v_W[idx])
        layer.b -= (self.beta * v_prev_b + (1 + self.beta) * self.v_b[idx])


class RMSProp:
    def __init__(self, lr=0.001, beta=0.9, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.s_W = {}
        self.s_b = {}

    def step(self, layer, idx):
        if idx not in self.s_W:
            self.s_W[idx] = np.zeros_like(layer.W)
            self.s_b[idx] = np.zeros_like(layer.b)

        self.s_W[idx] = self.beta * self.s_W[idx] + (1 - self.beta) * (layer.grad_W ** 2)
        self.s_b[idx] = self.beta * self.s_b[idx] + (1 - self.beta) * (layer.grad_b ** 2)

        layer.W -= self.lr * layer.grad_W / (np.sqrt(self.s_W[idx]) + self.eps)
        layer.b -= self.lr * layer.grad_b / (np.sqrt(self.s_b[idx]) + self.eps)