import numpy as np

class ReLU:
    def __init__(self):
        self.X = None

    def forward(self, X):
        self.X = X
        return np.maximum(0, X)

    def backward(self, d_out):
        dX = d_out.copy()
        dX[self.X <= 0] = 0
        return dX

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, X):
        self.out = 1 / (1 + np.exp(-X))
        return self.out

    def backward(self, d_out):
        return d_out * self.out * (1 - self.out)

class Tanh:
    def __init__(self):
        self.out = None

    def forward(self, X):
        self.out = np.tanh(X)
        return self.out

    def backward(self, d_out):
        return d_out * (1 - self.out ** 2)

class Softmax:
    def __init__(self):
        self.out = None

    def forward(self, X):
        X_shifted = X - np.max(X, axis=1, keepdims=True)
        exp_X = np.exp(X_shifted)
        self.out = exp_X / np.sum(exp_X, axis=1, keepdims=True)
        return self.out

    def backward(self, d_out):
        return d_out
