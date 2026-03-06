import numpy as np


class NeuralLayer:
    def __init__(self, input_dim, output_dim, activation=None, weight_init="xavier"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation

        if weight_init == "xavier":
            limit = np.sqrt(6 / (input_dim + output_dim))
            self.W = np.random.uniform(-limit, limit, (input_dim, output_dim))
        else:
            self.W = np.random.randn(input_dim, output_dim) * 0.01

        self.b = np.zeros((1, output_dim))

        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

        self.X = None
        self.Z = None



    def activate(self, Z):
        if self.activation == "relu":
            return np.maximum(0, Z)
        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-Z))
        elif self.activation == "tanh":
            return np.tanh(Z)
        
        return Z

    def activation_grad(self, Z):
        if self.activation == "relu":
            return (Z > 0).astype(float)
        elif self.activation == "sigmoid":
            s = 1 / (1 + np.exp(-Z))
            return s * (1 - s)
        elif self.activation == "tanh":
            return 1 - np.tanh(Z) ** 2
        
        return np.ones_like(Z)

    def forward(self, X):
        self.X = X
        self.Z = X @ self.W + self.b
        return self.activate(self.Z)


    def backward(self, d_out):
        dZ = d_out * self.activation_grad(self.Z)
        batch_size = self.X.shape[0]

        self.grad_W = (self.X.T @ dZ) / batch_size
        self.grad_b = np.sum(dZ, axis=0, keepdims=True) / batch_size

        dX = dZ @ self.W.T
        return dX