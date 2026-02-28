import numpy as np

class NeuralLayer:
    def __init__(self, input_dim, output_dim, weight_init="xavier"):
        self.input_dim = input_dim
        self.output_dim = output_dim

        if weight_init == "xavier":
            limit = np.sqrt(6 / (input_dim + output_dim))
            self.W = np.random.uniform(-limit, limit,
                                       (input_dim, output_dim))
        else:
            self.W = np.random.randn(input_dim, output_dim) * 0.01

        self.b = np.zeros((1, output_dim))

        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

        self.X = None

    def forward(self, X):
        self.X = X
        return X @ self.W + self.b

    def backward(self, d_out):
        batch_size = self.X.shape[0]
        self.grad_W = (self.X.T @ d_out) / batch_size
        self.grad_b = np.sum(d_out, axis=0, keepdims=True) / batch_size

        dX = d_out @ self.W.T
        return dX

