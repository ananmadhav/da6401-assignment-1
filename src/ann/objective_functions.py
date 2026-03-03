import numpy as np


class MeanSquaredError:
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true) ** 2)

    def backward(self):
        batch_size = self.y_true.shape[0]
        return (2 / batch_size) * (self.y_pred - self.y_true)


class CrossEntropy:
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, logits, y_true):

        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        self.y_pred = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        self.y_true = y_true

        eps = 1e-12  
        log_probs = np.log(self.y_pred + eps)
        loss = -np.sum(y_true * log_probs) / y_true.shape[0]

        return loss

    def backward(self):
        batch_size = self.y_true.shape[0]
        return (self.y_pred - self.y_true) / batch_size
