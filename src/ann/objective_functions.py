import numpy as np
from ann.activations import Softmax


class CrossEntropyLoss:
    def __init__(self):
        self.softmax = Softmax()
        self.probs = None
        self.y_true = None

    def forward(self, logits, y_true):
        self.y_true = y_true
        self.probs = self.softmax.forward(logits)

        batch_size = logits.shape[0]

        log_likelihood = -np.log(
            self.probs[np.arange(batch_size), y_true] + 1e-9
        )

        loss = np.mean(log_likelihood)
        return loss

    def backward(self):
        batch_size = self.probs.shape[0]

        grad = self.probs.copy()
        grad[np.arange(batch_size), self.y_true] -= 1
        grad = grad / batch_size

        return grad


class MSELoss:
    def __init__(self):
        self.y_true = None
        self.y_pred = None

    def forward(self, y_pred, y_true):
        self.y_true = y_true
        self.y_pred = y_pred
        return np.mean((y_pred - y_true) ** 2)

    def backward(self):
        batch_size = self.y_true.shape[0]
        return 2 * (self.y_pred - self.y_true) / batch_size