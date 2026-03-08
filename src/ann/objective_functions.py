import numpy as np
from ann.activations import Softmax



class CrossEntropyLoss:
    def __init__(self):
        self.probs = None
        self.y_onehot = None

    def forward(self, logits, y_true):
        # Stable softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        batch_size = logits.shape[0]

        # Convert labels to one-hot once and store
        self.y_onehot = np.zeros_like(self.probs)
        self.y_onehot[np.arange(batch_size), y_true] = 1

        loss = -np.sum(self.y_onehot * np.log(self.probs + 1e-8)) / batch_size

        return loss

    def backward(self):
        batch_size = self.probs.shape[0]
        return (self.probs - self.y_onehot) / batch_size


class MSELoss:
    def __init__(self):
        self.y_true = None
        self.y_pred = None

    def forward(self, y_pred, y_true):

        self.y_pred = y_pred
        self.y_true = y_true

        if y_true.ndim == 1:
            one_hot = np.zeros((y_true.size, y_pred.shape[1]))
            one_hot[np.arange(y_true.size), y_true] = 1
            self.y_true = one_hot

        return np.mean((y_pred - self.y_true) ** 2)

    def backward(self):

        batch_size = self.y_true.shape[0]
        return 2 * (self.y_pred - self.y_true) / batch_size