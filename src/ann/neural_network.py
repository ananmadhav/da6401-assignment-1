import numpy as np
from .neural_layer import NeuralLayer
from .activations import ReLU, Sigmoid, Tanh
from .objective_functions import MeanSquaredError, CrossEntropy
from .optimizers import SGD  


class NeuralNetwork:

    def __init__(self, cli_args):
        self.layers = []

        input_size = 784   
        output_size = 10   

        hidden_sizes = cli_args.hidden_size
        activation = cli_args.activation
        weight_init = cli_args.weight_init

        if activation == "relu":
            activation_class = ReLU
        elif activation == "sigmoid":
            activation_class = Sigmoid
        elif activation == "tanh":
            activation_class = Tanh
        else:
            raise ValueError("Unsupported activation")

        prev_size = input_size

        for size in hidden_sizes:
            self.layers.append(
                NeuralLayer(prev_size, size, weight_init)
            )
            self.layers.append(
                activation_class()
            )
            prev_size = size

        self.layers.append(
            NeuralLayer(prev_size, output_size, weight_init)
        )

        if cli_args.loss == "mse":
            self.loss_fn = MeanSquaredError()
        elif cli_args.loss == "cross_entropy":
            self.loss_fn = CrossEntropy()
        else:
            raise ValueError("Unsupported loss")

        self.optimizer = SGD(cli_args.learning_rate)


    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, y_true, logits):
        loss = self.loss_fn.forward(logits, y_true)
        d_out = self.loss_fn.backward()

        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)

        return loss

    def update_weights(self):
        trainable_layers = [
            layer for layer in self.layers
            if isinstance(layer, NeuralLayer)
        ]
        self.optimizer.step(trainable_layers)


    def train(self, X_train, y_train, epochs, batch_size):
        n = X_train.shape[0]

        for epoch in range(epochs):
            permutation = np.random.permutation(n)
            X_train = X_train[permutation]
            y_train = y_train[permutation]

            epoch_loss = 0

            for i in range(0, n, batch_size):

                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                logits = self.forward(X_batch)
                loss = self.backward(y_batch, logits)

                self.update_weights()
                epoch_loss += loss * X_batch.shape[0]
            epoch_loss /= n

            print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

    def evaluate(self, X, y):
        logits = self.forward(X)

        if isinstance(self.loss_fn, CrossEntropy):
            logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits_shifted)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            predictions = np.argmax(probs, axis=1)
        else:
            predictions = np.argmax(logits, axis=1)

        true_labels = np.argmax(y, axis=1)

        accuracy = np.mean(predictions == true_labels)

        return accuracy
