import numpy as np
from ann.neural_layer import NeuralLayer
from ann.objective_functions import CrossEntropyLoss, MSELoss


class NeuralNetwork:
    def __init__(self, cli_args):
        self.layers = []

        input_dim = 784
        hidden_sizes = cli_args.hidden_layers
        activation = cli_args.activation
        weight_init = getattr(cli_args, "weight_init", "xavier")

        prev_dim = input_dim

        for h in hidden_sizes:
            layer = NeuralLayer(prev_dim, h, activation=activation, weight_init=weight_init)
            self.layers.append(layer)
            prev_dim = h

        self.layers.append(NeuralLayer(prev_dim, 10, activation=None, weight_init=weight_init))
        self.learning_rate = cli_args.learning_rate

        if cli_args.loss == "cross_entropy":
            self.loss_fn = CrossEntropyLoss()
        else:
            self.loss_fn = MSELoss()


    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X


    def backward(self, y_true, logits):
        d_out = self.loss_fn.backward()
        grad_W_list = []
        grad_b_list = []

        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)
            grad_W_list.append(layer.grad_W)
            grad_b_list.append(layer.grad_b)

        self.grad_W = grad_W_list
        self.grad_b = grad_b_list
        return self.grad_W, self.grad_b

    def update_weights(self):
        for layer in self.layers:
            layer.W -= self.learning_rate * layer.grad_W
            layer.b -= self.learning_rate * layer.grad_b


    def train(self, X_train, y_train, X_val, y_val, epochs=1, batch_size=32):

        n_samples = X_train.shape[0]

        for epoch in range(epochs):

            indices = np.random.permutation(n_samples)
            X_train = X_train[indices]
            y_train = y_train[indices]

            epoch_loss = 0

            for start in range(0, n_samples, batch_size):

                end = start + batch_size

                X_batch = X_train[start:end]
                y_batch = y_train[start:end]

                logits = self.forward(X_batch)

                y_onehot = np.zeros((y_batch.size, logits.shape[1]))
                y_onehot[np.arange(y_batch.size), y_batch] = 1

                loss = -np.sum(y_onehot * np.log(logits + 1e-8)) / y_batch.size
                epoch_loss += loss

                grad_W, grad_b = self.backward(y_onehot, logits)

                self.update_weights()

            epoch_loss /= (n_samples // batch_size)

            val_acc = self.evaluate(X_val, y_val)

            print(
                f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Val Accuracy: {val_acc:.4f}"
            )

            import wandb
            wandb.log({
                "train_loss": epoch_loss,
                "val_accuracy": val_acc
            })

    def evaluate(self, X, y):
        logits = self.forward(X)
        preds = np.argmax(logits, axis=1)

        acc = np.mean(preds == y)
        return acc


    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d


    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"

            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()

            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()