import argparse
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.datasets import mnist, fashion_mnist

from ann.neural_network import NeuralNetwork


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference on test set")

    parser.add_argument("--model_path", default="src/best_model.npy")
    parser.add_argument("--dataset", default="mnist")

    parser.add_argument("--hidden_layers", nargs="+", type=int, default=[128, 64])
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--loss", default="cross_entropy")

    return parser.parse_args()


def load_dataset(dataset):
    if dataset == "mnist":
        (_, _), (X_test, y_test) = mnist.load_data()
    else:
        (_, _), (X_test, y_test) = fashion_mnist.load_data()

    X_test = X_test.reshape(-1, 784) / 255
    return X_test, y_test


def load_model(model_path):
    return np.load(model_path, allow_pickle=True).item()


def evaluate_model(model, X_test, y_test):
    logits = model.forward(X_test)
    preds = np.argmax(logits, axis=1)

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, average="macro")
    recall = recall_score(y_test, preds, average="macro")
    f1 = f1_score(y_test, preds, average="macro")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():
    args = parse_arguments()
    X_test, y_test = load_dataset(args.dataset)
    model = NeuralNetwork(args)

    weights = load_model(args.model_path)
    model.set_weights(weights)
    metrics = evaluate_model(model, X_test, y_test)
    print(metrics)


if __name__ == "__main__":
    main()