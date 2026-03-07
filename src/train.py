import argparse
import numpy as np
import wandb

from tensorflow.keras.datasets import mnist, fashion_mnist
from ann.neural_network import NeuralNetwork


def parse_arguments():

    parser = argparse.ArgumentParser(description="Train a neural network")

    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist"])

    parser.add_argument("--epochs", type=int, default=20)

    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--learning_rate", type=float, default=0.01)

    parser.add_argument("--optimizer", type=str, default="sgd",
                        choices=["sgd", "momentum", "nag", "rmsprop"])

    parser.add_argument("--hidden_layers", nargs="+", type=int,
                        default=[128, 64])

    parser.add_argument("--activation", type=str, default="relu",
                        choices=["relu", "sigmoid", "tanh"])

    parser.add_argument("--loss", type=str, default="cross_entropy",
                        choices=["cross_entropy", "mse"])

    parser.add_argument("--weight_init", type=str, default="xavier",
                        choices=["xavier", "random"])

    parser.add_argument("--wandb_project", type=str, default="da6401_mlp")

    parser.add_argument("--model_save_path", type=str,
                        default="src/best_model.npy")

    return parser.parse_args()


def load_dataset(dataset):

    if dataset == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    else:
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    X_train = X_train.reshape(-1, 784)
    X_test = X_test.reshape(-1, 784)

    return X_train, y_train, X_test, y_test


def main():
    args = parse_arguments()

    wandb.init(project=args.wandb_project, config=vars(args))
    print("Loading dataset...")

    X_train, y_train, X_test, y_test = load_dataset(args.dataset)
    print("Creating model...")

    model = NeuralNetwork(args)
    print("Starting training...")

    model.train(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    print("Evaluating model...")
    accuracy = model.evaluate(X_test, y_test)

    print("Test Accuracy:", accuracy)
    wandb.log({"test_accuracy": accuracy})

    print("Saving model...")

    best_weights = model.get_weights()
    np.save(args.model_save_path, best_weights)
    print("Training complete.")


if __name__ == "__main__":
    main()