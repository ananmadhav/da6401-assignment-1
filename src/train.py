import argparse
import numpy as np
import wandb

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset, train_val_split


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a neural network")

    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist"])

    parser.add_argument("--epochs", type=int, default=5)

    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--learning_rate", type=float, default=0.05)

    parser.add_argument("--optimizer", type=str, default="sgd",
                        choices=["sgd", "momentum", "nag", "rmsprop"])

    parser.add_argument("--num_layers", type=int, default=2)

    parser.add_argument("--hidden_size", nargs="+", type=int, default=[128, 128, 64])

    parser.add_argument("--activation", type=str, default="relu",
                        choices=["relu", "sigmoid", "tanh"])

    parser.add_argument("--loss", type=str, default="cross_entropy",
                        choices=["cross_entropy", "mse"])

    parser.add_argument("--weight_init", type=str, default="xavier",
                        choices=["xavier", "random"])

    parser.add_argument("--wandb_project", type=str, default="da6401_mlp")

    parser.add_argument("--model_save_path", type=str,
                    default="src/best_model.npy")

    args = parser.parse_args()
    args.hidden_layers = args.hidden_size

    return args


def main():
    args = parse_arguments()

    wandb.init(
        project=args.wandb_project,
        config=vars(args),
        mode="disabled"
    )

    print("Loading dataset...")
    X_train, y_train, X_test, y_test = load_dataset(args.dataset)

    X_train, y_train, X_val, y_val = train_val_split(X_train, y_train)

    print("Creating model...")
    model = NeuralNetwork(args)

    print("Starting training...")
    model.train(
        X_train,
        y_train,
        X_val,
        y_val,
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