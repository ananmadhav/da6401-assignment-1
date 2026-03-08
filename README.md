# Assignment 1: Multi-Layer Perceptron for Image Classification

Name: Anan Madhav T V
Roll No: MM22B013

Github link: https://github.com/ananmadhav/da6401-assignment-1
WandB Report Link: https://wandb.ai/anan-madhav-iit-madras/da6401_mlp/reports/DA6401-Assignment-1-MLP-Experiments--VmlldzoxNjE0MjM0Mw

## Overview

This assignment implements a **Multi-Layer Perceptron (MLP)** from scratch using **NumPy only**, without using deep learning frameworks such as PyTorch or TensorFlow.

The network is trained to perform **multi-class image classification** on the **MNIST dataset**. All core neural network components such as forward propagation, backpropagation, activation functions, optimizers, and loss functions are implemented manually.

Experiments and visualizations are tracked using **Weights & Biases (W&B)**.


## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPO_LINK
cd da6401_assignment_1
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

## Training the Model

Run the training script using:

```bash
python src/train.py \
--dataset mnist \
--epochs 30 \
--batch_size 64 \
--loss cross_entropy \
--optimizer rmsprop \
--learning_rate 0.001 \
--num_layers 3 \
--hidden_size 128 128 128 \
--activation relu \
--weight_init xavier
```

Training metrics are logged using **Weights & Biases**.

---

## Running Inference

To evaluate the trained model:

```bash
python src/inference.py \
--dataset mnist \
--model_path src/best_model.npy
```

This will output:

- Accuracy  
- Precision  
- Recall  
- F1 Score  

---

## Best Model Configuration

The best performing configuration obtained from the hyperparameter sweep:

```
Optimizer: RMSProp
Activation: ReLU
Hidden Layers: 3
Hidden Size: 128
Learning Rate: 0.001
Batch Size: 64
Loss Function: Cross Entropy
Weight Initialization: Xavier
```

The trained model weights are stored in:

```
src/best_model.npy
```

---

## Weights & Biases Report

Full experiment report and visualizations:

The report includes:

- Dataset visualization
- Hyperparameter sweep analysis
- Optimizer comparison
- Vanishing gradient analysis
- Dead neuron investigation
- Loss function comparison
- Global performance analysis
- Error analysis with confusion matrix
- Weight initialization comparison
- Fashion-MNIST transfer analysis

---

## Results

The implemented MLP achieves strong classification performance on the MNIST dataset and demonstrates the effectiveness of properly tuned hyperparameters and optimization strategies.


