"""Example module for the runner: a simple DNN trained on MNIST."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from modules.ml_modules.mnist_classifier import BaseMNISTClassifier

MODULE_DIR = Path(__file__).parent


class NeuralNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(28 * 28, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten the 2D image into a 1D tensor for the neural network to consume
        x = torch.flatten(x, 1)
        # Linear layer with a ReLU activation from input to hidden layer 1
        x = F.relu(self.linear1(x))
        # Linear layer with a ReLU activation from hidden layer 1 to hidden layer 2
        x = F.relu(self.linear2(x))
        # Linear layer with a ReLU activation from hidden layer 2 to output layer
        x = self.linear3(x)
        # Log softmax to convert the output into log probabilities for each class (log useful during loss evaluation)
        return F.log_softmax(x, dim=1)


class MNISTClassifier(BaseMNISTClassifier):
    _model_class: type = NeuralNet
    _model_path: Path = MODULE_DIR / "dnn_mnist.pt"
