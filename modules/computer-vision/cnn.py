"""Example module for the runner: a simple CNN trained on MNIST."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from modules.ml_modules.mnist_classifier import BaseMNISTClassifier

MODULE_DIR = Path(__file__).parent


class ConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.linear1 = nn.Linear(9216, 128)
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional layers extract underlying patterns (edges, shapes) from the raw image input
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # Max pooling reduces the spatial dimensions of the feature maps, retaining the most important information
        x = F.max_pool2d(x, 2)
        # Dropout to selectively deactivate neurons during training to avoid overfitting, lighter after conv layers
        x = self.dropout1(x)
        # Flatten the dimensions from the CNN into a 1D tensor that the linear layers can consume
        x = torch.flatten(x, 1)
        # Linear layer for processing CNN features
        x = F.relu(self.linear1(x))
        # Dropout to selectively deactivate neurons during training to avoid overfitting, heavier after fc layer
        x = self.dropout2(x)
        # Second linear layer for further processing
        x = self.linear2(x)
        # Log softmax to convert the output into log probabilities for each class (log useful during loss evaluation)
        return F.log_softmax(x, dim=1)


class MNISTClassifier(BaseMNISTClassifier):
    _model_class: type = ConvNet
    _model_path: Path = MODULE_DIR / "cnn_mnist.pt"
