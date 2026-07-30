"""Example module for the runner: a simple DNN trained on MNIST."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.ml_modules.mnist_classifier import BaseMNISTClassifier

MODULE_DIR = Path(__file__).parent


class NeuralNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class MNISTClassifier(BaseMNISTClassifier):
    _model_class: type = NeuralNet
    _model_path: Path = MODULE_DIR / "dnn_mnist.pt"
