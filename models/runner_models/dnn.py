from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class NeuralNet(nn.Module):
    def __init__(self, layer_dims: tuple[int]) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(layer_dims[i], layer_dims[i + 1]) for i in range(len(layer_dims) - 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            # Linear layer with a ReLU activation between layers
            x = F.relu(layer(x))
        # Linear layer with a ReLU activation from the last hidden layer to output layer
        x = self.layers[-1](x)
        # Log softmax to convert the output into log probabilities for each class (log useful during loss evaluation)
        return F.log_softmax(x, dim=1)


class MNISTNeuralNet(NeuralNet):
    def __init__(self) -> None:
        super().__init__((28 * 28, 128, 64, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten the 2D image into a 1D tensor for the neural network to consume
        x = torch.flatten(x, 1)
        return super().forward(x)
