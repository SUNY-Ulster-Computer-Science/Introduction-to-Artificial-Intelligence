"""Example module for the runner: a simple DNN trained on MNIST."""

from pathlib import Path

from modules.runner_models.dnn import NeuralNet
from modules.runner_modules.mnist_runner import BaseMNISTRunner

MODULE_DIR = Path(__file__).parent


class MNISTRunner(BaseMNISTRunner):
    _model_class: type = NeuralNet
    _model_path: Path = MODULE_DIR / "dnn_mnist.pt"
