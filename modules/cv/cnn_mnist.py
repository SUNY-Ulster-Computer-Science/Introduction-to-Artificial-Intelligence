"""Example module for the runner: a simple CNN trained on MNIST."""

from pathlib import Path

from modules.runner_models.cnn import ConvNet
from modules.runner_modules.mnist_runner import BaseMNISTRunner

MODULE_DIR = Path(__file__).parent


class MNISTRunner(BaseMNISTRunner):
    _model_class: type = ConvNet
    _model_path: Path = MODULE_DIR / "cnn_mnist.pt"
