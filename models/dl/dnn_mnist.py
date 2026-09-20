"""Example module for the runner: a simple DNN trained on MNIST."""

from pathlib import Path

from models.runner_models.dnn import MNISTNeuralNet
from models.runner_modules.mnist_runner import BaseMNISTRunner

MODULE_DIR = Path(__file__).parent


class MNISTRunner(BaseMNISTRunner):
    _model_class: type = MNISTNeuralNet
    _model_path: Path = MODULE_DIR / "dnn_mnist.pt"
