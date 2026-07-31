"""Example module for the runner: a simple LSTM trained on IMDB movie review sentiment."""

from pathlib import Path

from modules.runner_models.lstm import ClassifierLSTM
from modules.runner_modules.imdb_runner import BaseIMDBRunner

MODULE_DIR = Path(__file__).parent


class IMDBRunner(BaseIMDBRunner):
    _model_class: type = ClassifierLSTM
    _model_path: Path = MODULE_DIR / "lstm_imdb.pt"
    _tokenizer_path: Path = MODULE_DIR / "imdb_vocab.json"
    _label_names: tuple[str] = ("negative", "positive")
