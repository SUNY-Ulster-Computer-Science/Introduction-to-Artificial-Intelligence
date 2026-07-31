"""Example module for the runner: a simple LSTM trained on IMDB movie review sentiment."""

from pathlib import Path

from modules.runner_models.lstm import LSTM
from modules.runner_modules.imdb_runner import BaseIMDBRunner

MODULE_DIR = Path(__file__).parent


class IMDBRunner(BaseIMDBRunner):
    _model_class: type = LSTM
    _model_path: Path = MODULE_DIR / "lstm_imdb.pt"
    _vocab_path: Path = MODULE_DIR / "imdb_vocab.json"
    _label_names: tuple[str] = ("negative", "positive")
