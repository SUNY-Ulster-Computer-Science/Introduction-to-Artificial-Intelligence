from pathlib import Path

from modules.runner_models.transformer import DecoderTransformer
from modules.runner_modules.wikitext_runner import BaseWikitextRunner

MODULE_DIR = Path(__file__).parent


class WikiTextLanguageModel(BaseWikitextRunner):
    _model_class: type = DecoderTransformer
    _model_path: Path = MODULE_DIR / "transformer_wikitext.pt"
    _vocab_path: Path = MODULE_DIR / "wikitext_vocab.json"
