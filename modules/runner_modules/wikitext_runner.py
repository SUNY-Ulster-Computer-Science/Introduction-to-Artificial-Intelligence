import argparse
import io
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
import tqdm
from datasets import load_dataset
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from modules.runner.base import MLModule

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UNK_TOKEN = "<unk>"

# Very simple word tokenizer: lowercase, keep runs of letters/apostrophes.
_TOKEN_RE = re.compile(r"[a-z']+")


def _tokenize(text: str) -> list[str]:
    """A simple tokenizer which splits by words and apostrophes."""

    return _TOKEN_RE.findall(text.lower())


class _ChunkedLMDataset(Dataset):
    """Wraps one long stream of token ids into fixed-length (input, target) chunks."""

    def __init__(self, tokens: list[int], seq_len: int) -> None:
        self.tokens = tokens
        self.seq_len = seq_len
        # Each chunk needs seq_len + 1 tokens (seq_len inputs, plus one extra token so the last input position has a
        # target to predict).
        self.length = max(0, (len(tokens) - 1) // seq_len)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        chunk = self.tokens[start : start + self.seq_len + 1]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        target_ids = torch.tensor(chunk[1:], dtype=torch.long)
        return input_ids, target_ids


class BaseWikitextRunner(MLModule):
    """Simple next-token wikitext language model runner. Implements all four runner commands."""

    _model_class: type = None
    _model_path: Path = None
    _vocab_path: Path = None
    _dataset_id: str = "Salesforce/wikitext"
    _dataset_config: str = "wikitext-2-raw-v1"
    _text_column: str = "text"
    _seq_len: int = 64
    _max_vocab_size: int = 20000

    def __init__(self) -> None:
        self._model: nn.Module | None = None
        self._vocab: dict[str, int] | None = None

    def _get_vocab(self) -> dict[str, int]:
        """Fetch the vocab JSON if it is already built."""

        if self._vocab is not None:
            return self._vocab
        if not self._vocab_path.exists():
            raise FileNotFoundError(f"No vocabulary found at {self._vocab_path}. Run 'train' first.")
        self._vocab = json.loads(self._vocab_path.read_text())
        return self._vocab

    def _build_vocab(self, lines: list[str]) -> dict[str, int]:
        """Builds a vocabulary based on the contents of the dataset."""

        counts = Counter()
        for line in tqdm.tqdm(lines, desc="Building vocabulary", unit="line"):
            counts.update(_tokenize(line))

        vocab = {UNK_TOKEN: 0}
        for token, _ in counts.most_common(self._max_vocab_size - len(vocab)):
            vocab[token] = len(vocab)
        return vocab

    def _tokenize_corpus(self, split: str) -> list[int]:
        """Tokenize an entire dataset split into one flat stream of token ids."""

        vocab = self._get_vocab()
        raw = load_dataset(self._dataset_id, self._dataset_config, split=split)

        tokens: list[int] = []
        for row in tqdm.tqdm(raw, desc=f"Tokenizing {split}", unit="line"):
            text = row[self._text_column]
            if not text.strip():
                continue  # WikiText includes blank lines and section headers
            tokens.extend(vocab.get(tok, vocab[UNK_TOKEN]) for tok in _tokenize(text))
        return tokens

    def _get_dataloader(self, split: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        """Load a Wikitext dataset split from the Hugging Face Hub and wrap it in a DataLoader.

        Samples are created by chunking the entire dataset by sequence length.

        Args:
            split: The dataset split to load.
            batch_size: The batch size for the DataLoader.
            shuffle: Whether to shuffle the data.

        Returns:
            DataLoader: The wrapped DataLoader.
        """

        tokens = self._tokenize_corpus(split)
        dataset = _ChunkedLMDataset(tokens, self._seq_len)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _load_model(self, load_weights: bool = True) -> nn.Module:
        if self._model is not None:
            return self._model

        vocab = self._get_vocab()
        model = self._model_class(vocab_size=len(vocab)).to(DEVICE)
        if load_weights:
            if self._model_path.exists():
                model.load_state_dict(torch.load(self._model_path, map_location=DEVICE))
            else:
                print(
                    f"Warning: no trained weights found at {self._model_path}. "
                    "Using randomly initialized weights. Run 'train' first.",
                    file=sys.stderr,
                )
        self._model = model
        return model

    def train(self, args: list[str]) -> None:
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("-e", "--epochs", type=int, help="The number of epochs to train for", default=3)
        parser.add_argument("-b", "--batch-size", type=int, help="The batch size for training", default=64)
        parser.add_argument("-l", "--lr", type=float, help="The learning rate for training", default=1e-3)
        args = parser.parse_args(args)

        # Build (or reuse) the vocabulary from the training split only.
        if self._vocab_path.exists():
            self._vocab = json.loads(self._vocab_path.read_text())
        else:
            raw_train = load_dataset(self._dataset_id, self._dataset_config, split="train")
            self._vocab = self._build_vocab(raw_train[self._text_column])
            self._vocab_path.parent.mkdir(parents=True, exist_ok=True)
            self._vocab_path.write_text(json.dumps(self._vocab))
            print(f"Vocabulary of {len(self._vocab)} tokens saved to {self._vocab_path}")

        train_loader = self._get_dataloader("train", batch_size=args.batch_size, shuffle=True)

        model = self._load_model(load_weights=False)

        # Use Adam optimizer to even out gradients through model passes, Adam also schedules the learning rate
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        model.train()  # Set model to train mode
        epoch_bar = tqdm.tqdm(range(1, args.epochs + 1), desc="Epochs", unit="epoch", position=0)
        for epoch in epoch_bar:
            batch_bar = tqdm.tqdm(
                train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch", position=1, leave=False
            )
            for input_ids, target_ids in batch_bar:
                # Send input_ids and target_ids to device
                input_ids, target_ids = input_ids.to(DEVICE), target_ids.to(DEVICE)
                # Reset gradients to zero so the model only learns based off of the current sample
                optimizer.zero_grad()
                # Inference the model to get its prediction
                output = model(input_ids)  # (batch, seq_len, vocab_size)
                # Flatten batch and sequence dims together so nll_loss compares every position's prediction against its
                # shifted-by-one target.
                loss = F.nll_loss(output.reshape(-1, output.size(-1)), target_ids.reshape(-1))
                # Propagate the loss error backward through the model's parameters to calculate pre-parameter gradients
                loss.backward()
                # Update the model's weights by one step of gradient decent based on propagated loss gradients
                optimizer.step()
                batch_bar.set_postfix(loss=f"{loss.item():.4f}")

            epoch_bar.set_postfix(loss=f"{loss.item():.4f}")

        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self._model_path)
        print(f"Model saved to {self._model_path}")

    def test(self, args: list[str]) -> None:
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("-b", "--batch-size", type=int, help="The batch size for testing", default=64)
        args = parser.parse_args(args)

        self._get_vocab()  # Fail fast with a clear error if train hasn't run yet
        test_loader = self._get_dataloader("test", batch_size=args.batch_size, shuffle=False)

        model = self._load_model()
        model.eval()  # Set model to evaluation mode

        total_loss = 0.0
        total_tokens = 0
        with torch.inference_mode():
            for input_ids, target_ids in tqdm.tqdm(test_loader, desc="Evaluating", unit="batch"):
                # Send input_ids and target_ids to device
                input_ids, target_ids = input_ids.to(DEVICE), target_ids.to(DEVICE)
                # Inference the model to get its prediction
                output = model(input_ids)
                # Flatten batch and sequence dims together so nll_loss compares every position's prediction against its
                # shifted-by-one target.
                loss = F.nll_loss(output.reshape(-1, output.size(-1)), target_ids.reshape(-1), reduction="sum")
                # Collect the total loss and total tokens
                total_loss += loss.item()
                total_tokens += target_ids.numel()

        avg_loss = total_loss / total_tokens
        # Perplexity (exp of average negative log-likelihood per token) is the standard language modeling metric.
        perplexity = math.exp(avg_loss)
        print(f"Test set: Average loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")

    def inference(self, args: list[str]) -> None:
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("prompt", type=str, help="The prompt to generate from")
        parser.add_argument("-n", "--num-tokens", type=int, help="Number of tokens to generate", default=50)
        args = parser.parse_args(args)

        vocab = self._get_vocab()
        inv_vocab = {idx: tok for tok, idx in vocab.items()}

        model = self._load_model()
        model.eval()  # Set model to evaluation mode

        generated = [vocab.get(tok, vocab[UNK_TOKEN]) for tok in _tokenize(args.prompt)]
        if not generated:
            print("Error: prompt contained no recognizable tokens.", file=sys.stderr)
            sys.exit(1)

        with torch.inference_mode():
            # Run an inference until the number of requested tokens has been reached
            for _ in range(args.num_tokens):
                # Naive approach: re-run the full forward pass over the whole sequence so far (last _seq_len tokens)
                # A stateful RNN/LSTM loop or a Transformer KV-cache would be more efficient, at a complexity cost.
                context = generated[-self._seq_len :]
                input_ids = torch.tensor([context], dtype=torch.long, device=DEVICE)
                # Inference the model to get its prediction
                output = model(input_ids)
                next_token_logits = output[0, -1]
                next_id = int(next_token_logits.argmax().item())
                generated.append(next_id)

        generated_text = " ".join(inv_vocab.get(idx, UNK_TOKEN) for idx in generated)
        print(generated_text)

    def view(self, args: list[str]) -> None:
        import matplotlib.pyplot as plt
        from torchview import draw_graph

        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("-d", "--dpi", type=int, help="The DPI for the output image", default=300)
        args = parser.parse_args(args)

        model = self._load_model()

        # A dummy batch of token ids (all padding) for torchview.
        dummy_input = torch.zeros((1, self._seq_len), dtype=torch.long, device=DEVICE)
        graph = draw_graph(model, input_data=dummy_input, device=DEVICE, expand_nested=True)
        graph.visual_graph.attr(dpi=str(args.dpi))
        png_bytes = graph.visual_graph.pipe(format="png")  # in-memory, no file written
        image = Image.open(io.BytesIO(png_bytes))

        plt.figure(
            num=f"{self._model_class.__name__} Architecture",
            figsize=(image.width / args.dpi, image.height / args.dpi),
            dpi=args.dpi,
        )
        plt.imshow(image)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
