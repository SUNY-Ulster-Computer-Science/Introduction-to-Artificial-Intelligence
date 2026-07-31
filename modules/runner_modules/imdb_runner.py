import argparse
import io
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import tqdm
from datasets import load_dataset
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from modules.runner.base import MLModule
from modules.runner_models.word_tokenizer import WordTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseIMDBRunner(MLModule):
    """Simple IMDB text runner. Implements all four runner commands."""

    _model_class: type = None
    _model_path: Path = None
    _tokenizer_path: Path = None
    _dataset_id: str = "stanfordnlp/imdb"
    _text_column: str = "text"
    _label_column: str = "label"
    _label_names: tuple[str] = None
    _max_seq_len: int = 200
    _max_vocab_size: int = 20000
    _padding_side: str = "left"

    def __init__(self) -> None:
        self._model: nn.Module | None = None
        self._tokenizer: WordTokenizer | None = None

    def _load_tokenizer(self) -> WordTokenizer:
        """Fetch the tokenizer from disk if it is already built."""

        if self._tokenizer is not None:
            return self._tokenizer
        if not self._tokenizer_path.exists():
            raise FileNotFoundError(f"No vocabulary found at {self._tokenizer_path}. Run 'train' first.")
        self._tokenizer = WordTokenizer.load(self._tokenizer_path)
        self._tokenizer.padding_side = self._padding_side
        return self._tokenizer

    def _train_tokenizer(self, lines: list[str]) -> WordTokenizer:
        """Builds a vocabulary based on the contents of the dataset."""

        tokenizer = WordTokenizer()
        tokenizer.train(lines, self._max_vocab_size)
        return tokenizer

    def _collate(self, examples: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        """Collate the input examples into token ids and labels."""

        id_list = []
        for e in examples:
            id_list.append(self._tokenizer.encode(e[self._text_column], self._max_seq_len, True))

        input_ids = torch.tensor(id_list, dtype=torch.long)
        labels = torch.tensor([e[self._label_column] for e in examples], dtype=torch.long)
        return input_ids, labels

    def _get_dataloader(self, split: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        """Load an IMDB dataset split from the Hugging Face Hub and wrap it in a DataLoader.

        Args:
            split: The dataset split to load.
            batch_size: The batch size for the DataLoader.
            shuffle: Whether to shuffle the data.

        Returns:
            DataLoader: The wrapped DataLoader.
        """

        dataset = load_dataset(self._dataset_id, split=split)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self._collate)

    def _load_model(self, load_weights: bool = True) -> nn.Module:
        if self._model is not None:
            return self._model

        tokenizer = self._load_tokenizer()
        model = self._model_class(vocab_size=tokenizer.vocab_size, num_classes=len(self._label_names)).to(DEVICE)
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

        # Build the tokenizer from the training split only.
        if not self._tokenizer_path.exists():
            raw_train = load_dataset(self._dataset_id, split="train")
            tokenizer = self._train_tokenizer(raw_train[self._text_column])
            self._tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
            tokenizer.save(self._tokenizer_path)
            print(f"Vocabulary of {tokenizer.vocab_size} tokens saved to {self._tokenizer_path}")

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
            for input_ids, target in batch_bar:
                # Send input_ids and target to device
                input_ids, target = input_ids.to(DEVICE), target.to(DEVICE)
                # Reset gradients to zero so the model only learns based off of the current sample
                optimizer.zero_grad()
                # Inference the model to get its prediction
                output = model(input_ids)
                # Calculate the negative log likelihood loss of the model's prediction versus the target
                loss = F.nll_loss(output, target)
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
        parser.add_argument("-b", "--batch-size", type=int, help="The batch size for testing", default=256)
        args = parser.parse_args(args)

        self._load_tokenizer()
        test_loader = self._get_dataloader("test", batch_size=args.batch_size, shuffle=False)

        model = self._load_model()
        model.eval()  # Set model to evaluation mode

        test_loss = 0.0
        correct = 0
        with torch.inference_mode():
            for input_ids, target in tqdm.tqdm(test_loader, desc="Evaluating", unit="batch"):
                # Send input_ids and target to device
                input_ids, target = input_ids.to(DEVICE), target.to(DEVICE)
                # Inference the model to get its prediction
                output = model(input_ids)
                # Add the negative log likelihood loss of the model's prediction versus the target to the sum
                test_loss += F.nll_loss(output, target, reduction="sum").item()
                # Get the most likely element within the prediction (the numerical result 0-9)
                pred = output.argmax(dim=1, keepdim=True)
                # Check if the model's prediction exactly matches the target label
                correct += pred.eq(target.view_as(pred)).sum().item()

        n = len(test_loader.dataset)
        test_loss /= n
        accuracy = 100.0 * correct / n
        print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{n} ({accuracy:.2f}%)")

    def inference(self, args: list[str]) -> None:
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("text", type=str, help="Text to classify")
        args = parser.parse_args(args)

        tokenizer = self._load_tokenizer()
        model = self._load_model()
        model.eval()  # Set model to evaluation mode

        input_ids = torch.tensor(tokenizer.encode(args.text), dtype=torch.long).to(DEVICE)
        with torch.inference_mode():
            # Inference the model to get its prediction
            output = model(input_ids)
            # Get the most likely element within the prediction
            pred = int(output.argmax(dim=1).item())
            # Get the model's confidence by transforming log probabilities back into probabilities
            confidence = float(output.exp().max().item())

        label = self._label_names[pred]
        print(f"Predicted label: {label} (confidence: {confidence * 100:.2f}%)")

    def view(self, args: list[str]) -> None:
        import matplotlib.pyplot as plt
        from torchview import draw_graph

        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("-d", "--dpi", type=int, help="The DPI for the output image", default=300)
        args = parser.parse_args(args)

        model = self._load_model()

        # A dummy batch of token ids (all padding) for torchview.
        dummy_input = torch.zeros((1, self._max_seq_len), dtype=torch.long, device=DEVICE)
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
