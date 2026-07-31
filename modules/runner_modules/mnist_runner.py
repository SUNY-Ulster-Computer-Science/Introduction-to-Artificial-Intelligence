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
from torchvision import transforms

from modules.runner.base import MLModule

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_convert_and_normalize = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # Mean and stddev of MNIST training dataset
    ]
)


def _transform_batch(batch: dict) -> dict:
    """Applied lazily per-batch by `with_transform`, converts the HF dataset's PIL images into normalized tensors."""

    batch["pixel_values"] = [_convert_and_normalize(img.convert("L")) for img in batch["image"]]
    return batch


def _collate(examples: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate the input examples into pixel values and labels."""

    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    labels = torch.tensor([e["label"] for e in examples])
    return pixel_values, labels


class BaseMNISTRunner(MLModule):
    """Digit runner for MNIST. Implements all four runner commands."""

    _model_class: type = None
    _model_path: Path = None
    _dataset_id: str = "ylecun/mnist"

    def __init__(self) -> None:
        self._model: nn.Module | None = None

    def _get_dataloader(self, split: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        """Load an MNIST split from the Hugging Face Hub and wrap it in a DataLoader.

        Args:
            split: The dataset split to load.
            batch_size: The batch size for the DataLoader.
            shuffle: Whether to shuffle the data.

        Returns:
            DataLoader: The wrapped DataLoader.
        """

        dataset = load_dataset(self._dataset_id, split=split)
        dataset = dataset.with_transform(_transform_batch)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate)

    def _load_model(self, load_weights: bool = True) -> nn.Module:
        if self._model is not None:
            return self._model

        model = self._model_class().to(DEVICE)
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
        parser = argparse.ArgumentParser()
        parser.add_argument("-e", "--epochs", type=int, help="The number of epochs to train for", default=3)
        parser.add_argument("-b", "--batch-size", type=int, help="The batch size for training", default=64)
        parser.add_argument("-l", "--lr", type=float, help="The learning rate for training", default=1.0)
        args = parser.parse_args(args)

        # Get the training dataset split
        train_loader = self._get_dataloader("train", batch_size=args.batch_size, shuffle=True)

        model = self._load_model(load_weights=False)
        # Object used to adjust the model's parameters through gradient decent
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
        # Decays the learning date over time to solidify early learning
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

        model.train()  # Set model to train mode
        epoch_bar = tqdm.tqdm(range(1, args.epochs + 1), desc="Epochs", unit="epoch", position=0)
        for epoch in epoch_bar:
            batch_bar = tqdm.tqdm(
                train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch", position=1, leave=False
            )
            for data, target in batch_bar:
                # Send data and target to device
                data, target = data.to(DEVICE), target.to(DEVICE)
                # Reset gradients to zero so the model only learns based off of the current sample
                optimizer.zero_grad()
                # Inference the model to get its prediction
                output = model(data)
                # Calculate the negative log likelihood loss of the model's prediction versus the target
                loss = F.nll_loss(output, target)
                # Propagate the loss error backward through the model's parameters to calculate pre-parameter gradients
                loss.backward()
                # Update the model's weights by one step of gradient decent based on propagated loss gradients
                optimizer.step()
                batch_bar.set_postfix(loss=f"{loss.item():.4f}")

            # Lower the learning rate based on the training schedule
            scheduler.step()
            epoch_bar.set_postfix(loss=f"{loss.item():.4f}")

        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self._model_path)  # save model weights to disk for loading later
        print(f"Model saved to {self._model_path}")

    def test(self, args: list[str]) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("-b", "--batch-size", type=int, help="The batch size for training", default=1000)
        args = parser.parse_args(args)

        # Get the testing dataset split
        test_loader = self._get_dataloader("test", batch_size=args.batch_size, shuffle=False)

        model = self._load_model()
        model.eval()  # Set model to evaluation mode

        test_loss = 0.0
        correct = 0
        with torch.inference_mode():  # Tell PyTorch that it does not need to calculate gradients
            for data, target in tqdm.tqdm(test_loader, desc="Evaluating", unit="batch"):
                # Send data and target to device
                data, target = data.to(DEVICE), target.to(DEVICE)
                # Inference the model to get its prediction
                output = model(data)
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
        parser = argparse.ArgumentParser()
        parser.add_argument("image_path", type=str, help="The path to the image to classify")
        args = parser.parse_args(args)

        image_path = Path(args.image_path)
        if not image_path.exists():
            print(f"Error: image not found: {image_path}", file=sys.stderr)
            sys.exit(1)

        # Resize and normalize input image into a format compatible with MNIST
        # Expects a white number on a black background
        image = Image.open(image_path).convert("L").resize((28, 28))
        tensor = _convert_and_normalize(image).unsqueeze(0).to(DEVICE)

        model = self._load_model()
        model.eval()  # Set model to evaluation mode
        with torch.inference_mode():
            # Inference the model to get its prediction
            output = model(tensor)
            # Get the most likely element within the prediction (the numerical result 0-9)
            pred = int(output.argmax(dim=1).item())
            # Get the model's confidence by transforming og probabilities back into probabilities
            confidence = float(output.exp().max().item())

        print(f"Predicted digit: {pred} (confidence: {confidence * 100:.2f}%)")

    def view(self, args: list[str]) -> None:
        import matplotlib.pyplot as plt
        from torchview import draw_graph

        parser = argparse.ArgumentParser()
        parser.add_argument("-d", "--dpi", type=int, help="The DPI for the output image", default=300)
        args = parser.parse_args(args)

        model = self._load_model()

        graph = draw_graph(model, input_size=(1, 1, 28, 28), device=DEVICE, expand_nested=True)
        graph.visual_graph.attr(dpi=str(args.dpi))
        png_bytes = graph.visual_graph.pipe(format="png")
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
