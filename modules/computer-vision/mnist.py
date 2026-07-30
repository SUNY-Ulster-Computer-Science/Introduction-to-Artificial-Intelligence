"""Example module for the runner: a simple CNN trained on MNIST."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from modules.runner.base import MLModule

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODULE_DIR = Path(__file__).parent
MODEL_PATH = MODULE_DIR / "mnist_model.pt"

HF_DATASET_ID = "ylecun/mnist"

TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # Mean and stddev of MNIST training dataset
    ]
)


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def _transform_batch(batch: dict) -> dict:
    """Applied lazily per-batch by `with_transform`, converts the HF dataset's PIL images into normalized tensors."""

    batch["pixel_values"] = [TRANSFORM(img.convert("L")) for img in batch["image"]]
    return batch


def _collate(examples: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate the input examples into pixel values and labels."""

    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    labels = torch.tensor([e["label"] for e in examples])
    return pixel_values, labels


def _get_dataloader(split: str, batch_size: int, shuffle: bool = False) -> DataLoader:
    """Load an MNIST split from the Hugging Face Hub and wrap it in a DataLoader.

    Args:
        split: The dataset split to load.
        batch_size: The batch size for the DataLoader.
        shuffle: Whether to shuffle the data.

    Returns:
        DataLoader: The wrapped DataLoader.
    """

    dataset = load_dataset(HF_DATASET_ID, split=split)
    dataset = dataset.with_transform(_transform_batch)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate)


class MNISTClassifier(MLModule):
    """CNN digit classifier for MNIST. Implements all four runner commands."""

    def __init__(self) -> None:
        self._model: Net | None = None

    def _load_model(self) -> Net:
        if self._model is not None:
            return self._model

        model = Net().to(DEVICE)
        if MODEL_PATH.exists():
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        else:
            print(
                f"Warning: no trained weights found at {MODEL_PATH}. "
                "Using randomly initialized weights. Run 'train' first.",
                file=sys.stderr,
            )
        self._model = model
        return model

    def train(self, args: list[str]) -> None:
        epochs = int(args[0]) if len(args) > 0 else 3
        batch_size = int(args[1]) if len(args) > 1 else 64
        lr = float(args[2]) if len(args) > 2 else 1.0

        train_loader = _get_dataloader("train", batch_size=batch_size, shuffle=True)

        model = self._load_model()
        optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

        model.train()
        epoch_bar = tqdm.tqdm(range(1, epochs + 1), desc="Epochs", unit="epoch", position=0)
        for epoch in epoch_bar:
            batch_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch", position=1, leave=False)
            for data, target in batch_bar:
                data, target = data.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                batch_bar.set_postfix(loss=f"{loss.item():.4f}")
            scheduler.step()
            epoch_bar.set_postfix(loss=f"{loss.item():.4f}")

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

    def test(self, args: list[str]) -> None:
        batch_size = int(args[0]) if len(args) > 0 else 1000

        test_loader = _get_dataloader("test", batch_size=batch_size, shuffle=False)

        model = self._load_model()
        model.eval()

        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in tqdm.tqdm(test_loader, desc="Evaluating", unit="batch"):
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        n = len(test_loader.dataset)
        test_loss /= n
        accuracy = 100.0 * correct / n
        print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{n} ({accuracy:.2f}%)")

    def inference(self, args: list[str]) -> None:
        if len(args) < 1:
            print(
                "Usage: python3 -m modules.runner inference modules.deep-learning.mnist <path/to/image>",
                file=sys.stderr,
            )
            sys.exit(1)

        image_path = Path(args[0])
        if not image_path.exists():
            print(f"Error: image not found: {image_path}", file=sys.stderr)
            sys.exit(1)

        image = Image.open(image_path).convert("L").resize((28, 28))
        tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

        model = self._load_model()
        model.eval()
        with torch.no_grad():
            output = model(tensor)
            pred = int(output.argmax(dim=1).item())
            confidence = float(output.exp().max().item())

        print(f"Predicted digit: {pred} (confidence: {confidence * 100:.2f}%)")

    def view(self, args: list[str]) -> None:
        from torchview import draw_graph

        model = self._load_model()
        output_path = Path(args[0]) if len(args) > 0 else MODULE_DIR / "mnist_architecture"
        dpi = int(args[1]) if len(args) > 1 else 300

        graph = draw_graph(model, input_size=(1, 1, 28, 28), device=DEVICE, expand_nested=True)
        graph.visual_graph.attr(dpi=str(dpi))
        graph.visual_graph.render(str(output_path), format="png", cleanup=True)
        print(f"Model architecture diagram saved to {output_path}.png")
