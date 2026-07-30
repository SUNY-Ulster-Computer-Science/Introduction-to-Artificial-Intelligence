"""Example module for the runner: a simple CNN trained on MNIST."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from modules.runner.base import MLModule

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODULE_DIR = Path(__file__).parent
MODEL_PATH = MODULE_DIR / "mnist_model.pt"
DATA_DIR = MODULE_DIR / "data"

TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
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

        train_set = datasets.MNIST(str(DATA_DIR), train=True, download=True, transform=TRANSFORM)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

        model = self._load_model()
        optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

        model.train()
        for epoch in range(1, epochs + 1):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % 100 == 0:
                    print(
                        f"Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.4f}"
                    )
            scheduler.step()

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

    def test(self, args: list[str]) -> None:
        batch_size = int(args[0]) if len(args) > 0 else 1000

        test_set = datasets.MNIST(str(DATA_DIR), train=False, download=True, transform=TRANSFORM)
        test_loader = DataLoader(test_set, batch_size=batch_size)

        model = self._load_model()
        model.eval()

        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
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
