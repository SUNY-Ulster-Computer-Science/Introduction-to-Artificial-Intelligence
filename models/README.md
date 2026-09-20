# ML Module Runner

A single CLI entry point for running commands on a machine-learning module.

## Usage

Run from the project root:

```bash
python3 -m models.runner <command> <module.path> [args...]
```

- `<command>` is one of: `inference`, `test`, `train`, `view`
- `<models.path>` is a dotted path resolved to a `.py` file relative to the current directory. To run the file `cv/cnn_mnist.py`, use `models.cv.cnn_mnist`.
- `[args...]` are passed straight through to the model's command function.

### Examples

```bash
# Run inference on an image
python3 -m models.runner inference models.cv.cnn_mnist /path/to/image.png

# Train the model (epochs=5, batch_size=64, lr=1.0)
python3 -m models.runner train models.cv.cnn_mnist 5 64 1.0

# Evaluate on the test set
python3 -m models.runner test models.cv.cnn_mnist

# Render a diagram of the model architecture (via torchview and graphviz)
python3 -m models.runner view models.cv.cnn_mnist
```

## Installing dependencies

```bash
pip install -e .
# Or
uv sync
```
