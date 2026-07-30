# ML Module Runner

A single CLI entry point for running commands on a machine-learning module.

## Usage

Run from the project root:

```bash
python3 -m modules.runner <command> <module.path> [args...]
```

- `<command>` is one of: `inference`, `test`, `train`, `view`
- `<module.path>` is a dotted path resolved to a `.py` file relative to the current directory. To run the file `computer-vision/cnn.py`, use `modules.deep-learning.mnist`.
- `[args...]` are passed straight through to the module's command function.

### Examples

```bash
# Run inference on an image
python3 -m modules.runner inference modules.deep-learning.mnist /path/to/image.png

# Train the model (epochs=5, batch_size=64, lr=1.0)
python3 -m modules.runner train modules.deep-learning.mnist 5 64 1.0

# Evaluate on the test set
python3 -m modules.runner test modules.deep-learning.mnist

# Render a diagram of the model architecture (via torchview and graphviz)
python3 -m modules.runner view modules.deep-learning.mnist
```

## Installing dependencies

```bash
pip install -e .
# Or
uv sync
```
