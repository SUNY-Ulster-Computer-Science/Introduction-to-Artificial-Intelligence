# Models

This directory contains runnable ML model demos, organized by domain. Each model is driven by a shared CLI runner.

## Available Models

| Dotted Path                      | Architecture | Task                 | Dataset                    |
|----------------------------------|--------------|----------------------|----------------------------|
| `models.cv.cnn_mnist`            | CNN          | Image classification | MNIST (handwritten digits) |
| `models.dl.dnn_mnist`            | DNN          | Image classification | MNIST (handwritten digits) |
| `models.nlp.rnn_imdb`            | RNN          | Sentiment analysis   | IMDB (movie reviews)       |
| `models.nlp.lstm_imdb`           | LSTM         | Sentiment analysis   | IMDB (movie reviews)       |
| `models.nlp.rnn_wikitext`        | RNN          | Language modeling    | WikiText-2                 |
| `models.nlp.lstm_wikitext`       | LSTM         | Language modeling    | WikiText-2                 |
| `models.lm.transformer_wikitext` | Transformer  | Language modeling    | WikiText-2                 |

Datasets are downloaded automatically from the [Hugging Face Hub](https://huggingface.co/datasets) on first use.

## Runner CLI

All models are run through a single CLI entry point. Run from the project root:

```bash
python3 -m models.runner <command> <dotted.path> [args...]
```

To see all available models and which commands each supports:

```bash
python3 -m models.runner list
```

Pass `--help` to any command for its full argument reference:

```bash
python3 -m models.runner train models.cv.cnn_mnist --help
```

### Commands

| Command     | Description                                                                            |
|-------------|----------------------------------------------------------------------------------------|
| `list`      | Print all discovered models and their supported commands                               |
| `train`     | Train the model and save its weights to disk                                           |
| `test`      | Evaluate the trained model on a held-out test set and report accuracy or perplexity    |
| `inference` | Run a single prediction (an image, a text string, or a prompt, depending on the model) |
| `view`      | Render a diagram of the model's architecture using torchview and Graphviz              |

> **Note:** `test` and `inference` load saved weights from a prior `train` run. If you have not trained a model yet, you will fall back to random weights.

> **Note:** `view` requires [Graphviz](https://graphviz.org/download/) to be installed at the system level (not just via pip) in addition to the Python packages.

### Examples

```bash
# List all models and their available commands
python3 -m models.runner list

# Train the CNN on MNIST
python3 -m models.runner train models.cv.cnn_mnist

# Evaluate the trained CNN on the MNIST test set
python3 -m models.runner test models.cv.cnn_mnist

# Classify a handwritten digit image
python3 -m models.runner inference models.cv.cnn_mnist path/to/image.png

# Classify a movie review with the LSTM
python3 -m models.runner inference models.nlp.lstm_imdb "This movie was fantastic!"

# Generate text from a prompt with the Transformer
python3 -m models.runner inference models.lm.transformer_wikitext "The history of"

# View the CNN's architecture diagram
python3 -m models.runner view models.cv.cnn_mnist
```
