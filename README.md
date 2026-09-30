# Introduction to Artificial Intelligence

This repository serves as a storage and versioning system for the *Introduction to Artificial Intelligence* course at SUNY Ulster. It is split into three main sections:

* `assignments/` holds starter code for class assignments.
* `modules/` holds sample code related to class lectures.
* `models/` holds runnable model demos. See the [models README](models/README.md) for details.

Code snippets will be introduced throughout the course as we build upon our understanding of machine learning models and concepts.

## Dependencies

Installing this project will pull in the following key packages:

* [PyTorch](https://pytorch.org/): tensor operations and model training
* [Torchvision](https://pytorch.org/vision/stable/index.html): image datasets and transforms
* [Hugging Face Transformers](https://huggingface.co/docs/transformers): pretrained NLP models
* [Hugging Face Datasets](https://huggingface.co/docs/datasets): dataset loading utilities
* [Matplotlib](https://matplotlib.org/): plotting and visualization
* [Pillow](https://pillow.readthedocs.io/): image loading and manipulation
* [Graphviz](https://graphviz.org/) + [Torchview](https://github.com/mert-kurttutan/torchview): model architecture diagrams

> **Note:** Training models (especially RNNs, LSTMs, and Transformers) can be slow on CPU. You do not need a GPU for this course, but expect some training runs to take several minutes.

## Getting Started

What you will need:

* git
* Python >= 3.10
* uv (optional, but recommended)

See: [how to install git](https://github.com/git-guides/install-git), [how to install Python](https://www.python.org/downloads), and [how to install uv](https://docs.astral.sh/uv/getting-started/installation).

1. Clone this repository.

    ```bash
    git clone https://github.com/SUNY-Ulster-Computer-Science/Introduction-to-Artificial-Intelligence.git
    cd Introduction-to-Artificial-Intelligence
    ```

2. Create a virtual environment and install packages.

   **Using uv (recommended):**

    ```bash
    uv venv
    uv sync
    ```

   **Using pip:**

    ```bash
    python3 -m venv .venv
    pip install .
    ```

3. Activate your virtual environment before running any code.

   **The command you run will depend on your operating system:**

   ```bash
   source .venv/bin/activate  # macOS / Linux

   .venv\Scripts\activate.bat  # Windows (cmd)

   .venv\Scripts\Activate.ps1  # Windows (PowerShell)
   ```

   You should see `(.venv)` appear in your terminal prompt when the environment is active.
