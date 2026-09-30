# Introduction to Artificial Intelligence

This repository serves as a storage and versioning system for the *Introduction to Artificial Intelligence* course at SUNY Ulster. It is split into two main sections:

* `assignments/` holds starter and demo code for the class' assignments.
* `modules/` holds sample code related to class lectures.
* `models/` holds the demo code associated with class models.
  * Demos with multiple source files are grouped into subdirectories of the parent module.

These code snippets will be shown throughout the semester as we build upon our understanding on machine learning models and concepts, but you are welcome to look now!

## Getting Started

What you will need:

* git
* python >= 3.10
* uv (optional)

See: [how to install git](https://github.com/git-guides/install-git), [how to install python](https://www.python.org/downloads), and [how to install uv](https://docs.astral.sh/uv/getting-started/installation).

1. Clone this repository.

    ```bash
    git clone https://github.com/SUNY-Ulster-Computer-Science/Introduction-to-Artificial-Intelligence.git
    ```

2. Create a virtual environment in the project directory and install packages.

    If using pip:

    ```bash
    python3 -m venv .venv
    pip install .
    ```

    if Using uv:

    ```bash
    uv venv
    uv sync
    ```

    Do not forget to [activate](https://docs.python.org/3/tutorial/venv.html#creating-virtual-environments) your virtual environment.
