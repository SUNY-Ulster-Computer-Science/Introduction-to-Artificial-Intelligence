"""
Representing Data as Tensors
Introduction to Artificial Intelligence

In this assignment, you will be working with different data formats and seeing how they are represented with tensors.

Parts:
  1. Numerical (tabular) data
  2. Text data
  3. Image data
"""

from pathlib import Path

import torchvision
from PIL import Image

# ==== #
# Part 1: Tabular Data
# ==== #

print("==== Tabular Data ====")

# Each row is [hours_studied, hours_slept, exam_score]
RAW_DATA = [
    [5.0, 7.5, 82.0],
    [2.0, 6.0, 61.0],
    [8.0, 8.0, 91.0],
    [3.5, 5.5, 70.0],
    [6.0, 7.0, 85.0],
    [1.0, 4.0, 52.0],
    [7.5, 8.5, 94.0],
    [4.0, 6.5, 74.0],
]

# Exercise 1.1
# Convert RAW_DATA into a PyTorch tensor called data_tensor.

data_tensor = None  # TODO

# Exercise 1.2
# Print the shape, dtype, and ndim of data_tensor.

print("Tensor info below:")
# TODO

# Exercise 1.3
# Using tensor slicing, extract:
#   hours_studied: all rows, first column
#   exam_scores: all rows, last column
#   first_three: first 3 rows, all columns

hours_studied = None  # TODO
exam_scores = None  # TODO
first_three = None  # TODO

print("Hours studied:", hours_studied)
print("Exam scores:", exam_scores)
print("First three samples:", first_three)

# Exercise 1.4
# Compute the mean exam score, max hours studied, and min hours slept using PyTorch built-ins (tensor.mean(), etc.).

mean_score = 0  # TODO
max_hours = 0  # TODO
min_sleep = 0  # TODO

print(f"Mean exam score:   {mean_score:.2f}")
print(f"Max hours studied: {max_hours:.2f}")
print(f"Min hours slept:   {min_sleep:.2f}")

# Exercise 1.5
# Normalize the exam_score column of data_tensor to [0, 1]
#   score_norm = score_percent / 100
# Do not normalize all values, only the exam_score column
# Hint: you can use tensor slicing to help

exam_score_norm = None  # TODO

print("Data with normalized exam scores:", exam_score_norm)


# ==== #
# Part 2: Text Data
# ==== #

# Text must be converted to numbers before a model can use it.
# The simplest approach is character-level encoding: assign a unique integer to every character in a vocabulary.

print("\n\n==== Text Data ====")

SAMPLE_TEXT = "The ships hung in the sky in much the same way that bricks don't."
print("Sample text: ", SAMPLE_TEXT)


def char_to_idx(char: str) -> int:
    if len(char) != 1:
        raise ValueError("Only one character allowed at a time! Got length ", len(char))

    return ord(char) - ord("A")


# Exercise 2.1
# Represent SAMPLE_TEXT as a list of character index integers using char_to_idx.
# Note that char_to_idx can only handle a single character at a time!
# Hint: List comprehension can be useful for this


sample_as_int_list = None  # TODO

print("Text as integers: ", sample_as_int_list)

# Exercise 2.2
# Encode SAMPLE_TEXT as a 1-D integer tensor.
# Shape should be (len(SAMPLE_TEXT),).

text_tensor = None  # TODO

print(f"Encoded shape:   {text_tensor.shape}")
print(f"First 10 values: {text_tensor[:10]}")

# Exercise 2.3
# Implement idx_to_char. It should take in a single integer and output the corresponding character
# Use idx_to_char to translate text_tensor back into the decoded_text string


def idx_to_char(idx: int) -> str:
    raise NotImplementedError  # TODO


decoded_text = ""  # TODO

assert decoded_text == SAMPLE_TEXT, f"Decoding failed!\n  Expected: {SAMPLE_TEXT}\n  Got: {decoded_text}"
print("Decoded text: ", decoded_text)


# ==== #
# Part 3: Image Data
# ==== #

# Images are represented as tensors of shape (C, H, W):
#   C = color channels, H = height, W = width

print("\n\n==== Image Data ====")

# Load the bundled sample image.
# torchvision.transforms.functional.to_tensor converts a PIL image from (H, W, C) uint8 [0, 255] to a float32
# tensor (C, H, W) with values in [0.0, 1.0].
current_dir = Path(__file__).resolve().parent
pil_image = Image.open(current_dir / "sample_image.png").convert("RGB")
sample_image = torchvision.transforms.functional.to_tensor(pil_image)

# Exercise 3.1
# Print the shape, dtype, min, and max of sample_image.

torchvision.transforms.ToPILImage()(sample_image).show()

print("Tensor info below:")
# TODO

# Exercise 3.2
# Extract the three color channels as separate tensors, then compute the mean of each.

red_channel = None  # TODO
green_channel = None  # TODO
blue_channel = None  # TODO

mean_red = 0  # TODO
mean_green = 0  # TODO
mean_blue = 0  # TODO

print(f"Mean intensity (R: {mean_red:.3f}  G: {mean_green:.3f}  B: {mean_blue:.3f})")

# Exercise 3.3
# Convert sample_image to grayscale by averaging across the channel dimension only.
# Hint: the channel dimension is dim zero.

grayscale_image = None  # TODO

torchvision.transforms.ToPILImage()(grayscale_image).show()

print(f"Grayscale intensity (R: {mean_red:.3f}  G: {mean_green:.3f}  B: {mean_blue:.3f})")

# Exercise 3.4
# Add a batch dimension so sample_image becomes shape (1, 3, 32, 32).
# Hint: Use torch unsqueeze on dimension zero

batched_image = None  # TODO

print(f"Batched shape: {batched_image.shape}")

# Exercise 3.5
# Create a random float tensor of shape (3, 64, 64) and represent it as an image
# Hint: Make sure your values are in the range [0, 1]

random_image = None  # TODO

torchvision.transforms.ToPILImage()(random_image).show()
