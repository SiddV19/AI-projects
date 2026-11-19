# AI Stock Market Predictor

**Python · PyTorch · Google Colab · Time-series modeling**

[← AI projects](../README.md) · [Open in Google Colab](https://colab.research.google.com/drive/1OG71mQ3pkkOUsqxHZ-SIsXXWK1x9irTk#scrollTo=tGjHZYtC0qw2) · [Source notes](SOURCE.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OG71mQ3pkkOUsqxHZ-SIsXXWK1x9irTk#scrollTo=tGjHZYtC0qw2)

A stock-price prediction learning project developed in Google Colab during the Harvard AI bootcamp. The shared code trains a PyTorch time-series model in mini-batches, generates predictions, and reverses target scaling.

## Available source

The [training and inference excerpt](training_excerpt.py) preserves the Colab cell shared by the author. It is a notebook fragment that depends on setup from earlier cells.

The [original notebook is hosted in Google Colab](https://colab.research.google.com/drive/1OG71mQ3pkkOUsqxHZ-SIsXXWK1x9irTk#scrollTo=tGjHZYtC0qw2). This repository contains the earlier shared excerpt rather than a standalone `.ipynb` copy. Open Colab for the notebook workflow; later Colab edits are not automatically synchronized to `training_excerpt.py`.

## What the shared cell does

1. Converts NumPy training and test arrays to floating-point PyTorch tensors.
2. Wraps them in the notebook's `TimeSeriesDataset` and creates data loaders with batch size 16 and `shuffle=False`.
3. Configures 50 epochs of training, computing predictions and loss, clearing gradients, running backpropagation, and updating model parameters.
4. Generates predictions for the test inputs and converts the predicted values to NumPy.
5. Uses the notebook's fitted target scaler to restore predictions and test targets to their original scale.

The printed loss is from the final training batch in each epoch. It is not an epoch average or a test-set evaluation.

## Running the excerpt

Run this cell after the original notebook's setup. It requires:

- NumPy arrays named `x_train`, `y_train`, `x_test`, and `y_test`.
- The `TimeSeriesDataset` class and a configured PyTorch `model`.
- A loss function named `criterion` and an `optimizer` connected to the model parameters.
- A fitted target scaler named `scaler_pred`.

The excerpt retains the original assumption that inputs and model are on CPU and that `y_batch.unsqueeze(1)` matches the model output shape. It should not be run as a standalone script. Use the linked Colab notebook for its setup cells and notebook environment.

## Reproducibility status

The excerpt alone does not include the original data source, sequence length, split dates, scaler-fitting procedure, model definition, or saved results. No accuracy, test-error, or trading-performance result is claimed here. The full training run has not been reproduced from this excerpt.

If a notebook copy is added to GitHub, retain its course and starter-code attribution and document its dependencies and evaluation results.
