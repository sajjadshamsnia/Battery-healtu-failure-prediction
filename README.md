# Battery Health/Failure Prediction Model

A binary classifier that predicts battery health/failure status from key operational parameters, using a neural network built from scratch with NumPy.

## Overview

This project trains a single-hidden-layer neural network (implemented manually with forward/backward propagation — no deep learning framework) to classify battery condition as **healthy** or **unhealthy/failing** based on measurable battery characteristics.

## Dataset

- **File:** `battery_synthetic_dataset.csv`
- **Type:** Synthetic dataset
- **Target column:** `label` (binary: 0 = unhealthy/fail, 1 = healthy)
- **Features:**
  - Voltage
  - Temperature
  - Battery life
  - Charge cycle count

> Note: Since the dataset is synthetic, results should be treated as a proof-of-concept rather than validated performance on real-world battery data.

## Model Architecture

- **Type:** Feedforward neural network (1 hidden layer), implemented from scratch in NumPy
- **Input layer:** size = number of features
- **Hidden layer:** 10 neurons, sigmoid activation
- **Output layer:** 1 neuron, sigmoid activation (binary classification)
- **Loss/training:** Manual gradient descent via backpropagation
- **Learning rate:** 1e-3
- **Epochs:** 10,000
- **Preprocessing:** Features standardized using `sklearn.preprocessing.StandardScaler`

## Results

| Metric | Value |
|---|---|
| Accuracy | 90.5% |
| R² score | 0.738 |
| MSE | 0.061 |

**Classification report:**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| 0 (unhealthy) | 0.84 | 0.92 | 0.88 | 73 |
| 1 (healthy) | 0.95 | 0.90 | 0.92 | 127 |

The model shows strong recall on the "unhealthy" class (0.92), which is important for this use case — catching batteries that are actually degrading matters more than occasionally over-flagging a healthy one.