# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

MNIST GAN implementation in TensorFlow/Keras. A DCGAN that generates handwritten digit images, based on the TensorFlow DCGAN tutorial with architecture optimizations.

## Build and Run Commands

```bash
# Create virtual environment (Python 3.10-3.12)
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# For macOS Apple Silicon GPU support:
pip install tensorflow==2.18.0 tensorflow-metal==1.2.0

# Train the GAN
python -c "from src import train_gan; train_gan(epochs=50)"

# Or use the notebook
jupyter notebook notebooks/training_demo.ipynb
```

## Architecture

### Generator
- Input: 100-dim latent vector
- Projection: Dense(3*3*112) + Reshape
- TransConv layers: 112 -> 56 -> 28 -> 1 filters
- Activation: BatchNorm + LeakyReLU(0.2), tanh output
- Output: 28x28x1 image in [-1, 1]

### Discriminator (Simplified - matches TensorFlow tutorial)
- Input: 28x28x1 image
- Conv layers: 64 -> 128 filters
- Activation: LeakyReLU(0.2) + Dropout(0.3)
- Output: single logit (no BatchNorm)

### Key Hyperparameters
| Parameter | Value |
|-----------|-------|
| Batch size | 100 |
| G learning rate | 0.0002 |
| D learning rate | 0.00002 (10:1 ratio) |
| LR decay | 0.96 every 1000 steps |
| Adam beta1 | 0.5 |
| Loss | BinaryCrossentropy(from_logits=True) |

## Project Structure

```
src/
├── __init__.py          # Package exports
├── models/
│   ├── generator.py     # Generator network
│   └── discriminator.py # Discriminator network
├── gan.py               # MNISTGAN with custom train_step
├── trainer.py           # Training orchestration
├── callbacks.py         # Progress and sample callbacks
└── utils.py             # Device detection, image utilities
notebooks/
└── training_demo.ipynb  # Interactive training demo
```

## Design Notes

- Uses TTUR (Two-Timescale Update Rule) with 10:1 G/D learning rate ratio
- Exponential learning rate decay for training stability
- BinaryCrossentropy(from_logits=True) for numerical stability
- Simplified discriminator (2 conv layers, dropout, no BatchNorm) for stable training
- Auto-detects GPU (Metal on Mac, CUDA on Windows/Linux)
