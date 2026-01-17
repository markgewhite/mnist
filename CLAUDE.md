# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MNIST GAN implementation in TensorFlow/Keras. A DCGAN (Deep Convolutional GAN) that generates handwritten digit images, ported from a MATLAB implementation.

## Architecture

- **Generator**: Latent vector z (100-dim) → Dense → Reshape(3,3,112) → TransposedConv2D upsampling (filters: 56→28→1) → 28x28x1 image (tanh output)
- **Discriminator**: 28x28x1 image → Dropout(0.5) → Conv2D downsampling (filters: 28→56→112→224→1) → logit output
- **Training**: Custom `train_step` using `tf.GradientTape` for alternating generator/discriminator updates with binary cross-entropy loss and 30% label flipping

## Project Structure

```
src/
├── __init__.py          # Package exports
├── models/
│   ├── __init__.py
│   ├── base.py          # Abstract base network class
│   ├── generator.py     # Generator network
│   └── discriminator.py # Discriminator network
├── gan.py               # MNISTGAN class combining both networks
├── trainer.py           # Training orchestration
├── callbacks.py         # Sample generation callback
└── utils.py             # Image saving, latent sampling
notebooks/
└── training_demo.ipynb  # Visual training demonstration
outputs/
└── generated_samples/   # Sample outputs during training
```

## Build and Run Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train the GAN (quick example)
python -c "from src import train_gan; train_gan(epochs=50)"

# Run from notebook for visualization
jupyter notebook notebooks/training_demo.ipynb
```

## Key TensorFlow Patterns

- Custom Keras Model subclassing with `train_step` override
- `tf.GradientTape` for manual gradient computation (required for multi-network GAN training)
- `tf.data.Dataset` pipelines for efficient data loading
- Images normalized to [-1, 1] range to match tanh activation

## Design Principles

- Object-oriented design: Generator and Discriminator as proper classes, not just builder functions
- Separation of concerns: Training logic in Trainer class, model definitions separate
- Callback system for generating sample images during training
