# MNIST GAN

A DCGAN (Deep Convolutional GAN) implementation in TensorFlow/Keras that generates handwritten digit images. Based on the TensorFlow DCGAN tutorial with architecture optimizations.

## Architecture

```mermaid
graph LR
    subgraph Generator
        Z[Latent z<br/>100-dim] --> D1[Dense<br/>1008]
        D1 --> R[Reshape<br/>3×3×112]
        R --> TC1[TransConv<br/>5×5, 56<br/>valid]
        TC1 --> BN1[BN + LeakyReLU]
        BN1 --> TC2[TransConv<br/>5×5, 28<br/>stride 2]
        TC2 --> BN2[BN + LeakyReLU]
        BN2 --> TC3[TransConv<br/>5×5, 1<br/>stride 2]
        TC3 --> T[tanh]
        T --> IMG[Image<br/>28×28×1]
    end

    subgraph Discriminator
        I[Image<br/>28×28×1] --> C1[Conv 5×5, 64<br/>stride 2]
        C1 --> LR1[LeakyReLU]
        LR1 --> DO1[Dropout 0.3]
        DO1 --> C2[Conv 5×5, 128<br/>stride 2]
        C2 --> LR2[LeakyReLU]
        LR2 --> DO2[Dropout 0.3]
        DO2 --> F[Flatten]
        F --> D[Dense 1]
        D --> L[Logit]
    end
```

### Generator
| Layer | Output Shape | Description |
|-------|--------------|-------------|
| Input | (batch, 100) | Latent vector |
| Dense | (batch, 1008) | Project to 3×3×112 |
| Reshape | (batch, 3, 3, 112) | Spatial structure |
| TransConv2D | (batch, 7, 7, 56) | 5×5, valid padding |
| BatchNorm + LeakyReLU | (batch, 7, 7, 56) | α=0.2 |
| TransConv2D | (batch, 14, 14, 28) | 5×5, stride 2 |
| BatchNorm + LeakyReLU | (batch, 14, 14, 28) | α=0.2 |
| TransConv2D | (batch, 28, 28, 1) | 5×5, stride 2 |
| tanh | (batch, 28, 28, 1) | Output in [-1, 1] |

### Discriminator
| Layer | Output Shape | Description |
|-------|--------------|-------------|
| Input | (batch, 28, 28, 1) | Image |
| Conv2D | (batch, 14, 14, 64) | 5×5, stride 2 |
| LeakyReLU + Dropout | (batch, 14, 14, 64) | α=0.2, p=0.3 |
| Conv2D | (batch, 7, 7, 128) | 5×5, stride 2 |
| LeakyReLU + Dropout | (batch, 7, 7, 128) | α=0.2, p=0.3 |
| Flatten | (batch, 6272) | |
| Dense | (batch, 1) | Logit output |

## Training Hyperparameters

Tuned for stable training with TTUR (Two-Timescale Update Rule):

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| G learning rate | 0.0002 |
| D learning rate | 0.00002 (10:1 ratio) |
| LR decay | 0.96 every 1000 steps |
| β₁ | 0.5 |
| β₂ | 0.999 |
| Batch size | 100 |
| Epochs | 50 |
| Loss | Binary cross-entropy (from_logits) |

## Installation

### Basic Installation

```bash
pip install -r requirements.txt
```

### GPU Acceleration

TensorFlow automatically detects and uses available GPUs. Platform-specific setup:

**macOS (Apple Silicon M1/M2/M3):**
```bash
# Requires TensorFlow 2.18.x for Metal support
pip install tensorflow==2.18.0 tensorflow-metal==1.2.0
```

**Windows/Linux (NVIDIA CUDA):**
```bash
# Install CUDA toolkit and cuDNN, then:
pip install tensorflow[and-cuda]
```

The code auto-detects available devices at runtime:
```python
from src import print_device_info
print_device_info()
# Output: Platform, TensorFlow version, GPU availability
```

## Usage

### Quick Start

```python
from src import train_gan

# Train with default MATLAB-matched parameters
trainer = train_gan(epochs=50)

# Generate samples
samples = trainer.generate_samples(num_samples=25)
```

### Custom Training

```python
from src import GANTrainer

# Initialize trainer with custom parameters
trainer = GANTrainer(
    batch_size=100,
    epochs=50,
    g_learning_rate=0.0002,
    d_learning_rate=0.00002
)

# Load data and build model
trainer.load_data()
trainer.build_model()

# Train with callbacks
callbacks = trainer.get_default_callbacks(sample_frequency=100)
history = trainer.train(callbacks=callbacks)
```

### Using the Model Directly

```python
from src import MNISTGAN
import tensorflow as tf

# Create and compile
gan = MNISTGAN(latent_dim=100)
gan.compile()

# Load MNIST and train
(train_images, _), _ = tf.keras.datasets.mnist.load_data()
train_images = (train_images.reshape(-1, 28, 28, 1) - 127.5) / 127.5
dataset = tf.data.Dataset.from_tensor_slices(train_images).batch(60)

gan.fit(dataset, epochs=50)

# Generate images
z = tf.random.normal((25, 100))
generated = gan.generator(z, training=False)
```

### Jupyter Notebook

For interactive training with visualizations:

```bash
jupyter notebook notebooks/training_demo.ipynb
```

## Project Structure

```
mnist/
├── src/
│   ├── __init__.py          # Package exports
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py          # Abstract base network class
│   │   ├── generator.py     # Generator network
│   │   └── discriminator.py # Discriminator network
│   ├── gan.py               # MNISTGAN class with custom train_step
│   ├── trainer.py           # GANTrainer orchestration
│   ├── callbacks.py         # Sample generation callback
│   └── utils.py             # Image utilities
├── notebooks/
│   └── training_demo.ipynb  # Interactive training demo
├── outputs/
│   └── generated_samples/   # Generated images during training
├── requirements.txt
└── README.md
```

## Key Design Decisions

| Aspect | This Implementation |
|--------|---------------------|
| Projection size | 3×3×112 |
| Filter sequence | 64, 128 (discriminator) |
| Discriminator | 2 conv layers, dropout, no BatchNorm |
| TTUR | 10:1 G/D learning rate ratio |
| LR decay | Exponential, 0.96 per 1000 steps |
| Loss | BinaryCrossentropy(from_logits=True) |

## Requirements

- Python 3.10 - 3.12
- TensorFlow 2.18.x (for Mac GPU) or TensorFlow >= 2.15.0 (CPU/CUDA)
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0
- Jupyter >= 1.0.0

### Platform Compatibility

| Platform | GPU Support | TensorFlow Version |
|----------|-------------|-------------------|
| macOS (Apple Silicon) | Metal | 2.18.0 + tensorflow-metal 1.2.0 |
| macOS (Intel) | None | >= 2.15.0 |
| Windows | CUDA | >= 2.15.0 |
| Linux | CUDA | >= 2.15.0 |

## License

MIT
