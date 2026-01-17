# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MNIST GAN implementation in TensorFlow/Keras. A DCGAN (Deep Convolutional GAN) that generates handwritten digit images, ported from a MATLAB implementation.

**Status**: Training diverges after 1-2 epochs. Discriminator dominates. See Troubleshooting section below.

## MATLAB Source Files

The original MATLAB implementation is at:
- `/Users/markgewhite/Documents/MyFiles/Academia/MATLAB/Postdoc/Examples/GAN/mnistGAN.m`
- `/Users/markgewhite/Documents/MyFiles/Academia/MATLAB/Postdoc/Examples/GAN/modelGradients.m`
- `/Users/markgewhite/Documents/MyFiles/Academia/MATLAB/Postdoc/Examples/GAN/ganLoss.m`

**Note**: User cannot verify MATLAB code works (license expired). Possible the MATLAB code was modified and no longer works either.

## Architecture (Exact MATLAB Match)

### Generator
```
featureInputLayer(100)
projectAndReshapeLayer([3,3,112], 100)      -> Dense(1008) + Reshape(3,3,112)
transposedConv2dLayer(5, 56)                -> BN -> ReLU   [output: 7x7x56]
transposedConv2dLayer(5, 28, stride=2, same) -> BN -> ReLU  [output: 14x14x28]
transposedConv2dLayer(5, 1, stride=2, same)  -> tanh        [output: 28x28x1]
```
- 2 BatchNorm layers (after tconv1, tconv2)
- No BN on output layer (DCGAN convention)

### Discriminator
```
imageInputLayer([28,28,1], Normalization='none')
dropoutLayer(0.5)
conv2dLayer(5, 28, stride=2, same)           -> LeakyReLU(0.2)  [output: 14x14x28]
conv2dLayer(5, 56, stride=2, same)  -> BN    -> LeakyReLU(0.2)  [output: 7x7x56]
conv2dLayer(5, 112, stride=2, same) -> BN    -> LeakyReLU(0.2)  [output: 4x4x112]
conv2dLayer(5, 224, stride=2, same) -> BN    -> LeakyReLU(0.2)  [output: 2x2x224]
conv2dLayer(2, 1)                                               [output: 1x1x1]
```
- 3 BatchNorm layers (after conv2, conv3, conv4)
- No BN on first conv or output layer (DCGAN convention)
- Input dropout 0.5

### Training Hyperparameters (MATLAB-matched)
- Optimizer: Adam(lr=0.0002, beta1=0.5, beta2=0.999)
- Batch size: 60
- Epochs: 500
- Flip factor: 0.3 (flip 30% of real PROBABILITIES, not labels)
- Images normalized to [-1, 1]

## Project Structure

```
src/
├── __init__.py          # Package exports
├── models/
│   ├── __init__.py
│   ├── base.py          # Abstract base network class
│   ├── generator.py     # Generator network
│   └── discriminator.py # Discriminator network
├── gan.py               # MNISTGAN class with custom train_step
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
# Requires Python 3.12 (not 3.13 - numpy compilation issues)
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train the GAN
python -c "from src import train_gan; train_gan(epochs=50)"

# Or use notebook
jupyter notebook notebooks/training_demo.ipynb
```

## Current Training Issue

**Problem**: Training diverges after 1-2 epochs. Pattern observed:
- Epoch 1: G loss drops from ~2.6 to ~2.2, D loss rises from ~0.85 to ~0.97
- Epoch 2: G loss jumps down to ~1.6 then starts RISING to ~1.76
- Epoch 3+: G loss continues rising, D loss slowly drops

**Expected behavior** (from working MATLAB): G and D losses should show complementary oscillation, both hovering with some symmetry. G score and D score should both be near 0.5.

**Observed**: G score ~0.2 (discriminator confident fakes are fake), D score ~0.65 (discriminator winning).

## Troubleshooting History

### What We Verified
1. **Architecture matches MATLAB exactly** - layer by layer comparison done
2. **Batch size is 60** (not 5 - that was just test code in notebook)
3. **Discriminator output shape correct**: (batch, 1) - one logit per image

### What We Tried (None Fixed the Issue)

1. **Fixed gradient computation** - Both G and D gradients now computed from SAME fake images in single forward pass (matching MATLAB's modelGradients.m)

2. **Label smoothing** - Tried smoothing real labels to 0.9. Didn't help. Removed.

3. **Manual loss function** - Implemented exact MATLAB loss:
   ```python
   # MATLAB: lossDiscriminator = -mean(log(probReal)) - mean(log(1-probGenerated))
   d_loss = -tf.reduce_mean(tf.math.log(prob_real_flipped + eps)) \
            -tf.reduce_mean(tf.math.log(1 - prob_fake + eps))

   # MATLAB: lossGenerator = -mean(log(probGenerated))
   g_loss = -tf.reduce_mean(tf.math.log(prob_fake + eps))
   ```

4. **Probability flipping** (not label flipping) - MATLAB flips probabilities directly:
   ```python
   # MATLAB: probReal(:,:,:,idx) = 1 - probReal(:,:,:,idx)
   prob_real_flipped = prob_real * (1 - flip_mask) + (1 - prob_real) * flip_mask
   ```

5. **DCGAN weight initialization** - Added `RandomNormal(mean=0.0, stddev=0.02)` to all conv/dense layers

6. **BatchNorm momentum** - Changed from TensorFlow default 0.99 to 0.9 to match MATLAB's momentum=0.1 behavior

### Things NOT Yet Tried

1. **Two-Timescale Update Rule (TTUR)** - Use lower learning rate for discriminator (e.g., D: 0.0001, G: 0.0002)

2. **Train G multiple times per D step** - Common stabilization technique

3. **Spectral normalization** - Constrains discriminator Lipschitz constant

4. **Different GAN formulation** - Wasserstein GAN with gradient penalty

5. **Verify MATLAB actually works** - User's MATLAB license expired, cannot confirm original code works

## Key Implementation Details

### train_step in gan.py
- Uses nested `tf.GradientTape()` to compute both gradients simultaneously
- Probability flipping happens INSIDE the tape (on prob_real, not labels)
- Order: Generate fake -> D(real) -> D(fake) -> compute losses -> compute grads -> update D -> update G

### MATLAB vs TensorFlow Differences Investigated
| Aspect | MATLAB | TensorFlow | Status |
|--------|--------|------------|--------|
| BN momentum | 0.1 | 0.9 (was 0.99) | Fixed |
| Loss function | Manual log | Manual log | Fixed |
| Label/prob flipping | Flip probs | Flip probs | Fixed |
| Weight init | Default | N(0, 0.02) | Fixed |
| Gradient computation | Single pass | Single pass | Fixed |

## Design Notes

- Object-oriented design: Generator and Discriminator as proper Keras Model subclasses
- Custom `train_step` override for GAN-specific training logic
- `tf.data.Dataset` pipelines for efficient data loading
- Images normalized to [-1, 1] to match tanh activation range
