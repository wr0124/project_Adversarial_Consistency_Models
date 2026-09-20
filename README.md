# Adversarial Consistency Models

Experimental PyTorch project exploring the combination of **Improved Consistency Training** with an **adversarial discriminator** for image generation and editing.

The project investigates whether an adversarial objective can complement consistency-model training and improve generated image quality.

## Overview

The implementation includes:

- a U-Net-based consistency model
- an adversarial discriminator
- Improved Consistency Training
- pseudo-Huber consistency loss
- mask-conditioned image generation / editing
- multi-step consistency sampling
- gradient accumulation
- FID-based evaluation
- training and sample visualization with Visdom

## Method

The generator is trained using a combination of:

1. **Consistency loss**, encouraging predictions at different noise levels to remain consistent.
2. **Adversarial loss**, using a discriminator to distinguish generated samples from real images.

The training objective combines both terms so that the consistency model learns the denoising trajectory while also receiving an adversarial image-quality signal.

The repository also contains experiments with binary masks for localized generation and editing.

## Repository structure

```text
.
├── cm_d.py                    # Main adversarial consistency training pipeline
├── UNet.py                    # Consistency-model U-Net
├── UNet_discriminator.py      # Discriminator architecture
├── consistency_generator.py  # Consistency training utilities
├── inference.py               # Consistency sampling and editing
├── options.py                 # Command-line configuration
├── utils.py                   # Utility functions
└── run_cm_d.sh                # Example training command
