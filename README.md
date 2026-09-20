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

```

## Technologies

Python · PyTorch · TorchVision · Consistency Models · Generative AI · Adversarial Training · U-Net · Computer Vision · Visdom · FID

## Running an experiment

The example training configuration is provided in:

```bash
run_cm_d.sh
```

Before running it, update the dataset paths and GPU configuration for your environment.

The training script supports options such as:

```text
--data_dir
--test_data_dir
--image_size
--batch_size
--max_steps
--lr
--iter_size
--device_cuda
```

Example:

```bash
python3 cm_d.py \
  --data_dir /path/to/train/data \
  --test_data_dir /path/to/test/data \
  --image_size 64 64 \
  --batch_size 8 \
  --max_steps 200000 \
  --lr 1e-4 \
  --device_cuda cuda:0
```

## Evaluation

The experimental pipeline includes:

- visual inspection of generated samples
- consistency sampling at multiple noise levels
- Fréchet Inception Distance (FID) evaluation on test data
- tracking of generator, discriminator, and consistency losses

## Status

This repository contains **research and experimental code** developed for exploring adversarial consistency models. It is not intended as a production-ready library.

## Author

**Ru Wang Pujos**

Machine Learning R&D Engineer  
Computer Vision · Deep Learning · Generative AI

Portfolio: https://wr0124.github.io/
