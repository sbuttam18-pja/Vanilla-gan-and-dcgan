# Vanilla GAN and DCGAN Implementations using PyTorch

Implementations of **Vanilla Generative Adversarial Network (GAN)** and **Deep Convolutional Generative Adversarial Network (DCGAN)** using PyTorch.

This project explores generative deep learning and adversarial training through two different GAN architectures for image generation.

## Overview

Generative Adversarial Networks (GANs) consist of two neural networks trained against each other:

- **Generator:** Generates synthetic images from random noise.
- **Discriminator:** Distinguishes between real images and generated images.

The generator learns to produce realistic images, while the discriminator learns to identify generated images. Through adversarial training, both networks improve over time.

This repository contains implementations of Vanilla GAN and DCGAN to understand their architectures, training process, and image generation capabilities.

## Implementations

### 1. Vanilla GAN

A basic GAN implementation using fully connected neural network layers.

**Features**
- Generator and discriminator architectures
- Fully connected neural networks
- Adversarial training
- Image generation from random noise
- Weight initialization

### 2. DCGAN

A Deep Convolutional GAN implementation using convolutional and transposed convolutional layers.

**Features**
- Convolutional discriminator
- Transposed convolutional generator
- Batch normalization
- Leaky ReLU activation
- Image generation using convolutional neural networks
- DCGAN weight initialization

## Tech Stack

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Pillow
- Jupyter Notebook

## Project Structure

```text
vanilla-gan-and-dcgan/
│
├── dc_gan/
│   ├── __init__.py
│   ├── model.py
│   └── train.py
│
├── vanilla_gan/
│   ├── __init__.py
│   ├── model.py
│   └── train.py
│
├── notebooks/
│   ├── dcgan.ipynb
│   └── vanilla_gan.ipynb
│
├── results/
│   ├── dc_gan/
│   └── vanilla_gan/
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

The implementations use the **CelebA dataset** for image generation.

The dataset contains face images and is used to train the generator and discriminator.

The dataset is not included in this repository.

Download the dataset separately and configure the dataset path in the notebooks or training scripts before running the training process.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/sbuttam18-pja/vanilla-gan-and-dcgan.git
```

### 2. Navigate to the project directory

```bash
cd vanilla-gan-and-dcgan
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Vanilla GAN

Run the Vanilla GAN initialization script:

```bash
python -m vanilla_gan.train
```

The script initializes the generator and discriminator and tests image generation.

### DCGAN

Run the DCGAN initialization script:

```bash
python -m dc_gan.train
```

The script initializes the generator and discriminator and tests image generation.

### Jupyter Notebooks

The complete notebook implementations are available in the `notebooks/` directory.

Open the notebooks to explore the model architectures, training process, and generated images.

```bash
jupyter notebook
```

Run the notebook cells sequentially after configuring the dataset path.

## Model Architecture Comparison

| Feature | Vanilla GAN | DCGAN |
|---|---|---|
| Generator | Fully connected layers | Transposed convolutional layers |
| Discriminator | Fully connected layers | Convolutional layers |
| Image representation | Flattened images | Spatial image tensors |
| Batch normalization | Supported | Supported |
| Image generation | Yes | Yes |

## Results

Generated image samples and training visualizations will be added to the `results/` directory.

### Vanilla GAN

Generated samples:

```text
results/vanilla_gan/
```

### DCGAN

Generated samples:

```text
results/dc_gan/
```

## Learning Objectives

- Understand the fundamental architecture of GANs.
- Implement generators and discriminators using PyTorch.
- Explore adversarial training.
- Understand the difference between fully connected and convolutional GAN architectures.
- Generate synthetic images using deep learning.
- Experiment with GAN model initialization and training configurations.

## Future Improvements

- Improve generated image quality.
- Experiment with different latent dimensions.
- Compare training stability between architectures.
- Add training loss visualizations.
- Evaluate generated images using suitable metrics.
- Experiment with additional GAN architectures.

## Author

**Uttam SB**

GitHub: https://github.com/sbuttam18-pja

## License

This project is intended for educational and experimental purposes.
