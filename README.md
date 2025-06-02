# Fashion MNIST GAN with TensorFlow

This repository contains an implementation of a **Generative Adversarial Network (GAN)** using **TensorFlow** to generate fashion images based on the **Fashion MNIST** dataset.

## 📌 Project Overview

A **Generative Adversarial Network (GAN)** is a type of neural network architecture composed of two competing models:

- **Generator**: Learns to create realistic data (in this case, fake fashion images).
- **Discriminator**: Learns to distinguish between real data (from the dataset) and fake data (from the generator).

These two models are trained simultaneously in a **minimax game**:
- The generator tries to fool the discriminator by producing increasingly realistic images.
- The discriminator tries not to be fooled, learning to better distinguish real from fake.

Over time, the generator ideally becomes good enough to produce images indistinguishable from real ones.

## 🧠 Model Architecture

### Generator
- Takes random noise as input (usually from a normal distribution).
- Outputs a 28x28 grayscale image (same shape as Fashion MNIST samples).
- Typically uses **Dense** layers followed by **Batch Normalization** and **ReLU/LeakyReLU** activations.

### Discriminator
- Takes an image (real or fake) as input.
- Outputs a probability indicating whether the image is real.
- Typically uses **Dense** layers with **LeakyReLU** activations and **Dropout** for regularization.

## 🧰 Tech Stack

- Python
- TensorFlow 2.x
- NumPy
- Matplotlib (for image visualization)

## 📊 Dataset

We use the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset, which contains 70,000 28x28 grayscale images of 10 different fashion item classes (t-shirts, shoes, bags, etc.).
