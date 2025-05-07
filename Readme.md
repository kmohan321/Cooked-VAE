# Variational Autoencoder (VAE) Project

## Overview

This project implements a Variational Autoencoder (VAE) from scratch for educational purposes and experimentation. The focus is on understanding the fundamentals of VAEs, optimizing the training process, and exploring architectural design choices.

## Features

- Custom-built VAE architecture
- Image reconstruction and generation capabilities 
- Experiment tracking with Weights & Biases (wandb)
- Trained on 128×128 pixel images
- Approximately 12 million parameters
- Dataset of ~15,000 images

## Mathematical Foundation: ELBO

The Evidence Lower Bound (ELBO) for a Variational Autoencoder is derived as follows:

Given a data point $x$ and latent variable $z$, the marginal likelihood can be written as:

$$\log p(x) = \log \int p(x, z) \, dz$$

Since this integral is intractable, we approximate using a variational distribution $q(z|x)$ and apply Jensen's inequality:

$$\log p(x) = \log \int q(z|x) \frac{p(x, z)}{q(z|x)} \, dz$$

$$\log p(x) \geq \mathbb{E}_{q(z|x)} \left[ \log \frac{p(x, z)}{q(z|x)} \right]$$

This inequality gives us the Evidence Lower Bound (ELBO):

$$\text{ELBO} = \mathbb{E}_{q(z|x)} [\log p(x|z)] - \text{KL}(q(z|x) \parallel p(z))$$

### Components Explained:

1. **Reconstruction Term:**  
   $\mathbb{E}_{q(z|x)} [\log p(x|z)]$ represents the expected log-likelihood of the data given the latent variable, encouraging accurate reconstruction.

2. **Regularization Term:**  
   $-\text{KL}(q(z|x) \parallel p(z))$ is the Kullback–Leibler divergence that regularizes the latent space by minimizing the difference between the approximate posterior $q(z|x)$ and the prior $p(z)$.

## Model Architecture

The VAE consists of two main components:

- **Encoder**: Transforms input data into a probabilistic latent representation
- **Decoder**: Reconstructs the original data from samples in the latent space
- **Latent Space**: Gaussian distribution with learned mean and variance parameters

## Training Methodology

The training process minimizes the Evidence Lower Bound (ELBO) loss:

$$\text{Loss} = \text{Reconstruction Loss} + \text{LPIPS Loss} + \beta \times \text{KL Divergence}$$

### Implementation Details:

- **LPIPS loss**: Used for perceptual similarity, capturing human-perceived differences more effectively
- **L1 loss**: Implemented for reconstruction as it provides stronger signal response compared to MSE loss
- **Beta factor**: Set to 0.02 to scale KL divergence, ensuring balanced regularization without excessive latent compression

## Results

Below are sample results showing original images alongside their reconstructions:

![Input and Reconstructed Image](images/original_vs_generated32.png)

![Input and Reconstructed Image](images/original_vs_generated100.png)

