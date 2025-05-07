VAE: Variational Autoencoder

Project Overview

This project implements a Variational Autoencoder (VAE) from scratch for fun and learning, aiming to experiment with the training process and the architecture itself.

Features

Custom-built VAE architecture

Image reconstruction and generation

Experiment tracking with Weights & Biases (wandb)

Trained on $128\times128$ images with approximately 12 million parameters

Trained on about 15,000 images

ELBO Mathematics

The Evidence Lower Bound (ELBO) for a Variational Autoencoder can be derived as follows:

Given a data point $x$ and latent variable $z$, the marginal likelihood can be written as:

$\log p(x) = \log \int p(x|z) , p(z) , dz$

Since the integral is intractable, we approximate it using a variational distribution $q(z|x)$:

$\log p(x) = \mathbb{E}_{q(z|x)} \left[ \log \frac{p(x|z) , p(z)}{q(z|x)} \right] + \text{KL}(q(z|x) \parallel p(z|x))$

The ELBO can then be written as:

$\text{ELBO} = \mathbb{E}_{q(z|x)} [\log p(x|z)] - \text{KL}(q(z|x) \parallel p(z))$

The first term is the reconstruction loss, and the second term is the KL divergence that regularizes the latent space to match the prior distribution.

Model Architecture

The VAE model consists of an encoder and decoder network:

Encoder: Learns a probabilistic latent representation of the input data.

Decoder: Reconstructs the image from the latent representation.

Latent Space: Gaussian distribution with learned mean and variance (typically prior is fixed).

Training

The training process minimizes the Evidence Lower Bound (ELBO) loss, which is the sum of the reconstruction loss and KL divergence:

$	ext{loss} = 	ext{reconstruction loss} + 	ext{LPIPS loss} + eta 	imes 	ext{KL divergence}$

In this implementation, the following modifications were made:

Loss Functions:

LPIPS loss is used for perceptual similarity, capturing perceptual differences effectively.

L1 loss is used for reconstruction as it provides a stronger signal response compared to MSE loss.

KL Divergence Regularization:

A beta value of 0.02 is used to scale the KL divergence, ensuring balanced regularization without excessive latent compression.



