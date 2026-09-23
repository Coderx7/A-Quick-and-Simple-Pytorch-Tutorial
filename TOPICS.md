# PyTorch Tutorial Topics Overview

A comprehensive collection of PyTorch tutorials covering fundamental to advanced deep learning concepts.

---

## Core Tutorials

### 1. PyTorch Basics & Tensors
**File:** `1_Pytorch_basic_introduction.py`

An extensive introduction to PyTorch that builds intuition through hands-on tensor manipulation. Starts from scratch and progressively covers everything needed to work confidently with the framework.

**Key Concepts:**
- PyTorch ecosystem overview (torch, torch.nn, torch.optim, autograd)
- Tensor creation, properties, and manipulation
- Device management (CUDA, MPS, ROCm, CPU)
- Dimension operations (reshape, view, permute, squeeze)
- Indexing, slicing, and boolean masking
- Mathematical operations and broadcasting
- Autograd and gradient computation

---

### 2. Neural Networks Fundamentals
**File:** `2_Pytorch_basic_introduction_NeuralNetworks.py`

A hands-on guide to building neural networks from scratch, covering multiple architectural patterns and the complete training pipeline. Includes practical tips on weight initialization, checkpointing, and fine-tuning pretrained models.

**Key Concepts:**
- Dataset creation and data loading (DataLoader, transforms)
- Model architecture patterns (Sequential, ModuleList, custom blocks)
- Training loops with validation
- Weight initialization strategies
- Model saving/loading (checkpoints)
- Fine-tuning pretrained models
- Softmax vs LogSoftmax (numerical stability)

---

### 3. Recurrent Neural Networks
**File:** `recurrent neural networks.py`

A deep dive into sequence modeling with RNNs, building up from simple time-series prediction to character-level text generation. Culminates with attention mechanisms and encoder-decoder architectures for sequence-to-sequence tasks.

**Key Concepts:**
- Vanilla RNN, GRU, and LSTM architectures
- Hidden state management and truncated BPTT
- Character-level language modeling
- Text preprocessing and tokenization
- Attention mechanism (Bahdanau/Additive attention)
- Encoder-Decoder (Seq2Seq) architecture
- Bidirectional RNNs

---

### 4. Autoencoders
**File:** `autoencoders.py`

A comprehensive exploration of autoencoder variants, from simple linear models to sophisticated generative architectures. Each type builds upon previous concepts, showing how different constraints and objectives shape the learned representations.

**Architectures Covered:**

| Type | Description |
|------|-------------|
| Linear Autoencoder | Simplest form with single linear layer |
| MLP Autoencoder | Multi-layer fully connected with non-linearities |
| Convolutional Autoencoder | Uses Conv/ConvTranspose for images |
| Denoising Autoencoder | Learns to remove noise from corrupted inputs |
| Sparse Autoencoder | L1 regularization for sparse representations |
| Variational Autoencoder (VAE) | Probabilistic generative model |
| Conditional VAE (CVAE) | Class-conditional generation |
| β-VAE | Disentangled representations |
| Vector Quantized VAE (VQ-VAE) | Discrete latent space |
| Contractive Autoencoder | Penalizes sensitivity to input variations |

**Key Concepts:**
- Encoder-Decoder architecture
- Latent space representation
- Reconstruction loss (MSE, BCE)
- KL divergence for VAEs
- Reparameterization trick
- Undercomplete vs overcomplete representations
- t-SNE/PCA visualization of embeddings

---

### 5. Multi-Task Learning
**File:** `MultiTaskLearning.py`

Demonstrates how to train a single model to perform multiple tasks simultaneously, using anime character classification as a practical example with both single-label and multi-label objectives.

**Key Concepts:**
- Custom Dataset implementation for complex labels
- Multiple classification heads on shared backbone
- Multi-label vs multi-class classification
- Mixed loss functions (CrossEntropy + BCEWithLogits)
- Differential learning rates per task head
- Per-label and subset accuracy metrics

---

### 6. Generative Adversarial Networks
**File:** `gans.py`

A journey through GAN evolution, starting from the original vanilla formulation and progressing through architectural innovations that enabled high-resolution image synthesis. Includes extensive training tips and stabilization techniques.

**Architectures Covered:**

| Type | Key Features |
|------|--------------|
| Vanilla GAN | Original formulation with MLP layers |
| DCGAN | Convolutional architecture, batch norm |
| Conditional GAN | Class-conditional generation |
| WGAN | Wasserstein distance, weight clipping |
| WGAN-GP | Gradient penalty for stable training |
| ProGAN | Progressive growing, high-resolution synthesis |
| StyleGAN1 | Mapping network, adaptive instance norm |
| StyleGAN2 | Weight demodulation, path length regularization |
| StyleGAN3 | Alias-free generation, continuous signal |
| CycleGAN | Unpaired image-to-image translation |

**Key Concepts:**
- Generator vs Discriminator training
- Adversarial loss and training dynamics
- Mode collapse and stabilization techniques
- Label smoothing and noisy labels
- Spectral normalization
- Progressive growing for high-resolution images
- Style mixing and latent space interpolation
- FID and Inception Score metrics
- EMA (Exponential Moving Average) for stable generation

---

## Legacy Tutorials (misc/ folder)

> **Note:** These are older implementations kept for reference. Core concepts are covered in the main tutorials above.

| File | Topic | Notes |
|------|-------|-------|
| `word2vec_negsampling_old.py` | Word2Vec embeddings | Skip-gram with negative sampling |
| `siamese.py` | Face verification | Triplet loss, Siamese architecture |
| `DCGAN.py` | DCGAN on SVHN | Earlier DCGAN implementation |
| `CycleGAN.py` | Image translation | Earlier CycleGAN implementation |
| `mnist_gan.py` | Basic GAN | Simple MNIST generation |
| `face_GANs.py` | Face generation | Face-specific GAN training |
| `semi_supervised_training.py` | Semi-supervised learning | GAN-based approach |
| `image_embedding.py` | Image embeddings | Feature extraction |

---

## Quick Reference

### By Application

| Application | Relevant Topics |
|-------------|-----------------|
| **Image Generation** | GANs (DCGAN → StyleGAN3), VAEs |
| **Image Translation** | CycleGAN, Conditional GANs |
| **Anomaly Detection** | Autoencoders, Sparse Autoencoders |
| **Representation Learning** | VAE, VQ-VAE, β-VAE |
| **Denoising** | Denoising Autoencoders |
| **NLP** | RNNs, LSTMs, Attention, Word2Vec |
| **Face Tasks** | Siamese Networks, StyleGAN |

---

*Last updated: 2026-09-23*
