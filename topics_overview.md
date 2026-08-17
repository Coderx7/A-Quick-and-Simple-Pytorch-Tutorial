# PyTorch Tutorials — Topics Covered
This file lists every concept, technique, and model explained or implemented in each file of this repository.


## 1_Pytorch_basic_introduction.py
Introductory file focused on tensors, the autograd engine, and PyTorch fundamentals.

- **Introduction to PyTorch**: what PyTorch is, the relationship to the underlying `torch` C/C++/CUDA library, and the broader ecosystem (`torchvision`, `torchtext`, `torchrec`, etc.).
- **What is a Tensor**: multi-dimensional array concept, terminology vs NumPy/TensorFlow/JAX/MXNet, GPU acceleration.
- **PyTorch layers overview**: tensors, `nn` (layers/activations/norm), `optim` (SGD, Adadelta, Adam, RMSProp, AdamW), `data` (Dataset/DataLoader), autograd, domain libraries.
- **Autograd & Automatic Differentiation**: tape-based autograd, computation graphs, backpropagation intuition, `requires_grad`, leaf vs non-leaf tensors, `retain_grad()`, in-place operation restrictions, `backward()`, `detach()`.
- **Section 1 — What is a Tensor & How to Create One**: creation methods (uninitialized, `empty`, `rand`, `randn`, `randint`, from Python lists/NumPy), signed/unsigned dtypes.
- **Bridging PyTorch and NumPy**: shared memory vs. copying, converting tensors ↔ NumPy arrays.
- **Section 2 — Essential Tensor Attributes/Properties**: `shape`, `dtype`, `device`, `requires_grad`, number of elements, `is_contiguous()`, strided layout (`torch.strided`), memory layout & contiguity, why some ops require contiguous tensors, `clone`.
- **Section 4 — Device Management & Custom Defaults**: CPU/GPU acceleration, `torch.cuda`, CUDA vs ROCm, inspecting available accelerators, inspecting a tensor's device, moving tensors between devices, creating tensors directly on a device, working with multiple GPUs, optimized backends (XNNPACK, MKL, cuDNN).
- **Section 6 — Dimension Manipulation**: `view()`, `reshape()`, `resize()`, `squeeze()`, `unsqueeze()`, `permute()`.
- **Section 5 — Precise Data-Type Control & Casting**: float16/float32/float64, underflow/overflow, `torch.compile` mention, `torch.amp` (Automatic Mixed Precision / lower-precision training) mention.
- **Section 7 — Tensor Operations**: element-wise math, broadcasting rules, advanced matrix multiplication (`@`, `torch.matmul`), stacking/RNN/attention use-cases.
- **Section 9 — Seeding & Reproducibility (RNG Management)**: `torch.manual_seed`, RNG state, non-determinism in RNNs/LSTMs, deleting optimizers to free memory.
- **Appendices**: extended RNG discussion, logging discussion, performance-optimization notes.

---

## 2_Pytorch_basic_introduction_NeuralNetworks.py
Building and training neural networks, from datasets to fine-tuning pretrained models.

- **torchvision module**: built-in datasets (MNIST, etc.), `transforms` (ToTensor, Normalize, resize, flip), data augmentation, and pretrained model zoo (AlexNet, VGGNet, ResNet, MobileNet, ViT, ConvNeXT, EfficientNet).
- **Custom transforms**: implementing `ToTensor` and `Resize` from scratch.
- **DataLoaders & batching**: `DataLoader`, `num_workers`, batching efficiency.
- **Samplers**: `SubsetRandomSampler` for train/validation splits (no validation set in source dataset).
- **Data visualization**: displaying image batches with matplotlib, unnormalizing, per-pixel intensity annotation.
- **Building neural networks**: `nn.Module`, the `forward()` method, fully-connected (MLP) layers.
- **Weight initialization**: `torch.nn.init` (Xavier/Glorot, Kaiming/He), `model.apply()`, direct init.
- **Model construction patterns**: simple layer definition, `nn.Sequential`, `OrderedDict` named modules, `nn.ModuleList`, `nn.ModuleDict`, Python lists (and why they break `parameters()`/saving).
- **Custom modules / layers**: `Flatten`, `ResBlock`, a from-scratch `ResNet` implementation (`Resnet` class), `MyNet`.
- **Activations & pooling**: ReLU, LeakyReLU, MaxPool, etc.
- **BatchNormalization**: behavior, train/eval modes, why it matters, implementing it later.
- **Softmax, log-softmax & loss**: sigmoid vs softmax, mutually-exclusive vs multi-class, `CrossEntropyLoss` (folds log-softmax + NLL), `NLLLoss`, numerical stability, exercise to implement softmax.
- **Dropout**: usage in train/eval.
- **Training & evaluation loops**: `zero_grad`, `backward`, `step`, accuracy calculation, improving/maintaining the loop.
- **Optimizers**: SGD, Adam (in `torch.optim`), learning-rate and per-parameter options.
- **LR schedulers**: step/multi-step scheduling.
- **Saving / loading models & checkpoints**: `state_dict`, persisting model, optimizer, scheduler, and hyperparameters; resuming training.
- **Fine-tuning pretrained models**: replacing the final layer/head, freezing backbone, adding param groups.
- **Architectures walkthrough (torchvision models)**: ResNet (resnet18), VGG (with BatchNorm), SqueezeNet, Inception (dual output), ConvNeXT, EfficientNet; `get_model()` API, quantization mention.

---

## autoencoders.py
Comprehensive study of autoencoders, their regularized variants, generative extensions, and autoregressive models.

- **Autoencoder fundamentals**: encoder/decoder, the bottleneck, latent-space (z) representation, undercomplete vs overcomplete representations, reconstruction as self-supervised target, use-cases (denoising, missing-part filling, generation, visualization).
- **Bayesian background**: posterior probability, posterior distribution, prior probability, uninformative & conjugate priors (motivation for VAEs).
- **PCA connection**: vanilla linear autoencoder equivalence to PCA under certain conditions; PCA as a baseline.
- **Linear AutoEncoder**: simplest single-layer implementation.
- **MLP AutoEncoder**: multi-layer fully-connected autoencoder.
- **Convolutional AutoEncoders (ConvAE)**: `ConvAutoEncoder`, `ConvolutionalAutoEncoder_v2`, conv/upconv/deconv building blocks.
- **Shared-weights / tied autoencoders**: `SharedWeightsAE`, `SharedWeightsAEFunctional` (tied encoder–decoder weights).
- **Denoising AutoEncoders (DAE)**: adding uniform vs Gaussian noise, natural-noise simulation, references to DnCNN, CBDNet, RIDNet; noise scaling/thresholding.
- **Sparse AutoEncoders**: L1 penalty (`L1Penalty` autograd Function), KL-divergence sparsity, Andrew Ng notes, sparsity on activations, interpretability, overcomplete sparse representations.
- **Variational AutoEncoders (VAE)**: two implementations (`VAE`), reparameterization trick, KL divergence loss, ELBO, NLL, latent-space continuity/smoothness, posterior collapse, MSE vs BCE reconstruction loss, disentanglement, β control.
- **Conditional Variational AutoEncoder (C-VAE / CVAE)**: `VAE_Conditional`, conditioning the encoder/decoder on class labels, conditional generation.
- **β-VAE (`B_VAE`)**: disentangled VAE, KL weighting (`beta`), Free-Bits regularization (`min_kl`), latent-capacity tuning, decoder/encoder capacity trade-offs, prior distribution changes.
- **Vector Quantised VAE (VQ-VAE)**: `VQVAE` with `Quantizer`, discrete latent codes, solving posterior collapse, needs a prior model (e.g. PixelCNN) to generate.
- **Contractive AutoEncoder (CAE)**: `Contractive_AutoEncoder`, Jacobian-based contractive penalty, regularized autoencoder that resists identity mapping, encoder robustness.
- **Adversarial Autoencoder (AAE)**: concept/reference (adversarial latent regularization).
- **Using the encoder for classification**: stripping the decoder, adding a classifier head, benefits of pretrained sparse representations.
- **Latent-space visualization**: PCA and t-SNE (`TSNE`) 2D projections of encoder outputs (MNIST, Frey Face dataset).
- **PixelCNN / PixelCNN++ (autoregressive generative models)**: `PixelCNN_old`, `PixelCNN`, `PixelCNN2`, `PixelCNN2Gated`; `MaskedConv2d`, vertical/horizontal masked convolutions (`vertical_masked_conv`, `horizontal_masked_conv`, `GatedMaskedConv`), `GatedConv2d`, `GatedResidualBlock`, `ResidualBlock`, `GatedActivation`; used as a prior over VQ-VAE latent codes.
- **Datasets used**: MNIST, Frey Face; reconstruction, manifold, and interpolation experiments.

---

## gans.py
Generative Adversarial Networks: from vanilla GANs up to StyleGAN3 and CycleGAN, plus metrics and stabilization tricks.

- **GAN fundamentals**: generator vs discriminator, latent vector z, learning a real-data distribution, fooling the discriminator.
- **Vanilla GAN**: `Discriminator` / `Generator` (linear MLP), BCE loss, adversarial training loop, fixed latent vector for sampling.
- **GAN loss implementations**: real/fake loss helpers, label smoothing, noisy labels.
- **Training visualization**: `display_images()`, loss curves.
- **DCGAN**: deep convolutional GAN (ICLR 2016), `ConvBlock`/`ConvTransBlock`, `DiscriminatorCNN`/`GeneratorCNN`, weight init (`weights_init_dcgan`), tips (normalize to [-1,1], tanh, LeakyReLU, avoid sparse gradients/ReLU/maxpool), Adam betas [0.5, 0.999].
- **Debugging & stabilization**: Gaussian noise injection on real/fake images, mode collapse (diagnosis & mitigation), experience replay mention, great-circle interpolation.
- **SEFA**: semantics factorization of latent directions (eigenvectors), attribute removal/extraction experiments.
- **Conditional GAN (CelebA)**: `DiscriminatorCNNConditional`/`GeneratorCNNConditional`, feeding attribute labels to both networks.
- **CelebA classifier**: `CelebAClassifier`, loading pretrained classifier weights, attribute-based image retrieval (e.g. men with hair).
- **FID / IS metrics**: `IS_FID_Calculator`, CPU and CUDA FID tests, dataloader-based evaluation.
- **LSGAN**: least-squares loss to avoid vanishing gradients.
- **WGAN-GP**: Wasserstein (Earth-Mover) distance, gradient penalty, stability vs mode collapse.
- **Improved GAN (64×64)**: `ConvBlock2`, `DiscConvBlock`, `DiscriminatorImproved64`, `UpsampleBlock`, `GeneratorImproved64`, spectral/weight considerations.
- **ProGAN**: progressive growing, `EqualizedConv2d`/`EqualizedConvTrans`/`EqualizedLinear` (equalized learning rate), `PixelNorm`, `AddBatchStdDev` (mini-batch standard deviation), `DiscBlockProGAN`/`GenBlockProGAN`, `DiscriminatorProGAN`/`GeneratorProGAN`, custom training loop & debug logs.
- **StyleGAN1**: `MappingNetwork`, `AdaIN` (adaptive instance norm), `NoiseInjection`, `StyleConvBlock`, `DiscriminatorStyleGAN1`/`GeneratorStyleGAN1`; style mixing, truncation trick, latent interpolation, eigen-direction manipulation, style changing.
- **StyleGAN2**: `ModulatedConv2d`, `MappingNetwork2`, `AdaAugment` (adaptive augmentation), `DiscriminatorStyleGAN2`/`GeneratorStyleGAN2`; interpolation & style-mix tests.
- **StyleGAN3**: Fourier-feature inputs (`FourierInput*` variants), `ModulatedConv2d3`, `StyleConvBlock3`, equivariance/rotation/translation, `DiscriminatorStyleGAN3`/`GeneratorStyleGAN3`, training.
- **CycleGAN**: unpaired image-to-image translation, `CycleGenerator`, `ResBlock`, `ConvTransposeBlock`, `ImageBuffer` (history/experience replay for discriminator).
- **Cross-cutting GAN concepts**: generator/discriminator capacity balance, sparse-gradient avoidance, sampling from Gaussian vs Uniform, label smoothing strength, truncation, interpolation, mode collapse in all its forms.

---

## MultiTaskLearning.py
Multi-task / multi-label learning with custom datasets and shared-backbone models (two full examples).

- **Multi-task learning introduction**: motivation, shared backbone with task-specific heads.
- **Custom Dataset class**: `AnimeMTLDataset` inheriting `torch.utils.data.Dataset`, reading labels from CSV, implementing `__getitem__`/`__len__`, storing label names for readable outputs.
- **Multi-label classification**: BCE/BCEWithLogits (numerically stable), one-hot labels, mutually-exclusive vs multi-label categories (colors can have multiple values).
- **Train/validation/test splits**: `SubsetRandomSampler`, custom train/val split, train/val/test split.
- **Multi-task architecture**: `Resnet18_multiTaskNet` (pretrained ResNet18 backbone + multiple classification heads: color, gender, region, fighting type, alignment).
- **Per-parameter optimizer options**: `add_param_group` for different learning rates per head, freezing backbone, `MultiStepLR` scheduler, weight decay.
- **Evaluation metrics**: subset accuracy (all labels must match) and per-label accuracy (label-wise average).
- **Two datasets / examples**: FGO anime dataset (fgo_multiclass_labels.csv) and the tiny anime hair/outfit multiclass-multilabel dataset (gender, adulthood, hair length/color, outfit colors).
- **Fine-tuning a pretrained model**: reusing features, adding heads, managing trainable groups.

---

## recurrent neural networks.py
Sequence models: from vanilla RNNs on time-series to seq2seq attention, sentiment analysis, and word embeddings.

- **RNN introduction**: PyTorch RNN/GRU/LSTM modules, `num_layers` (stacked RNN), `bidirectional`, `dropout`, hidden-state shapes.
- **Time-series forecasting**: generating sinusoidal sequences with `linspace`/`sin`, sample/label shifting, data generators, MSE loss, truncated backpropagation through time (Truncated BPTT).
- **Vanilla RNN**: `RNN_Net` implementation.
- **GRU & LSTM**: swapping in GRU/LSTM for the same task.
- **Char-level language modeling**: downloading text (Project Gutenberg), `lstm_char`, one-hot encoding, tokenization discussion, truncated BPTT, sampling/generation.
- **Embeddings**: `nn.Embedding` vs one-hot, specifying embedding dimensionality, capturing word relationships.
- **Seq2seq with Attention (machine translation)**: `Encoder`, `BahdanauAttentionDecoder`, `LSTMBahdanau`; alignment scores, context vector, fixed corrected version.
- **Sentiment analysis**: `SentimentLSTM`, building/training a sentiment classifier.
- **Word embeddings (word2vec)**: `SkipGram`, `SkipGramWithNegativeSampling`, `SkipGramNegativeSamplingLoss`; negative sampling, subsampling frequent words (Mikolov criteria), vocabulary/ frequency ordering, validation similarity tests.
- **Padding & packing**: `pack_padded_sequence` / padding utilities for variable-length sequences.
- **CTC Loss**: `CTCloss` for sequence tasks without pre-aligned targets.
- **Half-precision training**: brief mention (future chapters).
- **Transformers**: noted as the modern default for NLP (covered in later chapters).
