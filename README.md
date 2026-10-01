# A-Quick-and-Simple-Pytorch-Tutorial
This repo will contain simple tutorials for getting familiar with Pytorch quickly for beginners. 
This actually acts as a personal note for myself as well, as I review my old recollection of different algorithms and concepts
in Pytorch. I tried to explain everything so in case later on I forget something, I can quickly recall it or see  the refs I 
find useful. 
I'll tidy things up when I get the time, the following section will be updated as I finish different parts. 
Some sections are already done (e.g. syle transfer, RNNs, GANs, but they need full explanations, so when that is done, I'll push the changes to this repo. 
Hope this comes handy to some of you dear fellow software engineers/deeplearning researchers. 
Have a wonderful day/night :)

Subjcts : 
- [x] Introduction to Pytorch basics (`1_Pytorch_basic_introduction.py`)
  - [x] What a tensor is, and how to create one (`torch.Tensor`, `torch.tensor`, `zeros/ones/rand/randn/empty/eye`)
  - [x] Shared memory vs. copying when bridging NumPy and PyTorch (`torch.from_numpy`, `torch.tensor`, `.numpy()`)
  - [x] Proving shared memory by comparing raw pointers (`numpy.ctypes.data` vs. `Tensor.data_ptr()`)
  - [x] Tensor attributes: `shape`, `dtype`, `device`, `requires_grad`, `numel/nelement`, `ndim/dim`, `layout`, `is_contiguous`, `grad_fn`, `grad`, `element_size`
  - [x] Leaf vs. non-leaf tensors, `retain_grad()`, and when `.grad` is `None`
  - [x] Device management: CUDA / ROCm / MPS / XPU detection, `.to()`, `.cuda()`, `.cpu()`, direct-on-device creation
  - [x] Multi-GPU: `device_count`, `get_device_name`, `get_device_capability`, `get_device_properties`
  - [x] Default-device management (`with torch.device(...)`, `torch.set_default_device` / `get_default_device`)
  - [x] Dimension manipulation: `view` vs. `reshape` vs. `resize`, `squeeze`, `unsqueeze`, `permute`, contiguity and strides
  - [x] Reinterpreting a dtype as bytes (`view(torch.uint8)`, `torch.frombuffer`, `np.frombuffer`) instead of converting it
  - [x] Precise dtype control, dtype inference, `.to(dtype)`, `.float()`, `.long()`, `new_ones/new_zeros/new_tensor` factories
  - [x] Indexing, slicing, fancy/advanced indexing, boolean masking, in-place masked assignment
  - [x] Element-wise math, in-place ops, reduction ops (`sum/mean/max/min`), `.item()` vs. raw tensor repr
  - [x] Broadcasting rules (scalar, row-vector, 4-D right-aligned) and the deprecated 1-D pointwise behavior (`torch.utils.backcompat.broadcast_warning`)
  - [x] Matrix multiplication: `@` vs. `*`, `torch.mm`, `torch.matmul`, `torch.bmm`, 2-D-only vs. broadcasting batch matmul, shape→output summary table
  - [x] Joining / splitting / repeating: `cat`, `stack`, `split`, `chunk`, `unbind`, `repeat`, `expand`, `repeat_interleave`, `expand_copy`
  - [x] Seeding and reproducibility: `manual_seed`, `seed`, `initial_seed`, `get_rng_state`, isolated `torch.Generator` streams
  - [x] RNG-state forensics: proving the RNG state is not just the seed (endianness, `to_bytes`, `np.frombuffer`, `set_printoptions(profile='full')`)
  - [x] Determinism (documented, not executed): `use_deterministic_algorithms`, `cudnn.benchmark`, `cudnn.deterministic`, `CUBLAS_WORKSPACE_CONFIG`, `fill_uninitialized_memory`
  - [x] GPU diagnostics & memory profiling: `temperature()`, `utilization()`, `mem_get_info`, `memory_reserved`, `memory_stats`, `memory_summary`, `list_gpu_processes`
  - [x] A worked CUDA VRAM-leak experiment (alias not freed, `empty_cache()`, `nvidia-smi` vs. the caching allocator)
  - [x] A MiB / MB / KiB / GiB units sidebar
  - [x] A minimal 2-layer forward pass by hand (`torch.mm` + manual `sigmoid` vs. the built-in `.sigmoid()`)
  - [x] Utility helpers: `print_header`, `show_tensor`, and a hand-written `@contextmanager def printoptions(...)` emulating `np.printoptions`
- [x] Neural Networks, Datasets and Training (`2_Pytorch_basic_introduction_NeuralNetworks.py`)
  - [x] Datasets: MNIST, CIFAR10, `ImageFolder` (documented), plus custom `Dataset` usage
  - [x] Transforms: `transforms.Compose`, `ToTensor`, `Normalize`, and hand-written transform functors (`class ToTensor`, `class Resize` with `PIL.Image.BILINEAR`)
  - [x] `DataLoader`, batching, `num_workers`, and the `num_workers=0` debugging advice
  - [x] Train/validation splitting with `SubsetRandomSampler` (and why sampler + `shuffle=True` conflict)
  - [x] Building a network 5 different ways: explicit attributes, `nn.Sequential`, `OrderedDict`, python list of layers (and why it hides parameters), `nn.ModuleList`/`nn.ModuleDict`, `add_module`
  - [x] Swappable feature-extractor / classifier split (`SequentialNet7`) as the basis of fine-tuning
  - [x] A `Conv+BatchNorm+ReLU` block factory (`convlayer`) and 9 sequential-model variants
  - [x] Writing a custom `Flatten` layer (pre-`nn.Flatten`), and building a `MyNet` from scratch
  - [x] Residual connections: `ResBlock` and a small `Resnet`
  - [x] Module-tree introspection: `modules()`, `children()`, `named_children()`, `named_parameters(recurse=)`, `named_buffers()`, and a recursive tree printer
  - [x] Weight/bias initialization: direct `.data` calls, `torch.nn.init.*`, `calculate_gain`, `kaiming_normal_`, `model.apply(...)`
  - [x] Training and validation loops, progressively refactored into `training()` / `validation()` / `training_loop()` / `train_validation_loop()`
  - [x] `torch.no_grad()` for evaluation, `.item()` to avoid graph accumulation, `copy.deepcopy` of a model
  - [x] Accuracy three ways: `torch.max` + `torch.eq`, `preds.topk(1)` (including the `[32,1]` vs `[32]` broadcasting footgun), and batch-mean accuracy
  - [x] Top-5 accuracy, per-class accuracy counters, and collecting/visualizing wrongly-predicted images
  - [x] Loss theory: why no softmax before `CrossEntropyLoss`, `log_softmax` numerics, `NLLLoss` pairing, sigmoid vs. softmax
  - [x] Fine-tuning / transfer learning: swapping the head of AlexNet, VGG16-BN, SqueezeNet, InceptionV3 (aux + main head), MobileNetV3, ConvNeXt, Swin-B
  - [x] Freezing and unfreezing layers (`isinstance` → `named_parameters` → `parameters()` progression)
  - [x] LR scheduling: a manual decay loop, then `MultiStepLR`, plus the PyTorch ≥1.1 `scheduler.step()` ordering change
  - [x] Saving/loading: `state_dict` only, and a full checkpoint dict (`state_dict`, `optimizer`, `epoch`, accuracies, losses)
  - [x] The pretrained-weights API: `*_Weights.DEFAULT`, `weights.transforms()`, `get_model`, `get_model_weights`, `get_weight`, `list_models`
  - [x] Quantized weights (`models.quantization.*`) and static quantization vs. QAT
  - [x] `torch.hub` loading, including a hub SimpleNet (`coderx7/simplenet_pytorch`) and 7 commented variants
  - [x] End-to-end inference: `run_inference()`, softmax → top-5 percentages, `meta["categories"]` class names, manual ImageNet preprocessing
  - [x] Visualization: `torchvision.utils.make_grid`, image grids, per-pixel brightness heat-maps
  - [x] `torch.compile()` (documented only)
- [x] Multi-Task Learning (`MultiTaskLearning.py`)
  - [x] Two full versions of the same tutorial in one file (FGO dataset v1, anime-characters dataset v2)
  - [x] `AnimeMTLDataset`: reading labels from a CSV, capturing the header row, slicing CSV columns into per-task label groups, a `Label_names()` human-readable reverse map
  - [x] Multi-label encoding: one-hot vectors vs. `argmax`-based single-label encoding, multi-hot list comprehensions
  - [x] Multi-task architecture: pretrained ResNet-18 backbone (classifier stripped) + `BatchNorm1d` bottleneck + 5 separate output heads
  - [x] Per-head `xavier_normal_` init and a 5-tuple `forward()` returning raw logits (no sigmoid on the multilabel head)
  - [x] Loss selection per task: `BCEWithLogitsLoss` for the multi-label head, `CrossEntropyLoss` for the single-label heads
  - [x] Unweighted sum of all task losses, and `torch.set_grad_enabled(is_training)` for train/val sharing
  - [x] Multi-label metrics: subset accuracy, per-label accuracy, Hamming loss, and why a flat mean is not per-label accuracy
  - [x] Per-head `optimizer.add_param_group(...)` learning rates (an alternative to freeze/unfreeze)
  - [x] `MultiStepLR`, 3-way train/val/test `SubsetRandomSampler` splits, best-checkpoint autosaving on a composite score
  - [x] Freeze/unfreeze helpers and unfreeze-and-retrain from a saved checkpoint
  - [x] `parse_predictions` / `show_predictions` for human-readable multi-task output
  - [x] A `conv1d` vs `conv2d` input-dimension sidenote (3-D vs 4-D inputs) and the expected shape error
- [x] Autoencoders and Latent-Variable Generative Models (`autoencoders.py`)
  - [x] `LinearAutoEncoder` (AE)
  - [x] `MLPAutoEncoder` (MLPAE)
  - [x] `ConvAutoEncoder` and `ConvolutionalAutoEncoder_v2` (ConvAE, widening encoder)
  - [x] Denoising Autoencoder (DAE): Gaussian vs. uniform `add_noise`, clamp, MSE against the clean image
  - [x] Sparse Autoencoder (SAE): overcomplete latent, 4 sparsity modes (L1 on weights, L1-gradient enforcement via a custom `L1Penalty` autograd Function, L1 on activations, KL-divergence sparsity), layer-wise sparsity metrics, weight-distribution plots
  - [x] Tied weights as a self-regularizer (`SharedWeightsAE`, `SharedWeightsAEFunctional`)
  - [x] Contractive Autoencoder (CAE): Frobenius-norm-of-Jacobian penalty, both via autograd and a closed form
  - [x] Variational Autoencoder (VAE) — two versions, the second being the main one: skip connections, EMA skip-connection buffer, extra latent noise injection
  - [x] The reparameterization trick, ELBO derivation, and diagonal-covariance simplification
  - [x] β weighting, KL annealing with a logistic/sigmoid schedule, and free-bits regularization (`min_kl`)
  - [x] Posterior-collapse diagnostics (`mu/std` tracking, latent-diversity check, interpolation between latents)
  - [x] Conditional VAE (Cond-VAE) via one-hot label concatenation
  - [x] β-VAE (Disentangled VAE) with latent-dimension walks and manifold grids
  - [x] VQ-VAE: `Quantizer` codebook, nearest-neighbour lookup, straight-through estimator, codebook + commitment loss, EMA codebook update with Laplace smoothing, codebook-collapse/perplexity discussion
  - [x] VQ-VAE training: AdamW, `SequentialLR` (linear warmup → cosine), AMP fp16 + `GradScaler` (with fp16 NaN debugging), best-loss checkpointing
  - [x] PixelCNN prior v1 (masked convs A/B) and the improved label-conditional version with residual blocks
  - [x] PixelCNN++ prior: vertical/horizontal masked convs and gated residual blocks
  - [x] Prior training: AdamW, linear-warmup + cosine schedule, bits-per-dimension derivation, periodic in-training samples, GIF creation
  - [x] Latent-code inspection: `get_discrete_latent_codes`, latent-map visualization, prior-vs-real comparison, masked-conv receptive-field plots
  - [x] Autoregressive sampling with temperature, top-k and top-p (nucleus) truncation
  - [x] t-SNE / PCA projections of the encoder output and of the codebook (UMAP), latent-space grids, interpolation GIFs
  - [x] AE-pretraining-then-classifier experiment, batch-norm convergence notes, and ConvTranspose checkerboard artifacts
  - [x] Datasets: MNIST, CIFAR10, TinyImageNet-200, CelebA (40 attributes), anime characters
  - [ ] *Sequence-to-Sequence Autoencoder* and *Cyclical Annealing Schedule* are described in the notes but not implemented (the CA schedule is summarised at the end of the file)
- [x] Generative Adversarial Networks (`gans.py`)
  - [x] Vanilla fully-connected GAN (MNIST): `Discriminator` / `Generator`, label smoothing, separate D/G learning rates, fixed `z` monitoring, and a long mode-collapse / latent-entanglement debugging log
  - [x] Gaussian noise injection into D and G as a regularizer (`noise_addition`)
  - [x] DCGAN on SVHN: `ConvBlock` / `ConvTransBlock` with residual upsample branches, `weights_init_dcgan`, no BatchNorm in the first D layer or the last G layer, strided convs instead of pooling (avoiding sparse gradients)
  - [x] Conditional GAN (cGAN) on CelebA: label embeddings broadcast as extra input channels, plus a trained `CelebAClassifier` (ResNet-18 with a 40-logit head)
  - [x] LSGAN: least-squares loss on raw logits
  - [x] WGAN: Earth-Mover's-distance critic loss, weight clipping, and why BatchNorm must be off (it breaks 1-Lipschitzness)
  - [x] WGAN-GP: the full gradient-penalty implementation (random ε interpolation, `create_graph`/`retain_graph`, `(‖∇D‖₂−1)²`)
  - [x] SNGAN-style improvements: per-layer `spectral_norm` with a `no_spec_norm_list` escape hatch, `Dropout2d` in D, Upsample+Conv instead of ConvTranspose
  - [x] ProGAN: `EqualizedConv2d` (runtime weight scaling), `PixelNorm`, `AddBatchStdDev`, per-resolution blocks, α fade-in, LR halving, EMA generator with kimg-based warmup, full resume-from-checkpoint
  - [x] StyleGAN 1: `Blur`, `EqualizedLinear`/`EqualizedConv2d` (lr multipliers), `MappingNetwork`, `AdaIN`, learned `NoiseInjection`, `StyleConvBlock` (with an AdaIN-order ablation), learned `const_input`, 2 blocks per resolution
  - [x] StyleGAN 1 losses: non-saturating softplus loss, R1 penalty computed in fp32 under `autocast(enabled=False)`, style-mixing regularization, truncation (`psi`) with EMA `w`
  - [x] StyleGAN 2: pure-PyTorch `upfirdn2d` (plus a `make_kernel`/FIR version), residual discriminator blocks, `ModulatedConv2d` with weight demodulation, image/skip accumulation across resolutions, `AdaAugment` (differentiable flip/rotate/scale/translate/cutout/brightness/contrast) and `update_ada_p`
  - [x] StyleGAN 2: path-length regularization with an EMA `mean_path_length`, image-count-based EMA decay, `r1_penalty_interval`
  - [x] StyleGAN 3: Kaiser/sinc filter design and bandwidth-limited resampling, `FourierInput` variants, the LeakyReLU sandwich block, no noise injection (logged as an unsuccessful experiment)
  - [x] CycleGAN (summer↔winter, unpaired): `PatchGAN` discriminator, `InstanceNorm2d`, 6-residual-block `CycleGenerator`, LSGAN losses, cycle-consistency loss, identity loss, `ImageBuffer` history mixing
  - [x] A full custom FID / Inception-Score calculator (`IS_FID_Calculator`: 2048-d Inception-v3 features, mean/covariance, `scipy.linalg.sqrtm`, disk caching, 10-split IS)
  - [x] Training-health diagnostics: `get_status`, `get_overall_status`, loss history, D real/fake score plots, and mode-collapse heuristics
  - [x] Latent-space tooling: `lerp` and `slerp` in z and w, truncation sweeps, per-layer style-mixing tables, SVD/SEFA eigen-directions, CLIP text directions, an ipywidgets GUI, and an OpenCV μ/σ finder
  - [x] Interpolation GIF generation, per-epoch image grids, and configurable checkpointing with full hyper-parameter dumps
  - [x] Datasets: MNIST, SVHN, CIFAR10, CelebA, CelebA-HQ, FFHQ, Yosemite summer/winter
  - [ ] *BigGAN, StarGAN, InfoGAN, SNGAN (as a named architecture), pix2pix and StackGAN* are only referenced in the notes, not implemented (the conditional GAN here is a CelebA cGAN, not BigGAN)
- [x] Early / Standalone GAN Implementations (`misc/old_gan_files/`)
  - [x] `mnist_gan.py` — the original Goodfellow-style MNIST GAN baseline (superseded by `gans.py`)
  - [x] `DCGAN.py` — DCGAN on SVHN with the full Salimans "how to train a GAN" tips block (superseded by `gans.py`)
  - [x] `gan_test.py` — an MLP GAN *and* a DCGAN, both on MNIST (superseded by `gans.py`)
  - [x] `CycleGAN.py` — the earlier CycleGAN (superseded: `gans.py` adds the `ImageBuffer` and the identity loss)
  - [x] `face_GANs.py` — DCGAN on 64×64 faces plus latent "morphing", and a long essay on mode collapse, Inception Score/FID, StackGAN and IllustrationGAN
  - [x] `semi_supervised_training.py` — the **unique** semi-supervised GAN: label masking, a multi-task discriminator with `num_classes+1` outputs, GAN logits derived via the log-sum-exp trick, mean feature-matching generator loss, and masked accuracy
  - [x] `image_embedding.py` — the **unique** inverse `img2vec` encoder trained jointly with the generator so a generated image re-embeds back to its latent, plus latent/embedding-space arithmetic and t-SNE
  - [x] `test.py` — a leftover scratch file with no content
- [x] Recurrent Neural Networks (`recurrent neural networks.py` + `rnns_introduction.py`)
  - [x] Time-series prediction/regression on a synthetic `linspace().sin()` corpus with `nn.RNN` — both files
  - [x] A switchable `nn.RNN` / `nn.LSTM` / `nn.GRU` text generator (`rnn_type=` argument) — both files
  - [x] Multi-layer (stacked) and bidirectional variants — both files
  - [x] Character-level vocabulary building, corpus preprocessing, and one-hot encoding
  - [x] Truncated BPTT via hidden-state detaching
  - [x] Gutenberg *Anna Karenina* (`1399-0.txt`) as the character-level LM corpus
  - [x] **Only in `recurrent neural networks.py`:**
    - [x] Bahdanau (additive) attention seq2seq: `Encoder`, `BahdanauAttentionDecoder`, `LSTMBahdanau`, a `W_v` scoring vector, and attention-weight collection
    - [x] A long attention theory block (the fixed-vector bottleneck, scoring, softmax as weights, the step-by-step paper algorithm)
    - [x] Machine translation eng↔fra (`eng-fra.txt`), `<sos>/<eos>`, teacher forcing, greedy `topk(1)` decoding, and a refactored `Dataset` + `Vocab` namedtuple
    - [x] Attention-weight heat-map visualization over 10 curated sentence pairs (seaborn)
    - [x] Sentiment analysis with a word-level LSTM (embedding 500, hidden 300, `num_layers=3`), `BCELoss`, accuracy, and per-review inference
    - [x] Word embeddings: Skip-Gram with a full softmax, and Skip-Gram with Negative Sampling (`unigram^0.75` noise distribution, custom `bmm`/`sigmoid` loss)
    - [x] Mikolov subsampling with a threshold-vs-frequency table, analogy tests (`king−man+woman`), nearest neighbours, and PCA / t-SNE embedding plots
    - [x] Word-level tokenization, regex preprocessing, a `unidecode` vs `ascii` vs `unicodedata.normalize` comparison, and a `uint8` vs `int32` memory discussion
    - [x] Reference-only stub cells for Text-CNN, **CTC loss**, and NER
  - [x] **Only in `rnns_introduction.py`:**
    - [x] A dedicated decoding-strategies chapter: pure multinomial (ancestral) sampling, temperature sampling, temperature + top-k sampling (both a torch and a NumPy path), and top-k with prompt conditioning
    - [x] Beam search with per-beam hidden states, cumulative log-prob scoring, and parent hidden-state reassignment
    - [x] A long `detach()` / `.data` pedagogical section on why the hidden state is carried across batches but reset per epoch
    - [x] A `view()` vs. `reshape()` vs. `contiguous()` debugging walkthrough
    - [x] A NumPy one-hot encoding debugging arc (broadcasting failure → `.size` + `reshape(-1)`)
    - [x] Gradient clipping (`clip_grad_norm_`) and a punctuation/capitalization-removal essay with academic references
  - [x] `rnns_introduction.py` is the more complete home for *sampling methods*; `recurrent neural networks.py` is the only home for *attention, seq2seq, sentiment analysis and word embeddings*
  - [ ] *Word-level text generation* is not implemented (both generators are character-level; word-level only appears as an input representation)
  - [ ] *CTC loss* and *NER* are reference-only stubs, not implementations
- [x] Transformers from Scratch (`transformer_tutorial/transformer.py`)
  - [x] Character-level tokenization and `encode` / `decode` helpers, compared numerically against `tiktoken.get_encoding('gpt2')` (65 chars vs. 50,257 BPE tokens)
  - [x] A discussion of BPE / WordPiece / SentencePiece / unigram tokenizers (documented, not implemented)
  - [x] Dataset prep, a 90/10 train/val split (deliberately unshuffled, with a demo of what shuffling destroys), `block_size = context_size = 8`, and `get_batch`
  - [x] `BigramModel` baseline: token-embedding as a count table, `F.cross_entropy`, and autoregressive `generate`
  - [x] Linear-algebra foundations: broadcasting, `tril`, softmax-by-row-normalization, batched matmul, and proving that a `-inf`-masked softmax equals a masked average
  - [x] `AttentionHead`: separate key/query/value projections, `register_buffer` for the causal mask, `head_size**-0.5` scaling, and dynamic `[:T,:T]` masking for generation
  - [x] An empirical justification for scaling (unscaled vs. scaled attention variance, saturated vs. diffuse probability heatmaps)
  - [x] `AttentionHeadFused` (a single fused `kqv` projection) with equivalence proofs and a CUDA `%timeit` benchmark showing no consistent speedup
  - [x] `MultiHeadAttention` built from a `ModuleList` (no output projection, noted as a simplification)
  - [x] `FeedForward` with the paper's 4× inner expansion
  - [x] Transformer blocks built up in three stages: attention+FFN, then per-module residual connections, then pre-norm with a custom `LayerNorm`
  - [x] A custom `LayerNorm` verified against `nn.LayerNorm`, plus a documented bug post-mortem (missing `nn.Parameter` and BatchNorm-style last-two-dims aggregation)
  - [x] `BigramModelWithAttention` redefined six times as the model grows: learned position embeddings → multi-head → FFN → stacked blocks → residuals → LayerNorm
  - [x] `evaluate_loss` (`@torch.no_grad()`, train/val, `model.eval()`/`train()` toggling)
  - [x] Training: `AdamW`, mixed precision with `GradScaler` + `autocast` (scale/step/update), device handling, checkpoint saving, and CUDA memory instrumentation
  - [x] Hyper-parameter experiments with recorded results: 5 single-head sweeps, `head_size=8`, 4 heads, block/embedding scaling trade-offs, and two large CUDA runs (5.5M and 9.9M params)
  - [x] Learned vs. sinusoidal positional encodings: ~8 progressive NumPy implementations (never plugged into a model)
  - [x] Positional-encoding analysis: value curves, `pcolormesh` heatmaps, distance analysis (2-D and 3-D surfaces), pairwise-Euclidean and Gram-matrix heatmaps, and t-SNE / PCA / TruncatedSVD projections
  - [x] A long theory section on positional encodings (bounded magnitudes, why not integer counting, why sin/cos, uniqueness, phase orthogonality, add-vs-concat)
  - [x] Documented-only topics: cross-attention, RoPE, ALiBi, Flash Attention / SDPA backends, FlexAttention, weight tying, KV cache
  - [x] Corpora: `data/tiny_shakespear.txt` and `data/names.txt`; the paper is vendored in `transformer_tutorial/docs/`
- [x] Neural Network Internals Refresher (`transformer_tutorial/refresher_neuralnetworks_innerworkings/`)
  - [x] `micro_grad.py` — a from-scratch autograd engine: the `Value` class with `+ - * / ** exp tanh relu`, a DFS topological `backwardit()`, `graphviz` computation-graph rendering, and a finite-difference `gradient_check`
  - [x] `micro_grad.py` — a tiny NN library (`Neuron`, `Layer`, `MLP`) plus manual-SGD training on a toy dataset (no optimizer classes)
  - [x] `part1.py` — character-bigram counting, probability normalization, `torch.multinomial` sampling, likelihood / log-likelihood / NLL loss, epsilon and count smoothing, and a 1-layer neural bigram trained to the same loss
  - [x] `part2.py` — Bengio et al. 2003 neural probabilistic LM: learned character embeddings, a 2-layer MLP, softmax numerical-stability demos, and learning-rate sweeps plus decay
  - [x] `part3_BN.py` — initialization, saturation analysis, fan-in normalization / Kaiming init, a hand-written `BatchNorm1d` with running statistics, custom `Linear`/`Tanh`/`Embedding`/`Flatten` modules, and full activation/gradient/update-ratio diagnostics with and without BatchNorm
  - [x] `part4_backprop.py` — a full manual backward pass verified against PyTorch autograd with a `compare()` helper, including 7 different ways of computing the embedding gradient
  - [x] `part5.py` — a WaveNet-style hierarchical character LM: a mini `Module` framework, `FlattenConsecutive`, progressively doubled receptive fields, a `Sequential` with a `training` property, and eval-mode generation from running statistics
  - [x] `plot_logs.py` — a leftover ImageNet-style training-log plotter (unrelated to this tutorial's losses)
- [x] Diffusion Models (`nlp_diffusion_etc_tutorials/diffusion_models.py` + `notes/nlp.py` + `Stable Diffusion Deep Dive.ipynb`)
  - [x] **From `diffusion_models.py` — using pretrained Stable Diffusion pipelines**
    - [x] `StableDiffusionPipeline` / `DiffusionPipeline` / `AutoPipelineForText2Image`, fp16 variants, and pipeline/scheduler introspection
    - [x] The full scheduler catalogue (DDPM, DDIM, PNDM, LMS, Euler, Euler-a, Heun, UniPC, DEIS, DPMSolver single/multi-step and SDE, KDPM2) and swapping a scheduler with `DDIMScheduler.from_config`
    - [x] Inference parameters: `prompt`, `negative_prompt`, `guidance_scale`, `num_inference_steps`, `eta`, `generator`, `num_images_per_prompt`, `timesteps`
    - [x] IP-Adapter (`load_ip_adapter`, `set_ip_adapter_scale`, multi-reference prompting) with `OpenCLIP-ViT-H-14`
    - [x] Attention slicing / SDPA / xFormers caveats, the safety checker, and deterministic-generation settings
    - [x] Prose-only coverage: SD v1→v2→SDXL history, ControlNets, inpainting/outpainting, distilled models (SSD-1B, Vega, LCM-LoRA), checkpoint vs. LoRA formats, and utility models
  - [x] **From `diffusion_models.py` — training diffusion models from scratch**
    - [x] A first MNIST U-Net (`Unet`, `ConvBnAct`/`DeconvBnAct`, `create_betas`, `forward_diffusion`, L1 loss, ancestral DDPM sampling) and its 3000-epoch training loop
    - [x] A larger `UnetModel` + `ResBlock` + `SkipBlock` + `DiffusionMnist` with class and timestep conditioning, add-skip decoding, and per-block embedding control
    - [x] `SinusoidalPositionalEncoding` and `FourierFeatures` timestep embeddings
    - [x] `DiffusionNew` — an alternative hour-glass (non-U-Net) architecture experiment
    - [x] Beta schedules: linear, **cosine (Nichol & Dhariwal)**, and the **log-SNR / EDM (Karras) Sobol** schedule
    - [x] Loss variants: ε-prediction MSE vs. the log-SNR `v`-prediction EDM loss
    - [x] DDPM and DDIM (`eta`) samplers, EMA of the weights (`ema_update`), AMP fp16 + `GradScaler`, `MultiStepLR`, and a custom `GradualWarmupScheduler`
    - [x] A very long experiment log on `beta_end`, timestep count, batch size, learning rate, dropout, weight decay and overfitting
    - [x] Argument dumps to `imgs_generated_dirs/imgs_gen_<ts>/args_train.yaml` plus per-interval sample grids
    - [x] A DDPM Colab port (class-conditional with `nn.Embedding`, DDIM sampling, noise-schedule plots) and a commented ResNet-style variant
  - [x] **From `notes/nlp.py` — a reusable log-domain diffusion module**
    - [x] `GaussianDiffusion` with the full log-space maths (`alphas_cum`, `tilde_betas`, `q_posterior`, `p_mean_variance`, `_predict_x0_from_eps`)
    - [x] Classifier-free guidance inside `p_mean_variance`
    - [x] DDIM (`ddim_p_mean_variance`, `ddim_sample`) with linear and quadratic timestep discretizations, and `trainloss`
    - [x] Rank-aware printing via `torch.distributed.get_rank()`
  - [x] **From `Stable Diffusion Deep Dive.ipynb` — the internals**
    - [x] Loading `AutoencoderKL`, `CLIPTokenizer`/`CLIPTextModel`, `UNet2DConditionModel` and `LMSDiscreteScheduler` individually
    - [x] A hand-written txt2img loop mirroring `StableDiffusionPipeline.__call__`, and the img2img "start from a noised input" variant
    - [x] Why latent diffusion instead of pixel diffusion (compression factor 48), and `pil_to_latent` / `latents_to_pil`
    - [x] The scheduler's `add_noise` / `scale_model_input`, and variance-preserving vs. variance-exploding schedules
    - [x] Text encoder internals: token embeddings, positional embeddings, their sum, and a re-implemented forward pass
    - [x] Embedding manipulation experiments: swapping a token embedding, averaging two prompt embeddings, and loading a **Textual Inversion** `learned_embeds.bin`
    - [x] The U-Net call signature, `latents_x0 = latents - sigma*noise_pred`, and step-by-step progress videos via ffmpeg
    - [x] Classifier-free guidance formula, and a differentiable **style/perceptual guidance** loss (`blue_loss`) with a loss-scale sweep
  - [x] Datasets: MNIST, CIFAR10 and StanfordCars (the CIFAR runs are the 134 saved sample grids)
  - [ ] *LoRA / DreamBooth / Textual Inversion fine-tuning of diffusion models* is **prose only** in `diffusion_models.py`; only the Textual-Inversion *inference* swap is actually executed (in the notebook)
  - [ ] *FID* is mentioned once in the notes but is only implemented in `gans.py`
- [x] Object Detection (`nlp_diffusion_etc_tutorials/detection.py`)
  - [x] OWL-ViT (`google/owlvit-base-patch32/16`, `-large-patch14`) and OWLv2 (`google/owlv2-*`) via `transformers`
  - [x] Open-vocabulary / zero-shot text-prompted detection, including OWLv2's extra objectness head
  - [x] Text-guided detection, one-shot image-guided detection (`model.image_guided_detection`), and ROI-based detection with `cv2.selectROI`
  - [x] Post-processing: `processor.post_process()`, `post_process_image_guided_detection` (score threshold + NMS), and normalized `cx,cy,w,h` box decoding
  - [x] `denormalize_owlv2` (inverting the CLIP mean/std) and OpenCV box drawing
  - [x] Bipartite matching loss and the Hungarian algorithm, CLIP backbone integration (documented as model architecture, not trained here)
  - [ ] *No training or fine-tuning* — inference and post-processing only
- [x] Image Captioning and Sequence-to-Sequence (`nlp_diffusion_etc_tutorials/image_captioning.py`)
  - [x] A from-scratch `Encoder` with swappable backbones: `resnet50(pretrained=True)` or the local SimpleNet
  - [x] A from-scratch `Decoder` LSTM comparing `HIDDEN_STATE` conditioning (image features as the initial hidden state) vs. `INPUT` conditioning (image features as the first token)
  - [x] An `EncoderDecoderImageCaption` built on `nn.Transformer` / `nn.TransformerDecoder` with a frozen encoder
  - [x] A `BertTokenizer`-based captioning variant
  - [x] A custom `Tokenizer` with `<pad>/<start>/<end>/<unk>`, caption-length statistics, and a `vocabs.pkl` dump
  - [x] `OurColateFN` (a functor, not a lambda, for DDP) with `pack_padded_sequence` / `pad_sequence` / `unpad_sequence` and dynamic per-batch padding
  - [x] MS-COCO 2017 dataset loaders (raw and `datasets.load_dataset` variants)
  - [x] Metrics: BLEU (nltk and `evaluate`), ROUGE-1/2/L/Lsum, brevity penalty, length ratio, average generation length, and a `compute_metrics` hook for the HF `Trainer`
  - [x] Caption generation: a custom top-k sampler and `model.generate(do_sample=True, top_k=3)`
  - [x] Training: AdamW, `StepLR`, `CrossEntropyLoss(ignore_index=pad_token_id)`, fp16 `GradScaler`/`autocast`, `torch.compile()`
  - [x] A long section on bidirectional-LSTM pitfalls (exposure bias, train/inference inconsistency, overfitting)
  - [x] HF: `VisionEncoderDecoderModel.from_pretrained('nlpconnect/vit-gpt2-image-captioning')` for inference
  - [x] HF: `from_encoder_decoder_pretrained('microsoft/swin-base-patch4-window7-224-in22k', 'gpt2')` and `Seq2SeqTrainer` / `Seq2SeqTrainingArguments` fine-tuning (fp16, `predict_with_generate`, `torch_compile`, `resume_from_checkpoint`)
  - [x] CLM vs. Seq2Seq vs. MLM language-modeling paradigms, with a survey of encoder, decoder and seq2seq checkpoint candidates
  - [x] The finished Swin + GPT-2 captioning checkpoint is linked from `results_imgcaptioning-swin-gpt2/link.txt`
- [x] Sentiment Analysis and Text Classification (`nlp_diffusion_etc_tutorials/sentiment_analysis.py`)
  - [x] A from-scratch word-level `SentimentLSTM` (embedding → LSTM → dropout → `Linear` → sigmoid) with `BCELoss`, `StepLR` and `clip_grad_norm_`
  - [x] Corpus preprocessing on `data/reviews.txt` + `data/labels.txt` (25,000 reviews, 12,500/12,500 split), word-frequency vocabulary analysis, and `pad_input`
  - [x] A `classify_text()` single-sentence inference helper
  - [x] HF `pipeline('sentiment-analysis')` with `distilbert-base-uncased-finetuned-sst-2-english`
  - [x] `DistilBertForSequenceClassification` fine-tuning with a custom `ReviewDataset` (80/10/10 split), `TrainingArguments` and `Trainer`
  - [x] A manual PyTorch fine-tuning loop as an alternative to the `Trainer`
  - [x] Building a model from config alone (`DistilBertConfig` → `DistilBertModel`) and adding a custom classifier head
  - [x] Encoder-vs-decoder discussion and a BERT / RoBERTa / XLNet / GPT-2 / DistilBERT comparison
  - [ ] *BERT, RoBERTa and XLNet are compared in the notes but not actually fine-tuned here*
- [x] LLM Alignment: Supervised Fine-Tuning and DPO (`nlp_diffusion_etc_tutorials/notes/*.ipynb`)
  - [x] **SFT notebook** — the `HuggingFaceH4/ultrachat_200k` dataset, `mistralai/Mistral-7B-v0.1`, and `apply_chat_template` for the chat format
  - [x] **SFT notebook** — full fine-tuning vs. LoRA (PEFT) vs. QLoRA, with 4-bit **NF4** `BitsAndBytesConfig` and `attn_implementation='flash_attention_2'`
  - [x] **SFT notebook** — `trl.SFTTrainer`, `peft.LoraConfig`, sequence packing, gradient checkpointing, and cosine scheduling
  - [x] **DPO notebook** — `HuggingFaceH4/ultrafeedback_binarized` (from `openbmb/UltraFeedback`), `alignment-handbook/zephyr-7b-sft-lora`, and `assistant_prefix` templating
  - [x] **DPO notebook** — `trl.DPOTrainer` with QLoRA, `beta=0.01`, paged AdamW 32-bit, and a DPO vs. KTO vs. IPO comparison
  - [x] **DPO notebook** — reloading the base model 4-bit + `PeftModel.from_pretrained`, verifying only the adapter parameters are trainable
  - [x] 4-bit inference with `AutoModelForCausalLM.from_pretrained(..., load_in_4bit=True, device_map='auto')`
  - [x] Background references in `nlp_diffusion_etc_tutorials/docs/` (a DPO presentation video link and the HuggingFace DPO slides)
- [x] Hugging Face, Tokenization and Metrics Reference Notes (`nlp_diffusion_etc_tutorials/huggingface_tutorials.py` + `notes/*.py`)
  - [x] A curated index of HF references: `transformers`, tokenizers (BPE, fast vs. slow), `accelerate`, data collators, and prompt engineering
  - [x] Metric references: BLEU, ROUGE, perplexity (`exp(loss)`), cross-entropy, and causal-LM evaluation
  - [x] Encoder-model references: BERT, RoBERTa, ALBERT, and encoders-vs-decoders
  - [x] Fine-tuning references: QLoRA, LoRA/PEFT for Llama-2-7B, and LangChain/LlamaIndex with local HF models
  - [x] `notes/efficient_models_quantization_pruning_etc.py` — a reference-only list: dynamic 4-bit quantization, FP4 LLM training, SYCL/Intel-GPU sync, `torch.ao` pruning + quantization, and distributed training
  - [x] `notes/feature_extraction.py`, `notes/text_summarization.py`, `notes/GANs.py` — reference-only stubs (links and pip installs)
  - [x] `notes/utils.py` — `AverageMeter`, `RecorderMeter` (with `plot_curve`), and time-formatting helpers
  - [ ] *Efficient models, quantization and pruning* are **listed but not implemented**; the only working quantization in the repo is QLoRA/NF4 in the two notebooks
- [x] Siamese Networks and Face Verification (`misc/Face verification Siamese Networks/siamese.py`)
  - [x] A triplet `Dataset` that samples anchor/positive/negative on top of `ImageFolder`
  - [x] A shared-weight `siamese_net` backbone (5 strided conv layers, `Dropout2d`, a custom `global_max_pooling` instead of `Flatten` so it is input-size agnostic)
  - [x] **Triplet loss** with `nn.PairwiseDistance` and a clamped margin (α = 0.2)
  - [x] Offline-style triplet sampling (pre-sampled inside the `Dataset`) and a discussion of online (in-batch) mining
  - [x] The bundled AT&T/one-shot faces dataset: 400 `.pgm` 92×112 grayscale images, 37 training identities and 3 held-out identities (`s5`, `s6`, `s7`)
  - [ ] *Online hard-negative mining, margin-contrastive / BCE losses, one-shot episode inference and a verification-accuracy or LFW evaluation are discussed in the comments but not implemented* (negatives are chosen uniformly at random)
- [x] Word Embeddings / Word2Vec (`misc/word2vec_negsampling_old.py` + the word-embedding chapter of `recurrent neural networks.py`)
  - [x] Skip-Gram with a full softmax and `NLLLoss` — both files
  - [x] Skip-Gram with Negative Sampling: dual input/output embedding tables, a `multinomial` noise sampler, and a custom `bmm`+`sigmoid` loss — both files
  - [x] Mikolov frequent-word subsampling with a threshold-vs-frequency discussion — the RNN version is the more detailed of the two
  - [x] Punctuation-to-named-token mapping, `Counter`-based vocabulary building, and frequency-sorted `word2int` / `int2word`
  - [x] Evaluation: cosine-similarity nearest neighbours, common-vs-rare index probing, and the `king−man+woman` analogy test (RNN version only)
  - [x] Embedding-space visualization with PCA and t-SNE, plus a t-SNE of the first 600 embedding rows in the old file
  - [x] Checkpointing that stores the embedding dim, window size, vocab and `state_dict`; an embedding-dim / epoch sweep (600, 300, 50, 10)
  - [x] The old file is **superseded** by the RNN chapter's version (the same `text8` corpus, plus analogy testing and a cleaner loss)
  - [ ] *CBOW, hierarchical softmax and the `f(w)^(3/4)` noise distribution are not implemented* (the old file's noise distribution is uniform, and CBOW appears only in a comment)
  - [x] The `text8` corpus itself is **not committed** (`data/` is git-ignored) and must be downloaded to run either file
- [x] SimpleNet and Vision Backbones (`nlp_diffusion_etc_tutorials/simplenet.py` + `notes/testsimplenet.py`)
  - [x] The SimpleNet architecture ("Let's keep it simple", Hasanpour et al. 2016) with a config-table-driven builder
  - [x] All registered variants: `simplenet_cifar_310k/460k/5m/5m_extra_pool` and the ImageNet `small / 5m / 9m` models (`m1`, `m2` and the `small` dropout variants)
  - [x] Conv+BatchNorm+ReLU blocks, 1×1 `k1` convolutions, Xavier-uniform init, `MaxPool2d` + `Dropout2d` pairs, and a stride-mode table
  - [x] Pretrained-weight auto-download with `default_cfgs` (input size, crop percentage, interpolation, mean/std)
  - [x] Used as a drop-in vision encoder inside `image_captioning.py` (and loadable from `torch.hub` inside `2_Pytorch_basic_introduction_NeuralNetworks.py`)
  - [x] `notes/testsimplenet.py` — commented-out checkpoint loading, `remove_dropout_layers`, and OpenCV SIFT/DAISY unit tests
  - [ ] *No training loop* — this is the reference model definition only
- [x] Datasets, Weights and Generated Artifacts
  - [x] Corpora committed here: `nlp_diffusion_etc_tutorials/data/{corpus.txt, reviews.txt, labels.txt}`, `transformer_tutorial/data/{tiny_shakespear.txt, names.txt}`, and the 400 `.pgm` face images under `misc/`
  - [x] Corpora downloaded at runtime: MNIST, CIFAR10, SVHN, CelebA, CelebA-HQ, FFHQ, LSUN, TinyImageNet-200, ImageNet, `eng-fra.txt`, `text8`, and the summer↔winter Yosemite set
  - [x] `nlp_diffusion_etc_tutorials/imgs_generated_dirs/` — 134 timestamped CIFAR diffusion sample runs, each with an `args_train.yaml`
  - [x] `nlp_diffusion_etc_tutorials/vocabs.pkl` — the COCO caption vocabulary written by `image_captioning.py`
  - [x] `nlp_diffusion_etc_tutorials/results_imgcaptioning-swin-gpt2/link.txt` and the MEGA download links in `weights/weights.txt` (weights and visualizations, ~31 GB)
  - [x] Figures produced by the tutorials: `pixelCNN_blindspot.png`, `contractive_autoencoder_u1_u2_axis_plot.png`, and the positional-encoding plots in `transformer_tutorial/imgs/`
  - [x] `.gitignore` excludes `data/`, `results/`, `weights/`, `FutureTodos/` and `notebook_exports/`
- [x] Topics Referenced but Not Implemented/Included here yet: 
  - [ ] Neural style transfer (the previous readme listed it as done — there is no implementation)
  - [ ] Adversarial attacks (FGSM/PGD etc. — the previous readme listed it as done — there is no implementation)
  - [ ] Semantic segmentation (the previous readme listed it as done — only mentioned in quoted paper abstracts and directory names)
  - [ ] CTC loss (only a reference cell with links) and NER (a single comment)
  - [ ] Word-level text generation, and n-gram / bigram language models inside the RNN files (the bigram baseline lives in `transformer_tutorial/` instead)
  - [ ] BigGAN, StarGAN, InfoGAN, SNGAN, pix2pix and StackGAN as implementations
  - [ ] LoRA / DreamBooth diffusion fine-tuning (prose only), and quantization / pruning / distributed training (`notes/` lists them only)
  - [ ] One-shot-learning episode inference and triplet hard-negative mining in the Siamese chapter
  - [ ] Flash Attention / SDPA / FlexAttention / RoPE / ALiBi / KV-cache in the transformer chapter (all discussion-only TODOs)


### Update 2026:  

This update brings in the tutorials I originally wrote for myself back in 2018/2019,   
I didnt plan on publishing them publicly initially, as they were only my  
practice files, reminders and definitely not in a shape fit for others hence why  
It took a fair amount of work to consolidate and clean them up into something usable.  

While organizing them, I also added a few more architectures, so the repo now covers  
a reasonable range of topics, all heavily commented and explained.  

Each file/section/chapter includes the references I used while writing it. aside  
from the original papers and their implementations, many of the exercises   
and projects here are simply me implementing concepts I was learning at the time  
some inspired by online courses such as Coursera (deeplearning course) or   
Udacity (Pytorch introduction) circa 2017~2019.   
Please refer to each file for further references.   

I used LLMs such as ChatGPT, Gemini, Claude, Microsoft Copilot, Grok, and DeepSeek as   
a secondary aid throughout my work post-2023. I mainly used these for debugging, proofreading,  
and sanity-checking my explanations, mostly concerning newer architectures that I had not   
previously implemented, such as new GAN architectures, and, to a lesser degree, my existing  
explanations especially when I had made an incorrect assumption or explanation that needed to be corrected.  

Some chapters need more edits/proofing than others, but you should be able to read/experiment  
nonetheless. hopefully I'll take care of them when my time allows me and do a cleanup later on.  


[TOPICS.md](./TOPICS.md) contains an auto generated summary of what these tutorials cover  
I generated it using claude. 
