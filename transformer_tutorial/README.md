This repository contains my experimentation with transformer architecture and many major concepts surrounding it based on Andre Karpathy's Youtube series back in 2022/2023.

Since this was my first time actually covering transformers from scratch, I added a huge amount of comments, expanding on nearly each and every subject from adding small ordinary reminders, to more complex concepts around attention, transformer architecture, positional embeddings and more, things that werent covered in the Karpathy's video originally.

I have removed/rewritten many comments over and over, each time experimenting with what I would find by working with LLMs at the time and the sources I could find. I tried to add the websites, youtube videos, etc that I used, its more than 60 sources links to different articles, papers, youtube videos. they are referenced inside each file.
I used Microsoft Copilot and ChatGPT3.5 if I remember correctly. It lead to lots of contradictory points which made things more complicated than it needed to, but it also was very helpful at times.

Anyway, I tried and remove many wrong interpertations, lacking explanations as much as I could during my experimentations, I left the ones that I deemed interesting or require more thoughts for later. I need to revise this one last time and make it useable for the rest of us that might find my explanations and experimentations useful. 

## TL;DR

So far, [transformer_gpt.py](transformer_gpt.py) covers:

* **Data and tokenization basics** — Building a simple character-level vocabulary, implementing encode/decode helpers, and comparing it with GPT-2 tokenization using `tiktoken.get_encoding('gpt2')`.

* **Dataset preparation and batching** — Converting the corpus into tokens, creating train/validation splits, working with context windows, and implementing mini-batch sampling with `get_batch`.

* **A simple baseline language model** — Building a `BigramModel` that maps tokens to logits, computes loss, and demonstrates autoregressive text generation.

* **Embeddings** — Exploring token embeddings and positional embeddings as the input representations used by the Transformer.

* **Linear algebra fundamentals** — Working through broadcasting, matrix multiplication, batched matrix multiplication, lower-triangular (`tril`) masks, masking with `-inf`, softmax, tensor reshaping, and other essential operations.

* **Self-attention fundamentals** — Building intuition for query, key, and value projections, attention scores, attention weights, causal masking, scaled dot-product attention, and how weighted values are aggregated.

* **Attention experiments and implementations** — Progressively moving from simple bag-of-words-style aggregation to masked self-attention, implementing `AttentionHead` modules, including fused implementations, and building `MultiHeadAttention`.

* **Feed-forward networks** — Implementing the feed-forward component used alongside attention in Transformer blocks.

* **Transformer blocks** — Combining multi-head attention and feed-forward networks in `AttentionwithFFNetBlock`, then stacking multiple blocks to build `BigramModelWithAttention`, progressively evolving the baseline into a GPT-style Transformer.

* **Normalization** — Implementing a custom `LayerNorm` and comparing its behavior with PyTorch's `nn.LayerNorm`.

* **Positional encodings** — Exploring learned and sinusoidal positional encodings through multiple implementations, including vectorized and more efficient approaches.

* **Positional encoding visualizations** — Analyzing positional representations using heatmaps, distance analysis, PCA, SVD, and t-SNE.

* **Training and evaluation** — Implementing `evaluate_loss`, training loops, train/validation evaluation, device handling, mixed precision with `GradScaler`, and optimization with `AdamW`.

* **Hyperparameter experiments** — Experimenting with parameters such as embedding size, head size, number of attention heads, number of Transformer blocks, batch size, learning rate, and other scaling choices.

* **Autoregressive generation** — Generating text token by token through `generate` methods attached to the language models.

* **Performance and scaling experiments** — Exploring practical considerations such as CPU vs. CUDA execution, learning-rate adjustments, mixed precision, and model scaling.

* **Additional Transformer concepts** — Notes and experiments covering broader Transformer-related ideas, architectures, and models, including concepts beyond the minimal GPT-style implementation.


> 🚧 **Work in progress:** This is an evolving learning project focused on understanding Transformers by implementing, experimenting with, and visualizing their core components from scratch.
