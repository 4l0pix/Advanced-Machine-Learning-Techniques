# Stoic Transformer Language Model: Architecture and Implementation Report

## Abstract
This document provides a detailed, code-accurate description of the decoder-only Transformer language model implemented in this repository. It covers data sourcing and preprocessing, character-level tokenization, dataset construction, model architecture, optimization and training procedure, sampling-based generation, and the interactive chat application. The description is aligned with the implementation in [stoic_transformer_llm.py](stoic_transformer_llm.py) and [chat.py](chat.py).

## Repository Structure
- [stoic_transformer_llm.py](stoic_transformer_llm.py): End-to-end training script and utilities.
- [chat.py](chat.py): Inference-time console chat client.
- [stoic_lm_best.pt](stoic_lm_best.pt): Best checkpoint produced by training.
- [stoic_lm_config.json](stoic_lm_config.json): Saved configuration and vocabulary.

## Data Pipeline
### Sources
The training corpus is constructed from three Project Gutenberg texts:
- Marcus Aurelius, Meditations
- Epictetus, Enchiridion
- Seneca, Letters from a Stoic

If downloads fail, a fallback inline Stoic corpus is used and repeated to increase volume. The online fetch and fallback logic are implemented in [stoic_transformer_llm.py](stoic_transformer_llm.py).

### Cleaning
Text is normalized using the following steps:
1. Convert Windows newlines to Unix newlines.
2. Remove non-printable characters outside ASCII range 0x20 to 0x7E, preserving newlines.
3. Collapse consecutive spaces and tabs to a single space.
4. Limit blank lines to a maximum of two consecutive newlines.
5. Strip leading and trailing whitespace.

### Corpus Assembly
Cleaned documents are concatenated with double newlines. A corpus statistics report (characters, words, lines) is printed for diagnostic purposes.

## Tokenization and Vocabulary
### Character-Level Vocabulary
A character-level tokenizer is used. The vocabulary is the sorted set of distinct characters in the corpus. Let $\mathcal{V}$ denote this set and $|\mathcal{V}|$ its size. Two mappings are constructed:
- `char2idx`: character to integer index
- `idx2char`: integer index to character

### Encoding and Decoding
Encoding maps each character $c$ in the input string to its index if present in `char2idx`. Decoding maps a sequence of indices to characters using `idx2char` and joins them into a string.

## Dataset Construction
The tokenized corpus is represented as a 1D tensor of token indices. The sequence is split into a training segment and a validation segment using a 90/10 ratio. A `StoicDataset` class provides input-target pairs for next-token prediction:
- Input $x = [t_i, t_{i+1}, ..., t_{i+L-1}]$
- Target $y = [t_{i+1}, t_{i+2}, ..., t_{i+L}]$
where $L$ is the block size.

## Model Architecture
The model is a decoder-only Transformer (GPT-style). All components are defined in [stoic_transformer_llm.py](stoic_transformer_llm.py) and mirrored in [chat.py](chat.py).

### Notation
- Batch size: $B$
- Sequence length: $T$
- Model dimension: $d_{model}$
- Number of attention heads: $h$
- Per-head dimension: $d_k = d_{model} / h$

### Token Embedding
Tokens are embedded using a learned embedding matrix $E \in \mathbb{R}^{|\mathcal{V}| \times d_{model}}$. The input token indices $x \in \mathbb{N}^{B \times T}$ are mapped to embeddings $X \in \mathbb{R}^{B \times T \times d_{model}}$.

### Positional Encoding
A fixed sinusoidal positional encoding is added to token embeddings. For position $p$ and dimension $i$:
$$
PE_{p,2i} = \sin\left(p / 10000^{2i/d_{model}}\right), \quad
PE_{p,2i+1} = \cos\left(p / 10000^{2i/d_{model}}\right)
$$
The encoding is precomputed up to `max_seq_len` and added to the token embeddings, followed by dropout.

### Multi-Head Self-Attention
For each head, queries, keys, and values are linear projections of the input:
$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
$$
where $W_Q, W_K, W_V \in \mathbb{R}^{d_{model} \times d_{model}}$.

The scaled dot-product attention is:
$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V
$$
where $M$ is the causal mask described below. Head outputs are concatenated and projected with $W_O$.

#### Causal Mask
A lower-triangular mask ensures autoregressive behavior. For each position $t$, attention is restricted to positions $\le t$. The mask is implemented as a tensor of shape $(1, 1, T, T)$ with zeros above the diagonal, applied by setting masked logits to $-\infty$.

### Position-Wise Feed-Forward Network
Each decoder block contains a two-layer feed-forward subnetwork applied independently at each position:
$$
\mathrm{FFN}(x) = W_2 \; \mathrm{ReLU}(W_1 x)
$$
with $W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$ and $W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$.

### Decoder Block
Each block uses pre-normalization with residual connections:
1. Self-attention with dropout and residual addition, followed by layer normalization.
2. Feed-forward network with dropout and residual addition, followed by layer normalization.

Formally, for input $x$:
$$
\hat{x} = \mathrm{LayerNorm}(x + \mathrm{Dropout}(\mathrm{SelfAttn}(x)))
$$
$$
\mathrm{Block}(x) = \mathrm{LayerNorm}(\hat{x} + \mathrm{Dropout}(\mathrm{FFN}(\hat{x})))
$$

### Output Projection and Weight Tying
The final hidden states are normalized and projected to vocabulary logits using a linear layer. The output projection weights are tied to the input embedding weights, reducing parameters and aligning input-output representations.

### Loss Function
For training, the model computes the cross-entropy loss between predicted logits and target tokens:
$$
\mathcal{L} = - \frac{1}{N} \sum_{i=1}^{N} \log p_{\theta}(y_i | x)
$$
The logits are reshaped to 2D and passed to `torch.nn.functional.cross_entropy`.

## Optimization and Training
### Hyperparameters
Key training hyperparameters are defined in [stoic_transformer_llm.py](stoic_transformer_llm.py), including:
- `BLOCK_SIZE`: context length
- `D_MODEL`: model dimension
- `NUM_HEADS`: number of attention heads
- `NUM_LAYERS`: number of decoder blocks
- `D_FF`: feed-forward dimension
- `DROPOUT`: dropout probability
- `BATCH_SIZE`: mini-batch size
- `LEARNING_RATE`: optimizer learning rate
- `NUM_EPOCHS`: number of training epochs
- `GRAD_CLIP`: gradient clipping threshold

### Optimizer
AdamW is used with weight decay. The training loop performs the standard sequence:
1. Zero gradients.
2. Forward pass and loss computation.
3. Backward pass.
4. Gradient clipping.
5. Optimizer step.

### Scheduler
A cosine annealing learning rate schedule is applied across epochs.

### Checkpointing
The best validation loss is tracked. When improved, the model state is saved to `stoic_lm_best.pt`.

## Text Generation
Generation uses autoregressive sampling with both top-k and top-p (nucleus) filtering. For each step:
1. The context is truncated to the most recent `BLOCK_SIZE` tokens.
2. Logits for the next token are computed and temperature-scaled.
3. Top-k filtering keeps only the $k$ most probable tokens.
4. Top-p filtering keeps the smallest token set with cumulative probability at least $p$.
5. A token is sampled from the resulting distribution.

This logic is implemented in `generate` in both [stoic_transformer_llm.py](stoic_transformer_llm.py) and [chat.py](chat.py).

## Chat Application
The console chat client in [chat.py](chat.py) loads a checkpoint and configuration, reconstructs the tokenizer from `stoic_lm_config.json`, and generates responses conditioned on philosopher-specific prompts. It supports:
- Multiple philosopher personas with different base temperature settings.
- Top-k, top-p, temperature, and max token configuration via commands.
- Optional single-philosopher or all-philosopher mode.

The chat loop is purely inference-time. Dropout is disabled by setting `dropout=0.0` in the instantiated model.

## Configuration and Reproducibility
`stoic_lm_config.json` stores:
- Vocabulary mappings (`char2idx`, `idx2char`).
- Architectural hyperparameters and training metadata.
This file is required for correct decoding at inference time and must match the checkpoint vocabulary.

## Notes on Architecture Fidelity
The chat model definition mirrors the training model definition to ensure checkpoint compatibility. The causal mask, positional encoding, layer normalization placement, and weight tying are consistent between training and inference.
