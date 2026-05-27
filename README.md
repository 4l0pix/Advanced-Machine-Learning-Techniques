# Stoic Transformer Language Model: Technical Architecture Report

## Abstract
This repository implements a compact decoder-only Transformer language model trained on Stoic philosophical texts. The model is GPT-style in the sense that it performs autoregressive next-character prediction with masked self-attention, residual decoder blocks, a causal context window, and sampling-based text generation.

The implementation is intentionally small and readable: tokenization is character-level, the context length is 128 characters, the hidden width is 256, and the trained checkpoint contains 3,180,626 trainable parameters. This document describes the system in code-level detail, with emphasis on the model architecture implemented in [stoic_transformer_llm.py](stoic_transformer_llm.py) and mirrored for inference in [chat-with-the-stoics.py](chat-with-the-stoics.py).

## Repository Structure
- [stoic_transformer_llm.py](stoic_transformer_llm.py): End-to-end training, evaluation, generation, visualization, and artifact export script.
- [chat-with-the-stoics.py](chat-with-the-stics.py): Console inference client that reconstructs the model and tokenizer, loads a checkpoint, and generates philosopher-styled responses.
- [stoic_lm_config.json](stoic_lm_config.json): Saved vocabulary, architecture hyperparameters, training metadata, best validation loss, and parameter count.
- `stoic_lm_best.pt`: Best model checkpoint, expected by [chat-with-the-stoics.py](chat-with-the-stoics.py) by default.
- `extract_vocab_from_model.py`: Utility script for inspecting or reconstructing vocabulary-related model information.

## Data Pipeline
### Text Sources
The training corpus is assembled from three Project Gutenberg sources:
- Marcus Aurelius, *Meditations*
- Epictetus, *Enchiridion*
- Seneca, *Letters from a Stoic*

The script downloads each source with `urllib.request.urlopen`, strips Project Gutenberg header/footer markers, normalizes the text, and concatenates the cleaned documents with double newlines. If the remote downloads fail or produce less than 5,000 characters, the script falls back to an inline Stoic quote corpus repeated 120 times.

### Cleaning Procedure
The `clean_text` function applies a simple ASCII-focused normalization pipeline:
1. Converts Windows newlines `\r\n` to Unix newlines `\n`.
2. Replaces all non-printable or non-ASCII characters outside `0x20` to `0x7E`, except newline, with spaces.
3. Collapses repeated spaces and tabs into a single space.
4. Collapses runs of three or more newlines into two newlines.
5. Strips leading and trailing whitespace.

This keeps the tokenizer small and deterministic, at the cost of losing typographic punctuation, accents, and non-ASCII characters.

## Tokenization
The model uses character-level tokenization rather than word, subword, or byte-pair encoding.

The vocabulary is:

```text
V = sorted(set(FULL_CORPUS))
```

For the saved configuration in [stoic_lm_config.json](stoic_lm_config.json), the vocabulary size is:

```text
|V| = 82
```

Two lookup tables are built:
- `char2idx`: maps each character to an integer token id.
- `idx2char`: maps each token id back to a character.

Encoding drops characters that are not present in `char2idx`:

```python
def encode(text: str) -> list:
    return [char2idx[ch] for ch in text if ch in char2idx]
```

Decoding joins predicted character ids back into a string:

```python
def decode(indices: list) -> str:
    return ''.join(idx2char.get(i, '') for i in indices)
```

Because the model operates at the character level, its context window of 128 tokens means 128 characters, not 128 words or subword pieces. This makes the task harder than word-level generation because long-range semantic structure must be represented through many short character steps.

## Dataset Construction
After tokenization, the entire corpus is represented as a one-dimensional `torch.long` tensor:

```text
DATA.shape = [number_of_characters]
```

The sequence is split sequentially, not randomly:
- 90% training tokens
- 10% validation tokens

The `StoicDataset` class produces overlapping next-token training examples. For a block size `L = 128`, each sample is:

```text
x = [t_i,     t_{i+1}, ..., t_{i+L-1}]
y = [t_{i+1}, t_{i+2}, ..., t_{i+L}]
```

So every position in the input block predicts the next character. During training, the loss is computed over all `B * T` positions in the batch, where `B` is batch size and `T` is sequence length.

## Architecture Overview
The model class is `StoicLM`. It is a decoder-only Transformer with the following saved architecture:

| Component | Value |
|---|---:|
| Vocabulary size | 82 |
| Context window / block size | 128 characters |
| Model dimension, `d_model` | 256 |
| Decoder layers | 4 |
| Attention heads | 8 |
| Per-head dimension, `d_k` | 32 |
| Feed-forward hidden dimension, `d_ff` | 1,024 |
| Dropout | 0.10 during training, 0.0 in chat inference |
| Trainable parameters | 3,180,626 |

At a high level, the forward pass is:

```text
token ids
  -> token embedding
  -> sinusoidal positional encoding + dropout
  -> 4 decoder blocks
  -> final LayerNorm
  -> tied output projection
  -> vocabulary logits
```

For an input batch `x` with shape `(B, T)`, the logits have shape:

```text
logits.shape = (B, T, vocab_size)
```

With the saved configuration, this is:

```text
(B, T, 82)
```

## Embedding Layer
The token embedding table is:

```text
E in R^{82 x 256}
```

For input token ids:

```text
x in N^{B x T}
```

the embedding lookup produces:

```text
tok_emb in R^{B x T x 256}
```

Parameter count:

```text
82 * 256 = 20,992
```

The same weight matrix is reused by the output language-modeling head through weight tying:

```python
self.lm_head.weight = self.token_embedding.weight
```

This reduces the number of free parameters and forces input and output character representations to share the same learned geometry.

## Sinusoidal Positional Encoding
The model does not learn positional embeddings. It uses fixed sinusoidal encodings registered as a non-trainable buffer:

```python
self.register_buffer('pe', pe)
```

For position `pos` and channel pair `2i`, `2i + 1`:

```text
PE(pos, 2i)     = sin(pos / 10000^{2i / d_model})
PE(pos, 2i + 1) = cos(pos / 10000^{2i / d_model})
```

The buffer has shape:

```text
pe.shape = (1, max_seq_len, d_model)
         = (1, 128, 256)
```

During the forward pass:

```text
h = token_embedding(x) + pe[:, :T]
```

Dropout is then applied to the sum. Since positional encodings are fixed, they add no trainable parameters.

## Causal Self-Attention
Each decoder block contains one multi-head self-attention module. Because this is a language model, attention must be causal: position `t` may attend only to positions `0..t`.

### Projection Shapes
For hidden states:

```text
X in R^{B x T x 256}
```

the module applies learned projections:

```text
Q = X W_q
K = X W_k
V = X W_v
```

where each projection is implemented as `nn.Linear(256, 256)`.

The projected tensors are reshaped from:

```text
(B, T, 256)
```

to:

```text
(B, num_heads, T, d_k) = (B, 8, T, 32)
```

### Scaled Dot-Product Attention
For each head, attention scores are computed as:

```text
scores = Q K^T / sqrt(d_k)
```

The scaling factor is:

```text
sqrt(d_k) = sqrt(32)
```

This prevents dot products from growing too large as the head dimension increases, which helps keep the softmax distribution numerically stable.

### Causal Mask
The causal mask is created with:

```python
torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)
```

For sequence length `T`, its shape is:

```text
(1, 1, T, T)
```

The singleton batch and head dimensions allow broadcasting across all examples and all heads. Masked future positions are filled with negative infinity before softmax:

```python
attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
```

This makes the softmax probability of future positions zero.

### Attention Output
After softmax:

```text
attn_probs in R^{B x 8 x T x T}
```

The weighted value sum gives:

```text
attn_out_per_head in R^{B x 8 x T x 32}
```

Heads are transposed, concatenated, and projected back to model width:

```text
combined in R^{B x T x 256}
output   in R^{B x T x 256}
```

### Attention Parameter Count Per Block
Each attention layer has four linear transformations:
- Query projection `W_q`
- Key projection `W_k`
- Value projection `W_v`
- Output projection `W_o`

Each `Linear(256, 256)` has:

```text
256 * 256 + 256 = 65,792 parameters
```

So attention parameters per decoder block are:

```text
4 * 65,792 = 263,168
```

The implementation does not apply dropout directly to attention probabilities. Dropout is applied after the attention output projection as part of the residual branch.

## Position-Wise Feed-Forward Network
Each block contains a two-layer MLP applied independently at every sequence position:

```text
FFN(x) = W_2 ReLU(W_1 x + b_1) + b_2
```

with:

```text
fc1: Linear(256, 1024)
fc2: Linear(1024, 256)
```

The expansion ratio is:

```text
d_ff / d_model = 1024 / 256 = 4
```

Parameter count:

```text
fc1 = 256 * 1024 + 1024 = 263,168
fc2 = 1024 * 256 + 256 = 262,400
total FFN per block = 525,568
```

The nonlinearity is ReLU. There is no GELU, gated activation, SwiGLU, or convolutional component.

## Decoder Block
The `DecoderBlock` combines causal self-attention, feed-forward transformation, residual connections, dropout, and layer normalization.

The exact order in code is post-residual normalization:

```python
attn_out = self.self_attn(x, x, x, mask)
x = self.norm1(x + self.dropout(attn_out))

ff_out = self.ff(x)
x = self.norm2(x + self.dropout(ff_out))
```

Mathematically:

```text
x_1 = LayerNorm(x_0 + Dropout(SelfAttention(x_0)))
x_2 = LayerNorm(x_1 + Dropout(FFN(x_1)))
```

This is closer to the original Transformer post-norm layout than to modern GPT-style pre-norm blocks. The consequence is that normalization happens after each residual addition, which can make very deep networks harder to optimize than pre-norm designs, but this model is shallow enough at 4 layers that the choice is manageable.

### Decoder Block Parameter Count
Per block:

```text
attention = 263,168
FFN       = 525,568
norm1     = 256 gamma + 256 beta = 512
norm2     = 256 gamma + 256 beta = 512
total     = 789,760
```

For 4 decoder blocks:

```text
4 * 789,760 = 3,159,040
```

## Final Normalization and Language Modeling Head
After all decoder blocks, a final layer normalization is applied:

```python
h = self.norm(h)
```

This adds:

```text
256 gamma + 256 beta = 512 parameters
```

The language modeling head maps hidden states to vocabulary logits:

```text
lm_head: Linear(256, 82)
```

Its weight is tied to the token embedding matrix, so the weight itself does not add a separate `256 * 82` trainable matrix. However, the `nn.Linear` bias remains trainable:

```text
lm_head bias = 82 parameters
```

Final logits:

```text
logits = lm_head(LayerNorm(h))
logits in R^{B x T x 82}
```

## Full Parameter Breakdown
The saved model reports 3,180,626 trainable parameters. The breakdown is:

| Component | Parameters |
|---|---:|
| Token embedding / tied output weight | 20,992 |
| 4 decoder blocks | 3,159,040 |
| Final LayerNorm | 512 |
| Output head bias | 82 |
| **Total** | **3,180,626** |

Expanded per-block breakdown:

| Decoder Block Component | Parameters |
|---|---:|
| Multi-head attention | 263,168 |
| Feed-forward network | 525,568 |
| Two LayerNorm modules | 1,024 |
| **Total per block** | **789,760** |

## Weight Initialization
The training model initializes `nn.Linear` and `nn.Embedding` weights from a normal distribution:

```text
N(mean=0.0, std=0.02)
```

Linear biases are initialized to zero:

```python
nn.init.zeros_(module.bias)
```

LayerNorm parameters use PyTorch defaults: scale initialized to ones and bias initialized to zeros.

## Training Objective
The model is trained with teacher-forced next-character prediction. Given logits:

```text
logits in R^{B x T x V}
targets in N^{B x T}
```

the tensors are flattened:

```python
logits.view(-1, logits.size(-1))  # (B*T, V)
targets.view(-1)                  # (B*T)
```

and optimized with cross-entropy:

```text
L = - mean log p_theta(y_t | x_{\le t})
```

Because the mask prevents attention to future positions, the model cannot see the answer token at position `t + 1` when predicting it.

## Optimization Setup
The main training hyperparameters in [stoic_transformer_llm.py](stoic_transformer_llm.py) are:

| Hyperparameter | Value |
|---|---:|
| Batch size | 64 |
| Learning rate | 3e-4 |
| Epochs | 30 |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR |
| Scheduler `T_max` | 30 |

The training step is:
1. Move batch tensors to `DEVICE`.
2. Clear gradients with `optimizer.zero_grad(set_to_none=True)`.
3. Run the model and compute cross-entropy loss.
4. Backpropagate with `loss.backward()`.
5. Clip gradients with `clip_grad_norm_`.
6. Update parameters with `optimizer.step()`.

At the end of each epoch, the cosine scheduler is stepped once. Validation loss is estimated over at most 40 validation batches to keep evaluation bounded.

## Checkpointing and Exported Metadata
The best validation checkpoint is saved whenever validation loss improves:

```python
torch.save(model.state_dict(), CKPT_PATH)
```

The configuration export stores:
- Vocabulary size.
- `char2idx` and `idx2char`.
- `d_model`, `num_heads`, `num_layers`, `d_ff`, `block_size`, and `dropout`.
- Number of epochs.
- Best validation loss.
- Trainable parameter count.

The saved config currently records:

```text
best_val_loss = 1.4583159804344177
best_val_perplexity ~= exp(1.4583) = 4.30
```

## Generation Algorithm
Generation is autoregressive. The model repeatedly predicts one character, appends it to the context, and feeds the extended context back into the model.

For each generation step:
1. Keep only the most recent `block_size` tokens:

   ```python
   ctx = context[:, -BLOCK_SIZE:]
   ```

2. Compute logits for the current context.
3. Select the final time step:

   ```python
   logits = logits[:, -1, :]
   ```

4. Divide by temperature:

   ```text
   logits = logits / temperature
   ```

5. Apply top-k filtering.
6. Apply top-p nucleus filtering.
7. Convert to probabilities with softmax.
8. Sample the next id with `torch.multinomial`.
9. Append the sampled id to the running context.

### Temperature
Temperature controls distribution sharpness:
- Lower values make high-probability characters even more likely.
- Higher values flatten the distribution and increase variation.

The chat client uses philosopher-specific base temperatures and averages them with the user-controlled temperature.

### Top-k Filtering
If `top_k > 0`, only the `k` largest logits remain available for sampling. All other logits are set to negative infinity.

Default chat value:

```text
top_k = 40
```

### Top-p Filtering
Top-p, or nucleus sampling, sorts tokens by probability and suppresses tokens outside the nucleus of likely characters. The default chat value is:

```text
top_p = 0.92
```

This combines a hard candidate cap from top-k with a probability-mass-based cap from top-p.

## Chat Application
[chat.py](chat.py) defines the same architecture classes as the training script so the checkpoint keys match exactly. At runtime it:
1. Loads [stoic_lm_config.json](stoic_lm_config.json).
2. Rebuilds `char2idx` and `idx2char`.
3. Loads the checkpoint with `torch.load(..., weights_only=True)`.
4. Infers model dimensions from checkpoint tensors when possible.
5. Instantiates `StoicLM` with `dropout=0.0`.
6. Loads the state dict and enters an interactive console loop.

The chat interface supports:
- `help`
- `temp <value>`
- `tokens <N>`
- `topk <N>`
- `solo <marcus|epictetus|seneca>`
- `all`
- `quit` / `exit`

Persona prompting is implemented by prepending philosopher-specific introductions before the user message. The model itself is not instruction-tuned; personas are prompt-conditioning templates over the same character-level language model.

## Architectural Limitations
This model is useful as an educational implementation, but it has several important limitations:
- It uses character-level modeling, so semantic dependencies require many prediction steps.
- The context window is only 128 characters.
- The dataset is small and stylistically narrow.
- There is no learned tokenizer, no byte fallback, and unknown characters are silently dropped.
- There is no key-value cache during generation, so each new character recomputes the full context window.
- The decoder blocks use post-residual LayerNorm rather than pre-norm stabilization.
- Attention probability dropout is not used.
- The chat personas are prompt-based, not separately trained experts.

## Architecture Fidelity Notes
The inference implementation in [chat-with-the-stoics.py](chat-with-the-stoics.py) intentionally mirrors the training architecture:
- Same `MultiHeadAttention` tensor reshaping.
- Same causal mask construction.
- Same sinusoidal positional encoding.
- Same decoder block order.
- Same final LayerNorm and tied output projection.
- Same character-level vocabulary loaded from config.

Checkpoint compatibility depends on keeping these definitions aligned. Changing layer names, tensor shapes, block order, vocabulary order, or weight tying will require retraining or a careful checkpoint conversion.
