
import argparse
import json
import math
import os
import sys
import textwrap

import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================
# Model architecture  (must match the training code exactly)
# =====================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        return torch.matmul(torch.softmax(scores, dim=-1), V)

    def split_heads(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        B, _, T, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        return self.W_o(self.combine_heads(self.scaled_dot_product_attention(Q, K, V, mask)))


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff,    d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class StoicLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers,
                 d_ff, max_seq_len, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.lm_head.weight = self.token_embedding.weight
        self.d_model = d_model
        self.max_seq_len = max_seq_len

    def _causal_mask(self, size, device):
        return torch.tril(torch.ones(size, size, device=device)).unsqueeze(0).unsqueeze(0)

    def forward(self, x, targets=None):
        B, T = x.shape
        h = self.pos_encoding(self.token_embedding(x))
        mask = self._causal_mask(T, x.device)
        for block in self.blocks:
            h = block(h, mask)
        logits = self.lm_head(self.norm(h))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# =====================================
# Generation
# =====================================

@torch.no_grad()
def generate(model, encode_fn, decode_fn, prompt: str,
             max_new_tokens=300, temperature=0.75, top_k=40, top_p=0.92,
             block_size=128, device='cpu') -> str:
    model.eval()
    context   = torch.tensor(encode_fn(prompt), dtype=torch.long, device=device).unsqueeze(0)
    generated = list(encode_fn(prompt))

    for _ in range(max_new_tokens):
        ctx = context[:, -block_size:]
        logits, _ = model(ctx)
        logits = logits[:, -1, :] / max(temperature, 1e-8)

        #top-k
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        #top-p nucleus
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_logits[cum_probs - F.softmax(sorted_logits, dim=-1) > top_p] = float('-inf')
            logits = logits.scatter(1, sorted_idx, sorted_logits)

        probs= F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        context = torch.cat([context, next_id], dim=1)
        generated.append(next_id.item())

    return decode_fn(generated)


# =====================================
# Philosophers
# =====================================

PHILOSOPHERS = {
    'marcus': {
        'name':   'Marcus Aurelius',
        'title':  'Emperor of Rome',
        'intro':  'I, Marcus Aurelius, Emperor of Rome, write to you from the frontier:',
        'color':  '\033[94m',   # blue
    },
    'epictetus': {
        'name':  'Epictetus',
        'title': 'Freed Slave & Teacher',
        'intro': 'Epictetus speaks. Remember, slave or emperor, your will is your own:',
        'color': '\033[92m',   # green
    },
    'seneca': {
        'name':  'Seneca',
        'title': 'Statesman & Philosopher',
        'intro': 'Dear friend, Seneca writes on the matter of the soul:',
        'color': '\033[93m',   # yellow
    },
}

RESET  = '\033[0m'
BOLD   = '\033[1m'
DIM    = '\033[2m'
RED    = '\033[91m'
CYAN   = '\033[96m'
WHITE  = '\033[97m'

TEMPERATURES = {
    'marcus':    0.70,
    'epictetus': 0.75,
    'seneca':    0.78,
}

SEPARATOR = '─' * 64


# =====================================
# Display helpers
# =====================================

def clear_line():
    print('\r' + ' ' * 70 + '\r', end='', flush=True)

def print_banner():
    print(f'\n{BOLD}{CYAN}' + '╔' + '═' * 62 + '╗')
    print('║' + '  🏛️   STOIC PHILOSOPHER CONSOLE CHAT'.center(62) + '║')
    print('║' + ''.center(62) + '║')
    print('║' + '  Ask a question — all three philosophers will answer.'.center(62) + '║')
    print('║' + '  Type  help  for commands,  quit  to exit.'.center(62) + '║')
    print('╚' + '═' * 62 + '╝' + RESET + '\n')

def print_philosopher_header(key: str):
    p = PHILOSOPHERS[key]
    color = p['color']
    print(f'\n{color}{BOLD}┌─  {p["name"]}  ·  {p["title"]}{RESET}')
    print(f'{color}{DIM}│  {p["intro"]}{RESET}')
    print(f'{color}│{RESET}')

def print_response(text: str, color: str, width: int = 72):
    """Print wrapped response with a coloured left-bar."""
    # Strip the prompt prefix from the generated text (model repeats it)
    lines = textwrap.wrap(text, width=width)
    for line in lines:
        print(f'{color}│{RESET}  {line}')
    print(f'{color}└{"─" * 62}{RESET}')

def print_help():
    print(f'\n{BOLD}Commands:{RESET}')
    print(f'  {CYAN}help{RESET}          — show this message')
    print(f'  {CYAN}temp <0.5–1.2>{RESET} — set generation temperature  (default 0.75)')
    print(f'  {CYAN}tokens <N>{RESET}    — set max new tokens           (default 300)')
    print(f'  {CYAN}topk <N>{RESET}      — set top-k sampling           (default 40)')
    print(f'  {CYAN}solo <name>{RESET}   — chat with one philosopher    (marcus / epictetus / seneca)')
    print(f'  {CYAN}all{RESET}           — return to all-philosopher mode')
    print(f'  {CYAN}quit{RESET}  /  {CYAN}exit{RESET} — exit\n')


# =====================================
# Load checkpoint
# =====================================

DEFAULT_CONFIG = {
    'vocab_size':  97,
    'd_model':     256,
    'num_heads':   8,
    'num_layers':  4,
    'd_ff':        1024,
    'block_size':  128,
    'dropout':     0.1,
}

def load_model(checkpoint_path: str, config_path: str = None, device='cpu'):
    if not os.path.exists(checkpoint_path):
        print(f'{RED}ERROR: Checkpoint not found: {checkpoint_path}{RESET}')
        print('Run the Kaggle notebook first to generate stoic_lm_best.pt')
        sys.exit(1)

    # Load config
    cfg = dict(DEFAULT_CONFIG)
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            saved = json.load(f)
        for k in ('vocab_size', 'd_model', 'num_heads', 'num_layers',
                  'd_ff', 'block_size', 'dropout'):
            if k in saved:
                cfg[k] = saved[k]
        print(f'  Config loaded from {config_path}')

        # Rebuild encode/decode from saved vocabulary
        char2idx = saved.get('char2idx', {})
        idx2char = {int(k): v for k, v in saved.get('idx2char', {}).items()}
    else:
        char2idx = {}
        idx2char = {}

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)

    embed = state.get('token_embedding.weight')
    if embed is not None:
        cfg['vocab_size'] = embed.shape[0]
        cfg['d_model'] = embed.shape[1]

    for k, v in state.items():
        if k.endswith('ff.fc1.weight'):
            cfg['d_ff'] = v.shape[0]
            break

    layer_ids = set()
    for k in state.keys():
        if k.startswith('blocks.'):
            parts = k.split('.')
            if len(parts) > 1 and parts[1].isdigit():
                layer_ids.add(int(parts[1]))
    if layer_ids:
        cfg['num_layers'] = max(layer_ids) + 1

    # Build model
    model = StoicLM(
        vocab_size  = cfg['vocab_size'],
        d_model     = cfg['d_model'],
        num_heads   = cfg['num_heads'],
        num_layers  = cfg['num_layers'],
        d_ff        = cfg['d_ff'],
        max_seq_len = cfg['block_size'],
        dropout     = 0.0,           # inference: no dropout
    ).to(device)

    model.load_state_dict(state)
    model.eval()
    print(f'  Checkpoint loaded from {checkpoint_path}')
    print(f'  Parameters: {sum(p.numel() for p in model.parameters()):,}')

    return model, cfg, char2idx, idx2char


# =====================================
# Main chat loop
# =====================================

def chat(model, cfg, char2idx, idx2char, device):
    block_size = cfg['block_size']

    def encode(text):
        return [char2idx[ch] for ch in text if ch in char2idx]

    def decode(indices):
        return ''.join(idx2char.get(i, '') for i in indices)

    # Chat state
    temperature  = 0.75
    max_tokens   = 300
    top_k        = 40
    top_p        = 0.92
    active_phils = list(PHILOSOPHERS.keys())   # 'all' mode by default

    print_banner()

    while True:
        try:
            prompt_str = f'{BOLD}{WHITE}You >{RESET} '
            user_input = input(prompt_str).strip()
        except (EOFError, KeyboardInterrupt):
            print(f'\n\n{DIM}The wise man knows when to be silent. Farewell.{RESET}\n')
            break

        if not user_input:
            continue

        # ── Commands ──────────────────────────────────────────────────────────
        lower = user_input.lower()

        if lower in ('quit', 'exit', 'q'):
            print(f'\n{DIM}\"Perfection of character: to live each day as if it were your last.\"')
            print(f'                                        — Marcus Aurelius{RESET}\n')
            break

        if lower == 'help':
            print_help()
            continue

        if lower == 'all':
            active_phils = list(PHILOSOPHERS.keys())
            print(f'{DIM}  → All philosophers active.{RESET}')
            continue

        if lower.startswith('solo '):
            name = lower[5:].strip()
            if name in PHILOSOPHERS:
                active_phils = [name]
                print(f'{DIM}  → Solo mode: {PHILOSOPHERS[name]["name"]}{RESET}')
            else:
                print(f'{RED}  Unknown philosopher: {name}. Try: marcus, epictetus, seneca{RESET}')
            continue

        if lower.startswith('temp '):
            try:
                temperature = float(lower[5:])
                print(f'{DIM}  → Temperature set to {temperature:.2f}{RESET}')
            except ValueError:
                print(f'{RED}  Invalid value. Example: temp 0.8{RESET}')
            continue

        if lower.startswith('tokens '):
            try:
                max_tokens = int(lower[7:])
                print(f'{DIM}  → Max tokens set to {max_tokens}{RESET}')
            except ValueError:
                print(f'{RED}  Invalid value. Example: tokens 400{RESET}')
            continue

        if lower.startswith('topk '):
            try:
                top_k = int(lower[5:])
                print(f'{DIM}  → Top-k set to {top_k}{RESET}')
            except ValueError:
                print(f'{RED}  Invalid value. Example: topk 50{RESET}')
            continue

        # ── Generate responses ────────────────────────────────────────────────
        print()
        for key in active_phils:
            phil  = PHILOSOPHERS[key]
            color = phil['color']

            # Build the seed prompt the way the notebook does
            seed_prompt = f'{phil["intro"]} {user_input}:'

            # Use per-philosopher base temperature, nudged by user setting
            base_temp = TEMPERATURES[key]
            effective_temp = (base_temp + temperature) / 2.0

            print_philosopher_header(key)

            # Streaming-style: generate then display
            sys.stdout.write(f'{color}│{RESET}  {DIM}thinking...{RESET}')
            sys.stdout.flush()

            response = generate(
                model       = model,
                encode_fn   = encode,
                decode_fn   = decode,
                prompt      = seed_prompt,
                max_new_tokens = max_tokens,
                temperature = effective_temp,
                top_k       = top_k,
                top_p       = top_p,
                block_size  = block_size,
                device      = device,
            )

            clear_line()

            # Strip the seed prompt from the output so we only show new text
            display_text = response[len(seed_prompt):].lstrip()
            if not display_text:
                display_text = response   # fallback: show everything

            print_response(display_text, color)
            print()


# =====================================
# Entry point
# =====================================

def main():
    parser = argparse.ArgumentParser(
        description='Stoic Philosopher Console Chat — loads a trained StoicLM checkpoint.'
    )
    parser.add_argument(
        '--checkpoint', '-c',
        default='stoic_lm_best.pt',
        help='Path to the .pt checkpoint file  (default: stoic_lm_best.pt)'
    )
    parser.add_argument(
        '--config', '-cfg',
        default='stoic_lm_config.json',
        help='Path to the config JSON file     (default: stoic_lm_config.json)'
    )
    parser.add_argument(
        '--cpu', action='store_true',
        help='Force CPU even if CUDA is available'
    )
    args = parser.parse_args()

    device = 'cpu' if args.cpu or not torch.cuda.is_available() else 'cuda'

    print(f'\n{BOLD}Loading Stoic LM ...{RESET}')
    print(f'  Device     : {device}')

    model, cfg, char2idx, idx2char = load_model(
        checkpoint_path = args.checkpoint,
        config_path     = args.config,
        device          = device,
    )

    if not char2idx:
        print(f'{RED}WARNING: No vocabulary found in config. '
              f'Make sure stoic_lm_config.json is in the same folder.{RESET}')
        sys.exit(1)

    chat(model, cfg, char2idx, idx2char, device)


if __name__ == '__main__':
    main()