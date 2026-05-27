import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import math
import copy
import re
import time
import random
import os
import urllib.request
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings('ignore')


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


GUTENBERG_SOURCES = {
    'Marcus Aurelius — Meditations':   'https://www.gutenberg.org/cache/epub/2680/pg2680.txt',
    'Epictetus — Enchiridion':         'https://www.gutenberg.org/cache/epub/45109/pg45109.txt',
    'Seneca — Letters from a Stoic':   'https://www.gutenberg.org/cache/epub/900/pg900.txt',
}

def fetch_text(url: str, timeout: int = 15) -> str:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            raw = r.read().decode('utf-8', errors='replace')
        return raw
    except Exception as e:
        print(f'  Could not fetch {url}: {e}')
        return ''

def strip_gutenberg_header_footer(text: str) -> str:
    start_markers = ['*** START OF THE PROJECT', '*** START OF THIS PROJECT',
                     '*END*THE SMALL PRINT', 'THE FULL PROJECT']
    end_markers   = ['*** END OF THE PROJECT', '*** END OF THIS PROJECT',
                     'End of the Project Gutenberg']
    for m in start_markers:
        idx = text.find(m)
        if idx != -1:
            text = text[idx + len(m):]
            break
    for m in end_markers:
        idx = text.find(m)
        if idx != -1:
            text = text[:idx]
            break
    return text.strip()

def clean_text(text: str) -> str:
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'[^\x20-\x7E\n]', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


corpus_parts = []
for title, url in GUTENBERG_SOURCES.items():
    print(f'Fetching: {title} ...')
    raw = fetch_text(url)
    if raw:
        cleaned = clean_text(strip_gutenberg_header_footer(raw))
        corpus_parts.append(cleaned)
        print(f'  {len(cleaned):,} chars')
    else:
        print('  Skipped')

FULL_CORPUS = '\n\n'.join(corpus_parts)
print(f'\nTotal corpus size: {len(FULL_CORPUS):,} characters')


STOIC_FALLBACK = """
You have power over your mind, not outside events. Realize this, and you will find strength.
The happiness of your life depends upon the quality of your thoughts.
Very little is needed to make a happy life; it is all within yourself, in your way of thinking.
When you arise in the morning, think of what a precious privilege it is to be alive, to breathe, to think, to enjoy, to love.
If it is not right, do not do it; if it is not true, do not say it.
Confine yourself to the present. What is essential is invisible to the eye.
Never esteem anything as of advantage to you that will make you break your word or lose your self-respect.
The first rule is to keep an untroubled spirit. The second is to look things in the face and know them for what they are.
Loss is nothing else but change, and change is nature's delight.
The object of life is not to be on the side of the majority, but to escape finding oneself in the ranks of the insane.
He who fears death will never do anything worthy of a man who is alive.
Man is disturbed not by things, but by the opinions and fancies he forms about things.
Seek not the good in external things; seek it in yourself.
We suffer more in imagination than in reality.
It is not that I am brave, it is that I am busy. There is a difference.
Freedom is the only worthy goal in life. It is won by disregarding things that lie beyond our control.
Seek not for events to happen as you wish but rather wish for events to happen as they do, and you will have tranquility.
Make the best use of what is in your power, and take the rest as it happens.
No man is free who is not master of himself.
First say to yourself what you would be; then do what you have to do.
It is not what happens to you, but how you react to it that matters.
Wealth consists not in having great possessions, but in having few wants.
He is a wise man who does not grieve for the things which he has not, but rejoices for those which he has.
We are more often frightened than hurt; and we suffer more from imagination than from reality.
Difficulties strengthen the mind, as labor does the body.
It is not the man who has too little, but the man who craves more, that is poor.
Retire into yourself as much as possible with those who will improve you, and let in those who you yourself can improve.
Begin at once to live, and count each separate day as a separate life.
The mind that is anxious about future events is miserable.
As long as you live, keep learning how to live.
Associate with those who will make a better man of you.
Waste no more time arguing about what a good man should be. Be one.
The best revenge is to be unlike him who performed the injury.
Accept the things to which fate binds you, and love the people with whom fate brings you together.
Everything we hear is an opinion, not a fact. Everything we see is a perspective, not the truth.
The impediment to action advances action. What stands in the way becomes the way.
You have power over your mind, not outside events. Realize this, and you will find strength.
Perfection of character is this: to live each day as if it were your last, without frenzy, without apathy, without pretense.
Execute every act of thy life as though it were thy last.
The soul becomes dyed with the colour of its thoughts.
To live a good life: We have the potential for it. If we can learn to be indifferent to what makes no difference.
Whoever does not regard what he has as most ample wealth, is unhappy, though he be master of the world.
Let not your mind run on what you lack as much as on what you have already.
How soon will you be ashes or bare bones, and either a name or not even a name; and a name is just noise and echo.
We are what we repeatedly do. Excellence, therefore, is not an act but a habit.
Dwell on the beauty of life. Watch the stars, and see yourself running with them.
Look back over the past, with its changing empires that rose and fell, and you can foresee the future, too.
The impediment to action advances action. What stands in the way becomes the way.
Confine yourself to the present.
Nothing happens to any man that he is not formed by nature to bear.
A man's life is a mere moment, his existence a flux, his perception clouded, his body's composition corruptible.
In your actions, do not procrastinate. In your conversations, do not confuse. In your thoughts, do not wander.
The whole future lies in uncertainty: live immediately.
True happiness is to enjoy the present, without anxious dependence upon the future.
Endure and renounce. That is the whole of philosophy.
Reason is not measured by size or height, but by principle.
God, grant me the serenity to accept what I cannot change, the courage to change what I can, and the wisdom to know the difference.
The more we value things outside our control, the less control we have.
People are not disturbed by things, but by their opinions about things.
There is only one way to happiness and that is to cease worrying about things which are beyond the power of our will.
Practice yourself in little things and thence proceed to greater.
Never say about anything I have lost it. Only say I have given it back.
Seek not that the things which happen should happen as you wish; but wish the things which happen to be as they are, and you will have a tranquil flow of life.
He is a wise man who does not grieve for the things which he has not, but rejoices for those which he has.
"""

if len(FULL_CORPUS) < 5000:
    print('Using fallback inline Stoic corpus (network unavailable).')

    FULL_CORPUS = (STOIC_FALLBACK.strip() + '\n\n') * 120
    print(f'Fallback corpus: {len(FULL_CORPUS):,} chars')


print(f'\nSample (first 500 chars):\n{FULL_CORPUS[:500]}')


n_chars  = len(FULL_CORPUS)
n_words  = len(FULL_CORPUS.split())
n_lines  = FULL_CORPUS.count('\n')

print('Corpus Statistics')
print('─' * 35)
print(f'  Characters : {n_chars:>12,}')
print(f'  Words      : {n_words:>12,}')
print(f'  Lines      : {n_lines:>12,}')


chars     = sorted(set(FULL_CORPUS))
VOCAB_SIZE = len(chars)
print(f'Vocabulary size: {VOCAB_SIZE} characters')
print(f'Characters: {repr("".join(chars))}')


char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for ch, i in char2idx.items()}

def encode(text: str) -> list:
    return [char2idx[ch] for ch in text if ch in char2idx]

def decode(indices: list) -> str:
    return ''.join(idx2char.get(i, '') for i in indices)


DATA = torch.tensor(encode(FULL_CORPUS), dtype=torch.long)
print(f'\nEncoded tensor: {DATA.shape}')


SPLIT = 0.90
n_train = int(len(DATA) * SPLIT)
train_data = DATA[:n_train]
val_data   = DATA[n_train:]
print(f'Train tokens: {len(train_data):,}')
print(f'Val   tokens: {len(val_data):,}')


class StoicDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int):
        self.data       = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx      : idx + self.block_size]
        y = self.data[idx + 1  : idx + self.block_size + 1]
        return x, y




class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, 'd_model must be divisible by num_heads'

        self.d_model    = d_model
        self.num_heads  = num_heads
        self.d_k        = d_model // num_heads


        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_probs, V)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, _, T, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.d_model)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_out = self.scaled_dot_product_attention(Q, K, V, mask)
        return self.W_o(self.combine_heads(attn_out))




class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1  = nn.Linear(d_model, d_ff)
        self.fc2  = nn.Linear(d_ff,    d_model)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))




class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe       = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)




class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads)
        self.ff         = PositionWiseFeedForward(d_model, d_ff)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        attn_out = self.self_attn(x, x, x, mask)
        x        = self.norm1(x + self.dropout(attn_out))


        ff_out   = self.ff(x)
        x        = self.norm2(x + self.dropout(ff_out))
        return x




class StoicLM(nn.Module):
    def __init__(
        self,
        vocab_size:    int,
        d_model:       int,
        num_heads:     int,
        num_layers:    int,
        d_ff:          int,
        max_seq_len:   int,
        dropout:       float = 0.1
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding    = PositionalEncoding(d_model, max_seq_len, dropout)
        self.blocks          = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm   = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        self.d_model     = d_model
        self.max_seq_len = max_seq_len


        self.lm_head.weight = self.token_embedding.weight


        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _causal_mask(self, size: int, device) -> torch.Tensor:
        mask = torch.tril(torch.ones(size, size, device=device)).unsqueeze(0).unsqueeze(0)
        return mask

    def forward(
        self,
        x:      torch.Tensor,
        targets: torch.Tensor = None
    ):
        B, T = x.shape
        assert T <= self.max_seq_len, f'Sequence length {T} exceeds max {self.max_seq_len}'


        tok_emb = self.token_embedding(x)
        h       = self.pos_encoding(tok_emb)


        mask = self._causal_mask(T, x.device)


        for block in self.blocks:
            h = block(h, mask)

        h      = self.norm(h)
        logits = self.lm_head(h)

        if targets is None:
            return logits, None


        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )
        return logits, loss

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)






D_MODEL      = 256
NUM_HEADS    = 8
NUM_LAYERS   = 4
D_FF         = 1024
BLOCK_SIZE   = 128
DROPOUT      = 0.10


BATCH_SIZE   = 64
LEARNING_RATE = 3e-4
NUM_EPOCHS   = 30
GRAD_CLIP    = 1.0


train_dataset = StoicDataset(train_data, BLOCK_SIZE)
val_dataset   = StoicDataset(val_data,   BLOCK_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          drop_last=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          drop_last=False, num_workers=0, pin_memory=True)


model = StoicLM(
    vocab_size  = VOCAB_SIZE,
    d_model     = D_MODEL,
    num_heads   = NUM_HEADS,
    num_layers  = NUM_LAYERS,
    d_ff        = D_FF,
    max_seq_len = BLOCK_SIZE,
    dropout     = DROPOUT
).to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

print(f'Model parameters : {model.count_parameters():,}')
print(f'Vocab size        : {VOCAB_SIZE}')
print(f'Context window    : {BLOCK_SIZE} tokens')
print(f'Train batches/ep  : {len(train_loader)}')
print(f'\nModel:\n{model}')




@torch.no_grad()
def estimate_loss(model, loader, max_batches: int = 40) -> float:
    model.eval()
    total, count = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)
        _, loss = model(x, y)
        total += loss.item()
        count += 1
    model.train()
    return total / max(count, 1)




train_losses = []
val_losses   = []
best_val_loss = float('inf')
CKPT_PATH = '/kaggle/working/stoic_lm_best.pt'

print('Training Stoic Transformer LM')
print('=' * 55)

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    t0 = time.time()

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()


        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()
        epoch_loss += loss.item()

    scheduler.step()

    avg_train = epoch_loss / len(train_loader)
    avg_val   = estimate_loss(model, val_loader)

    train_losses.append(avg_train)
    val_losses.append(avg_val)

    elapsed = time.time() - t0
    lr_now  = scheduler.get_last_lr()[0]

    print(f'Ep {epoch:03d}/{NUM_EPOCHS}  |  '
          f'train {avg_train:.4f}  val {avg_val:.4f}  |  '
          f'lr {lr_now:.2e}  |  {elapsed:.1f}s')


    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(), CKPT_PATH)

print(f'\nTraining complete. Best val loss: {best_val_loss:.4f}')




epochs = range(1, len(train_losses) + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle('Stoic Transformer LM — Training Curves', fontsize=14, fontweight='bold')


axes[0].plot(epochs, train_losses, label='Train loss', color='#1f77b4', linewidth=2)
axes[0].plot(epochs, val_losses,   label='Val loss',   color='#d62728', linewidth=2, linestyle='--')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Cross-Entropy Loss')
axes[0].set_title('Loss')
axes[0].legend()
axes[0].grid(alpha=0.3)


train_ppl = [math.exp(l) for l in train_losses]
val_ppl   = [math.exp(l) for l in val_losses]
axes[1].plot(epochs, train_ppl, label='Train PPL', color='#1f77b4', linewidth=2)
axes[1].plot(epochs, val_ppl,   label='Val PPL',   color='#d62728', linewidth=2, linestyle='--')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Perplexity')
axes[1].set_title('Perplexity')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/kaggle/working/training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'Final train perplexity: {train_ppl[-1]:.2f}')
print(f'Final val   perplexity: {val_ppl[-1]:.2f}')




if os.path.exists(CKPT_PATH):
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    print(f'Loaded best checkpoint from {CKPT_PATH}')
model.eval()




@torch.no_grad()
def generate(
    model,
    prompt:      str,
    max_new_tokens: int  = 300,
    temperature: float   = 0.8,
    top_k:       int     = 50,
    top_p:       float   = 0.95,
) -> str:
    model.eval()


    context = torch.tensor(encode(prompt), dtype=torch.long, device=DEVICE).unsqueeze(0)

    generated = list(encode(prompt))

    for _ in range(max_new_tokens):

        ctx = context[:, -BLOCK_SIZE:]

        logits, _ = model(ctx)
        logits     = logits[:, -1, :] / temperature


        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')


        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_logits[cum_probs - F.softmax(sorted_logits, dim=-1) > top_p] = float('-inf')

            logits = logits.scatter(1, sorted_idx, sorted_logits)

        probs   = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        context = torch.cat([context, next_id], dim=1)
        generated.append(next_id.item())

    return decode(generated)




PROMPTS = [
    ('On the nature of death',   'Death is nothing to fear, for'),
    ('On virtue and reason',     'Virtue consists not in'),
    ('On controlling the mind',  'You have power over your mind'),
    ('On daily practice',        'Begin each morning with the thought'),
    ('On external things',       'Seek not the good in external'),
]

print('=' * 65)
print('STOIC PHILOSOPHER — GENERATED TEXTS')
print('=' * 65)

for topic, seed in PROMPTS:
    print(f'\nTopic: {topic}')
    print(f'Seed  : "{seed}"')
    print('─' * 55)
    text = generate(
        model, seed,
        max_new_tokens=250,
        temperature=0.75,
        top_k=40,
        top_p=0.92
    )
    print(text)
    print()




PHILOSOPHER_INTROS = {
    'marcus':   'I, Marcus Aurelius, Emperor of Rome, write to you from the frontier:',
    'epictetus':'Epictetus speaks. Remember, slave or emperor, your will is your own:',
    'seneca':   'Dear friend, Seneca writes on the matter of the soul:',
}

TOPICS = {
    'death':    'On the nature of death and impermanence:',
    'virtue':   'Concerning virtue and the good life:',
    'control':  'What lies within our power and what does not:',
    'fortune':  'On Fortune and adversity:',
    'mind':     'On mastering the ruling faculty of the mind:',
    'time':     'On the use of time and the present moment:',
    'anger':    'On passion, anger, and tranquility:',
    'wealth':   'On wealth, poverty, and true richness:',
}

def stoic_response(
    philosopher: str = 'marcus',
    topic:       str = 'virtue',
    temperature: float = 0.75,
    max_tokens:  int   = 350,
) -> str:
    intro = PHILOSOPHER_INTROS.get(philosopher.lower(),
                                   PHILOSOPHER_INTROS['marcus'])
    subj  = TOPICS.get(topic.lower(), topic)
    prompt = f'{intro} {subj}'

    return generate(
        model, prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=40,
        top_p=0.92
    )




CONVERSATIONS = [
    ('marcus',   'death'),
    ('epictetus','control'),
    ('seneca',   'time'),
    ('marcus',   'anger'),
    ('seneca',   'wealth'),
]

print('=' * 65)
print('STOIC PHILOSOPHER CHAT')
print('=' * 65)

for philosopher, topic in CONVERSATIONS:
    print(f'\n[You ask {philosopher.capitalize()} about "{topic}"]')
    print('─' * 55)
    response = stoic_response(philosopher, topic, temperature=0.72)
    print(response)
    print()










SCRIPT = [
    ('marcus',   'death',                               0.70),
    ('epictetus','control',                             0.72),
    ('seneca',   'time',                                0.75),
    ('marcus',   'virtue',                              0.68),
    ('epictetus','wealth',                              0.80),
    ('seneca',   'anger',                               0.73),
    ('marcus',   'mind',                                0.70),
    ('epictetus','fortune',                             0.76),
    ('seneca',   'On the friendship of wise men:',      0.78),
    ('marcus',   'On the soldier and his duty:',        0.72),
    ('epictetus','On what we owe to others:',           0.74),
    ('seneca',   'On retreat and solitude:',            0.80),
]

print('=' * 65)
print('STOIC PHILOSOPHER — FULL SCRIPTED DIALOGUE')
print('   (Edit SCRIPT in this cell to customise topics)')
print('=' * 65)

for turn, (philosopher, topic, temp) in enumerate(SCRIPT, 1):
    label = TOPICS.get(topic.lower(), topic)
    print(f'\n[Turn {turn:02d}]  {philosopher.capitalize()} — "{label}"')
    print('─' * 55)
    response = stoic_response(philosopher, topic, temperature=temp, max_tokens=300)
    print(response)

print('\n' + '=' * 65)
print('Session complete. The wise man knows when to be silent.')
print('=' * 65)




attention_maps = []

def attention_hook(module, input, output):
    pass


class MHAWithRecord(MultiHeadAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_attn = None

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_probs = torch.softmax(attn_scores, dim=-1)
        self.last_attn = attn_probs.detach().cpu()
        return torch.matmul(attn_probs, V)



class StoicLMRecord(StoicLM):
    pass



VIZ_PHRASE = 'You have power over your mind'
tokens     = encode(VIZ_PHRASE[:BLOCK_SIZE])
ctx        = torch.tensor(tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)


attn_weights_list = []

def make_hook():
    def hook_fn(module, inp, out):
        pass
    return hook_fn

with torch.no_grad():
    T = ctx.shape[1]
    tok_emb  = model.token_embedding(ctx)
    h        = model.pos_encoding(tok_emb)
    mask     = model._causal_mask(T, DEVICE)


    block = model.blocks[0]
    Q = block.self_attn.split_heads(block.self_attn.W_q(h))
    K = block.self_attn.split_heads(block.self_attn.W_k(h))
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(block.self_attn.d_k)
    attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
    attn_probs  = torch.softmax(attn_scores, dim=-1).squeeze(0).cpu()

print(f'Attention tensor shape: {attn_probs.shape}  (heads, T, T)')


chars_to_show = [idx2char[i] for i in tokens]
chars_display = [repr(c) for c in chars_to_show]

n_heads_plot = min(4, NUM_HEADS)
fig, axes    = plt.subplots(1, n_heads_plot, figsize=(4 * n_heads_plot, 4))
fig.suptitle(f'Attention Maps — Block 0\n"{VIZ_PHRASE}"', fontsize=12, fontweight='bold')

for h_idx, ax in enumerate(axes):
    im = ax.imshow(attn_probs[h_idx].numpy(), cmap='Blues', vmin=0, vmax=1)
    ax.set_title(f'Head {h_idx}')
    ax.set_xticks(range(T))
    ax.set_yticks(range(T))
    ax.set_xticklabels(chars_display, rotation=90, fontsize=7)
    ax.set_yticklabels(chars_display, fontsize=7)

plt.colorbar(im, ax=axes, shrink=0.6)
plt.tight_layout()
plt.savefig('/kaggle/working/attention_maps.png', dpi=150, bbox_inches='tight')
plt.show()




import json

config = {
    'vocab_size':  VOCAB_SIZE,
    'char2idx':    char2idx,
    'idx2char':    {str(k): v for k, v in idx2char.items()},
    'd_model':     D_MODEL,
    'num_heads':   NUM_HEADS,
    'num_layers':  NUM_LAYERS,
    'd_ff':        D_FF,
    'block_size':  BLOCK_SIZE,
    'dropout':     DROPOUT,
    'num_epochs':  NUM_EPOCHS,
    'best_val_loss': best_val_loss,
    'parameters':  model.count_parameters(),
}

with open('/kaggle/working/stoic_lm_config.json', 'w') as f:
    json.dump(config, f, indent=2)

torch.save(model.state_dict(), '/kaggle/working/stoic_lm_final.pt')

print('Saved artefacts:')
print('  /kaggle/working/stoic_lm_best.pt')
print('  /kaggle/working/stoic_lm_final.pt')
print('  /kaggle/working/stoic_lm_config.json')
print('  /kaggle/working/training_curves.png')
print('  /kaggle/working/attention_maps.png')




print('╔══════════════════════════════════════════════════════════╗')
print('║          STOIC LM — EXPERIMENT SUMMARY              ║')
print('╠══════════════════════════════════════════════════════════╣')
print(f'║  Architecture  : Decoder-only Transformer (GPT-style)   ║')
print(f'║  Corpus        : Marcus Aurelius, Epictetus, Seneca      ║')
print(f'║  Vocab size    : {VOCAB_SIZE:<5}  (character-level)             ║')
print(f'║  Parameters    : {model.count_parameters():>10,}                         ║')
print(f'║  d_model       : {D_MODEL:<5}                                  ║')
print(f'║  Attention heads: {NUM_HEADS:<4}                                  ║')
print(f'║  Decoder layers: {NUM_LAYERS:<4}                                  ║')
print(f'║  Context window: {BLOCK_SIZE:<4} tokens                          ║')
print(f'║  Best val loss : {best_val_loss:<6.4f}                               ║')
print(f'║  Best val PPL  : {math.exp(best_val_loss):<8.2f}                             ║')
print('╚══════════════════════════════════════════════════════════╝')
print()
print('"Waste no more time arguing about what a good man should be. Be one."')
print('                                          — Marcus Aurelius')
