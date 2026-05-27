#!/usr/bin/env python3
"""
Extract the actual character vocabulary from the trained model.

The model's token embedding has shape (vocab_size, d_model).
We can't know the actual characters from just weights, but we can infer
the vocab_size and rebuild a reasonable character set to match.

Usage:
    python extract_vocab_from_model.py --checkpoint stoic_lm_best.pt --output stoic_lm_config.json
"""

import argparse
import json
import torch
import sys

def main():
    parser = argparse.ArgumentParser(description='Extract vocab size from trained model')
    parser.add_argument('--checkpoint', '-c', default='stoic_lm_best.pt',
                       help='Path to checkpoint')
    parser.add_argument('--output', '-o', default='stoic_lm_config.json',
                       help='Output config path')
    args = parser.parse_args()
    
    print(f'\n📦  Loading checkpoint: {args.checkpoint}')
    
    try:
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    except Exception as e:
        print(f'❌  Failed to load: {e}')
        sys.exit(1)
    
    # Get vocab size from token_embedding.weight
    if 'token_embedding.weight' not in state:
        print(f'❌  token_embedding.weight not found in checkpoint')
        sys.exit(1)
    
    vocab_size = state['token_embedding.weight'].shape[0]
    print(f'   vocab_size from embedding: {vocab_size}')
    
    # Build a complete character set to match vocab_size
    # Start with common characters
    base_chars = '\n ,.\'"!?;:-()[]{}0123456789'
    letters_lower = 'abcdefghijklmnopqrstuvwxyz'
    letters_upper = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    special = 'éèêëàâäãöôõòüûúùçñ'
    
    all_chars = base_chars + letters_lower + letters_upper + special
    
    # Trim or pad to exact vocab_size
    chars = list(dict.fromkeys(all_chars))[:vocab_size]  # Remove duplicates, preserve order
    
    # If we still need more, add ASCII symbols
    if len(chars) < vocab_size:
        import string
        for ch in string.printable:
            if ch not in chars and len(chars) < vocab_size:
                chars.append(ch)
    
    if len(chars) < vocab_size:
        # Pad with spaces or blanks
        while len(chars) < vocab_size:
            chars.append(f'<pad_{len(chars)-len(all_chars)}>')
    
    chars = chars[:vocab_size]
    
    print(f'   Generated {len(chars)} characters')
    
    # Create mappings
    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = {str(i): ch for i, ch in enumerate(chars)}
    
    # Build config
    config = {
        'vocab_size': vocab_size,
        'd_model': state['token_embedding.weight'].shape[1],
        'num_heads': 8,  # from training defaults
        'num_layers': 4,
        'd_ff': 1024,
        'block_size': 128,
        'dropout': 0.1,
        'char2idx': char2idx,
        'idx2char': idx2char,
        'auto_generated': True,
        'note': 'This config was auto-generated from the checkpoint. '
                'If decoding produces gibberish, the vocab is incorrect. '
                'Re-run the Kaggle notebook with the correct character set.',
    }
    
    # Save
    with open(args.output, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f'\n✅  Config saved to {args.output}')
    print(f'   char2idx: {len(char2idx)} entries')
    print(f'   idx2char: {len(idx2char)} entries')
    
    # Test
    print(f'\n📝  Test encoding:')
    test = 'Marcus Aurelius'
    enc = [char2idx.get(ch, char2idx.get(' ')) for ch in test]
    dec = ''.join(idx2char.get(str(i), '?') for i in enc)
    print(f'   "{test}" → {dec}')

if __name__ == '__main__':
    main()
