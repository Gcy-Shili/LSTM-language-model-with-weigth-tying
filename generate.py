from data import Vocabulary
from model import LSTMLanguageModel

import pickle
import torch
import os
import argparse

def load_vocab(path):
    with open(path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab

def generate(model, vocab, device, seed_text='<bos>', max_length=100, temperature=1.0):
    model.eval()
    hidden = model.init_hidden(1, device)
    tokens = seed_text.split()
    input_indices = vocab.numericalize(tokens)
    input_tensor = torch.tensor(input_indices, dtype=torch.long, device=device).unsqueeze(0)

    generated_tokens = tokens.copy()
    with torch.no_grad():
        for _ in range(max_length):
            output, hidden = model(input_tensor, hidden)
            last_logits = output[:, -1, :]
            last_logits = last_logits / temperature
            probs = torch.softmax(last_logits, dim=1)
            next_token_index = torch.multinomial(input=probs, num_samples=1).item()
            next_token = vocab.idx2word.get(next_token_index, '<unk>')
            if next_token == '<eos>':
                break
            generated_tokens.append(next_token)
            input_tensor = torch.tensor([[next_token_index]], dtype=torch.long, device=device)

    return ' '.join(generated_tokens)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(f'Using device: {device}')

    if not os.path.exists(args.vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found at {args.vocab_path}")
    vocab = load_vocab(args.vocab_path)
    print(f'Vocabulary size: {vocab.size}')

    model = LSTMLanguageModel(
        vocab_size=vocab.size,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        tied=args.tied
    ).to(device)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found at {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f'Model loaded from {args.model_path}')

    generated_text = generate(
        model,
        vocab,
        device,
        seed_text=args.seed_text,
        max_length=args.max_length,
        temperature=args.temperature
    )

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        f.write(generated_text)
    print(f'Generated text saved to {args.output_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate text using a trained LSTM Language Model')
    parser.add_argument('--model_path', type=str, default='./best_model.pt', help='Path to the trained model state dict')
    parser.add_argument('--vocab_path', type=str, default='./best_model_vocab.pkl', help='Path to the saved vocabulary')
    parser.add_argument('--output_file', type=str, default='./generated.txt', help='File to save the generated text')
    parser.add_argument('--seed_text', type=str, default='<bos>', help='Initial text to start generation')
    parser.add_argument('--max_length', type=int, default=1000, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature (higher increases diversity)')
    parser.add_argument('--embed_size', type=int, default=650, help='Embedding size (should match training)')
    parser.add_argument('--hidden_size', type=int, default=650, help='Hidden size of LSTM (should match training)')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers (should match training)')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability (should match training)')
    parser.add_argument('--tied', action='store_true', help='Enable weight tying (should match training)')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA for generation')
    args = parser.parse_args()
    main(args)
