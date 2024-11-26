from tqdm import tqdm
import argparse
import os
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import PTBDataset
from model import LSTMLanguageModel

class Trainer:
    def __init__(self, model, train_loader, valid_loader, criterion, optimizer, scheduler, vocab, device, clip=5.0):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.vocab = vocab
        self.device = device
        self.clip = clip

    def train_epoch(self, epoch):
        self.model.train()
        hidden = self.model.init_hidden(self.train_loader.batch_size, self.device)
        total_loss = 0
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch} Training")
        for batch, (inputs, targets) in progress_bar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            hidden = tuple([h.data for h in hidden])
            self.optimizer.zero_grad()
            logits, hidden = self.model(inputs, hidden)
            loss = self.criterion(logits.view(-1, logits.size(2)), targets.view(-1))
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
            total_loss += loss.item()

            if (batch + 1) % 100 == 0:
                avg_loss = total_loss / 100
                progress_bar.set_postfix({'Loss': f'{avg_loss:.4f}', 'Perplexity': f'{math.exp(avg_loss):.4f}'})
                total_loss = 0

    def evaluate(self, loader, description="Validation"):
        self.model.eval()
        hidden = self.model.init_hidden(loader.batch_size, self.device)
        total_loss = 0
        progress_bar = tqdm(enumerate(loader), total=len(loader), desc=f"{description} Evaluation")
        with torch.no_grad():
            
            for batch, (inputs, targets) in progress_bar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                logits, hidden = self.model(inputs, hidden)
                loss = self.criterion(logits.view(-1, logits.size(2)), targets.view(-1))
                total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        perplexity = math.exp(avg_loss)
        return avg_loss, perplexity

    def test(self, test_loader):
        print("-----------------------------------------\nTesting: ")
        return self.evaluate(test_loader, description="Test")


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_dataset = PTBDataset(os.path.join(args.data_dir, 'train.txt'), seq_length=args.seq_length, build_vocab=True)
    vocab = train_dataset.vocab
    print(f'Vocabulary size: {vocab.size}')

    valid_dataset = PTBDataset(os.path.join(args.data_dir, 'valid.txt'), vocab=vocab, seq_length=args.seq_length)
    test_dataset = PTBDataset(os.path.join(args.data_dir, 'test.txt'), vocab=vocab, seq_length=args.seq_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    model = LSTMLanguageModel(
        vocab_size=vocab.size,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        tied=args.tied,
    ).to(device)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_factor, patience=args.lr_patience)

    trainer = Trainer(model, train_loader, valid_loader, criterion, optimizer, scheduler, vocab, device, clip=args.clip)

    best_valid_ppl = float('inf')

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        trainer.train_epoch(epoch)
        valid_loss, valid_ppl = trainer.evaluate(valid_loader, description="Validation")
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(int(end_time - start_time), 60)
        print(f'Epoch: {epoch}, Validation Loss: {valid_loss:.4f}, Validation Perplexity: {valid_ppl:.4f}, Time: {epoch_mins}m {epoch_secs}s')

        scheduler.step(metrics=valid_ppl)

        current_lr = scheduler.get_last_lr()
        print(f'current lr: {current_lr}')

        if valid_ppl < best_valid_ppl:
            best_valid_ppl = valid_ppl
            torch.save(model.state_dict(), args.save_path)
            print(f'Best model saved with Perplexity: {best_valid_ppl:.4f}')

    model.load_state_dict(torch.load(args.save_path))
    test_loss, test_ppl = trainer.test(test_loader)
    print("-----------------------------------------")
    print(f'Test Loss: {test_loss:.4f}, Test Perplexity: {test_ppl:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM Language Model on Penn Treebank')

    parser.add_argument('--data_dir', type=str, default='./data/penn/', help='Directory containing train.txt, valid.txt, test.txt')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size')
    parser.add_argument('--embed_size', type=int, default=650, help='Embedding size')
    parser.add_argument('--hidden_size', type=int, default=650, help='Hidden size of LSTM')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=40, help='Number of training epochs')
    parser.add_argument('--clip', type=float, default=5.0, help='Gradient clipping')
    parser.add_argument('--seq_length', type=int, default=30, help='Sequence length')
    parser.add_argument('--save_path', type=str, default='best_model.pt', help='Path to save the best model')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 regularization)')
    parser.add_argument('--lr_factor', type=float, default=0.5, help='Factor by which the learning rate will be reduced')
    parser.add_argument('--lr_patience', type=int, default=2, help='Number of epochs with no improvement after which learning rate will be reduced')
    parser.add_argument('--tied', action='store_true', help='Enable weight tying')

    args = parser.parse_args()

    main(args=args)
