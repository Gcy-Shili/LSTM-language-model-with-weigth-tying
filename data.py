from collections import Counter
import torch
from torch.utils.data import Dataset


class Vocabulary:
    def __init__(self, min_freq=1, specials=['<unk>', '<pad>', '<bos>', '<eos>']):
        self.min_freq = min_freq
        self.specials = specials
        self.word2idx = {}
        self.idx2word = {}
        self.freqs = Counter()
        self.size = 0

    def build_vocab(self, sentences):
        for sentence in sentences:
            self.freqs.update(sentence)
        
        for token in self.specials:
            self.word2idx[token] = self.size
            self.idx2word[self.size] = token
            self.size += 1

        for word, freq in self.freqs.items():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = self.size
                self.idx2word[self.size] = word
                self.size += 1

    def numericalize(self, sentence):
        return [self.word2idx.get(word, self.word2idx['<unk>']) for word in sentence]

    def denumericalize(self, indices):
        return [self.idx2word.get(idx, '<unk>') for idx in indices]


class PTBDataset(Dataset):
    def __init__(self, filepath, vocab=None, seq_length=30, build_vocab=False):
        self.seq_length = seq_length
        self.sentences = self._read_file(filepath)
        
        if build_vocab:
            self.vocab = Vocabulary()
            self.vocab.build_vocab(self.sentences)
        else:
            self.vocab = vocab
        
        self.data = self._numericalize(self.sentences)
        self.num_sequences = len(self.data) // self.seq_length

    def _read_file(self, filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()
        sentences = [line.strip().split() for line in lines]
        return sentences

    def _numericalize(self, sentences):
        data = []
        for sentence in sentences:
            tokens = ['<bos>'] + sentence + ['<eos>']
            data.extend(self.vocab.numericalize(tokens))
        return data

    def __len__(self):
        return (len(self.data) - 1) // self.seq_length

    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = start + self.seq_length
        input_seq = torch.tensor(self.data[start:end], dtype=torch.long)
        target_seq = torch.tensor(self.data[start + 1:end + 1], dtype=torch.long)
        return input_seq, target_seq
