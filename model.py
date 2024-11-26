from torch import nn

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.5, tied=False):
        super(LSTMLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.tied = tied
        if tied:
            if hidden_size != embed_size:
                raise ValueError("When using the tied flag, hidden_size must be equal to embed_size")
            self.fc.weight = self.embedding.weight


    def forward(self, x, hidden):
        embeds = self.dropout(self.embedding(x))
        output, hidden = self.lstm(embeds, hidden)   # (batch_size, seq_length, hidden_size)
        output = self.dropout(output)
        if self.tied:
            decoded = self.fc(output.reshape(output.size(0) * output.size(1), output.size(2)))   
            # output.view  ->  (batch_size * seq_length, hidden_size); self.fc  ->  (batch_size * seq_length, vocab_size)
            return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden    # decoded.view  ->  (batch_size, seq_length, vocab_size)
        else:
            logits = self.fc(output)
            return logits, hidden

    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        return (weight.new_zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device),
                weight.new_zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device))
