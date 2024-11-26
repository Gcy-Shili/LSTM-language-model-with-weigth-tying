# LSTM-language-model-with-weigth-tying

LSTM language model with weigth tying using pytorch and PTB dataset.

The model is just a simple language model with the implementation of **weight tying**.

And the idea of weight tying is from the papers:

[Using the Output Embedding to Improve Language Models (Press & Wolf 2016)](https://arxiv.org/abs/1608.05859) 

and 

[Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling (Inan et al. 2016)](https://arxiv.org/pdf/1611.01462.pdf)

the dataset and the file `getdata.sh` is from [https://github.com/salesforce/awd-lstm-lm](https://github.com/salesforce/awd-lstm-lm)

the code of the class of the model is partly from [https://github.com/floydhub/word-language-model/tree/master](https://github.com/floydhub/word-language-model/tree/master)

using the params like:
```python
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
```

and run:
```
python demo_tie.py --tied
```

the result is as follow:
```
Using device: cuda
Vocabulary size: 10002
LSTMLanguageModel(
  (embedding): Embedding(10002, 650)
  (lstm): LSTM(650, 650, num_layers=2, batch_first=True, dropout=0.5)
  (dropout): Dropout(p=0.5, inplace=False)
  (fc): Linear(in_features=650, out_features=10002, bias=True)
)
Total parameters: 13281702

Epoch: 1, Validation Loss: 5.3832, Validation Perplexity: 217.7246, Time: 0m 9s
current lr: [0.001]
Best model saved with Perplexity: 217.7246
Epoch: 2, Validation Loss: 5.1202, Validation Perplexity: 167.3639, Time: 0m 9s
current lr: [0.001]
Best model saved with Perplexity: 167.3639

......

Epoch: 40, Validation Loss: 4.3567, Validation Perplexity: 77.9963, Time: 0m 9s
current lr: [0.001]
Best model saved with Perplexity: 77.9963
-----------------------------------------
Testing: 
Test Evaluation: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 143/143 [00:00<00:00, 655.08it/s]
-----------------------------------------
Test Loss: 4.3189, Test Perplexity: 75.1059
```

And text generation can be used by using `generate.py`:
```
python generate.py --seed_text "some words" --cuda --..(some other params)
```
