import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot_encoding(word, word2index):
    one_hot_v = [0]*(len(word2index))
    index = word2index[word]
    one_hot_v[index] = 1
    return one_hot_v


class CBOW(nn.Module):
    def __init__(self, vocab_size, emb_size, padding_idx=0):
        super(CBOW, self).__init__()
        self.emb = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_size,
            padding_idx=padding_idx,
        )
        self.linear = nn.Linear(
            in_features=emb_size,
            out_features=vocab_size,
        )

    def forward(self, inputs):
        x = self.emb(inputs).sum(dim=1)
        x = F.dropout(x, 0.3)
        y = self.linear(x)
        y = F.softmax(y)
        return y





class SKIP(nn.Module):
    def __init__(self):
