import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(52)

BATCH_SIZE = 64
TRAIN_SPLIT = 0.9
CTX_LENGTH = 128

N_HEADS = 6
N_EMBD = 128
N_BLOCKS = 4
VOCAB_SIZE = 50000

EPOCHS = 100
VAL_INTERVAL = 10
LR = 1e-2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_size):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embed_matrix = nn.Parameter(torch.randn((num_embeddings, embedding_size)))

    def forward(self, x):
        '''
        x is of shape (B, C) or (C)
        the output is of shape (B, C, embedding_size) or (C, embedding_size)
        '''
        one_h = nn.functional.one_hot(x, num_classes=self.num_embeddings).float()
        return torch.matmul(one_h, self.embed_matrix)

class TokenEmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.embedding_layer = Embedding(vocab_size, n_embd)

    def forward(self, x):
        '''
        x is of shape (BATCH_SIZE, CTX_LENGTH)
        returns embedding of shape (BATCH_SIZE, CTX_LENGTH, n_embd)
        '''
        return self.embedding_layer(x)

class PositionEmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, n_embd):
        super.__init__()
        self.embedding_layer = Embedding(vocab_size, n_embd)

    def forward(self, x):
        '''
        x is of shape (BATCH_SIZE, CTX_LENGTH)
        returns embedding of shape (BATCH_SIZE, CTX_LENGTH, n_embd)
        '''
        _, ctx_length = x.shape
        positions = torch.arange(0, ctx_length) # (CTX_LENGTH)
        return self.embedding_layer(positions) # (CTX_LENGTH, n_embd)

class LayerNorm(nn.Module):
    def __init__(self, n_embd, eps):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(n_embd))
        self.beta = nn.Parameter(torch.zeros(n_embd))
        self.eps = eps

    def forward(self, x):
        '''
        x is of shape (B, C, E) or (C, E)
        normalize x along the last dimension
        '''
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        out = (x - mean) / (torch.sqrt(var + self.eps))
        out = out * self.gamma + self.beta
        return out

class FeedForwardLayer(nn.Module):
    pass

class AttentionLayer(nn.Module):
    pass

class MultiHeadAttentionLayer(nn.Module):
    pass

class Block(nn.Module):
    pass

class Transformer(nn.Module):
    def __init__(self):
        self.token_embd_layer = TokenEmbeddingLayer(VOCAB_SIZE, N_EMBD)
        self.pos_embd_layer = TokenEmbeddingLayer(VOCAB_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block(N_EMBD, N_HEADS) for _ in range(N_BLOCKS)])
        self.layer_norm = nn.LayerNorm(N_EMBD)
        self.final_layer = nn.Linear(N_EMBD, VOCAB_SIZE)

    def forward(self, idx, targets=None):
        pass

    def sample(self, idx, max_samples):
        '''
        Generate more ids starting from give ids using the model.
        '''
        for _ in range(max_samples):
            pass
        return idx

if __name__ == '__main__':
    emb = Embedding(26, 4)
    o = emb(torch.tensor([[5,10,22],[5,10,2]]))
    print(o) # 2 x 3 x 4
    print(LayerNorm(4, 1e-5)(o))