import torch
from torch import nn
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
import einops
from WikiText2 import WikiTextDataset
from torch.utils.data import random_split
from tqdm import tqdm

torch.manual_seed(52)

BATCH_SIZE = 64
TRAIN_SPLIT = 0.9
CTX_LENGTH = 128

N_HEADS = 4
N_EMBD = 128
N_BLOCKS = 4
VOCAB_SIZE = 50000

N_EPOCHS = 10
VAL_ITERS = 100
LR = 3e-4

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@torch.no_grad()
def estimate_loss(m, loader):
    m.eval()
    losses = torch.zeros(VAL_ITERS)
    for batch_index, (x, y) in tqdm(enumerate(loader)):
        _, loss = m(x.to(DEVICE), y.to(DEVICE))
        losses[batch_index-1] = loss.item()        
        if batch_index == VAL_ITERS:
            break
    m.train()
    return losses.mean()

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
    def __init__(self, n_embd, eps=1e-5):
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
    def __init__(self, n_in, n_hidden, n_out, dropout_p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_out),
            nn.Dropout(dropout_p)
        )
    
    def forward(self, x):
        return self.net(x)

class AttentionHead(nn.Module):
    def __init__(self, n_embd, head_size, ctx_length, dropout_p):
        super().__init__()
        self.ctx_length = ctx_length
        self.key = nn.Linear(n_embd, head_size)
        self.query = nn.Linear(n_embd, head_size)
        self.value = nn.Linear(n_embd, head_size)
        self.register_buffer('tril', torch.tril(torch.ones(ctx_length, ctx_length)))
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        '''
        x is of shape (B, ctx_length, n_embd)
        '''
        k = self.key(x) # (B, ctx_length, head_size)
        q = self.key(x) # (B, ctx_length, head_size)
        k_T = einops.rearrange(k, 'b c h -> b h c')
        w = torch.einsum('b c h, b h C -> b c C', q, k_T)
        w = w / (self.ctx_length ** 0.5)
        w = w.masked_fill(self.tril[:,:] == 0, float('-inf'))
        w = F.softmax(w, dim=-1)
        w = self.dropout(w)
        v = self.value(x)
        return w @ v # (B, ctx_length, head_size)

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, n_embd, n_heads, ctx_length, dropout_p=0.2):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(n_embd, n_embd // n_heads, ctx_length, dropout_p) for _ in range(n_heads)])
        self.project = nn.Linear(n_embd, n_embd)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, ctx_length, n_embd)
        return self.project(out) # (B, ctx_length, n_embd)

class Block(nn.Module):
    def __init__(self, n_embd, n_heads, ctx_length):
        super().__init__()
        self.multi_head_attention = MultiHeadAttentionLayer(n_embd, n_heads, ctx_length)
        self.ffwd = FeedForwardLayer(n_embd, 4*n_embd, n_embd)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.multi_head_attention(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embd_layer = TokenEmbeddingLayer(VOCAB_SIZE, N_EMBD)
        self.pos_embd_layer = TokenEmbeddingLayer(VOCAB_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block(N_EMBD, N_HEADS, CTX_LENGTH) for _ in range(N_BLOCKS)])
        self.layer_norm = nn.LayerNorm(N_EMBD)
        self.final_layer = nn.Linear(N_EMBD, VOCAB_SIZE)

    def forward(self, idx, targets=None):
        # idx - (B, C)
        tok = self.token_embd_layer(idx)
        pos = self.pos_embd_layer(idx)
        x = tok + pos # (B, C, N_EMBD)
        x = self.blocks(x) # (B, C, N_EMBD)
        x = self.layer_norm(x)
        logits = self.final_layer(x) # (B, C, VOCAB_SIZE)

        if targets == None:
            loss = None
        else:
            logits = einops.rearrange(logits, 'b c v -> (b c) v')
            targets = einops.rearrange(targets, 'b c -> (b c)')
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def sample(self, idx, max_samples):
        '''
        Generate more ids starting from give ids using the model.
        '''
        for _ in range(max_samples):
            idx_cond = idx[:, -CTX_LENGTH:]
            logits, _ = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

if __name__ == '__main__':
    ds = WikiTextDataset(CTX_LENGTH)
    VOCAB_SIZE = ds.vocab_size
    train_size = int(TRAIN_SPLIT * len(ds))
    val_size = len(ds) - train_size
    indices = list(range(len(ds)))
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    g = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(ds, (train_size, val_size), g)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE)

    m = Transformer().to(DEVICE)
    optimizer = torch.optim.Adam(m.parameters(), LR)
    for i in range(N_EPOCHS):
        epoch_loss = 0
        for batch_index, (x, y) in tqdm(enumerate(train_loader)):
            logits, loss = m(x.to(DEVICE), y.to(DEVICE))
            epoch_loss += loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        val_loss = estimate_loss(m, val_loader)
        print(f"Epoch {i} Loss {epoch_loss.item()/batch_index} Val Loss {val_loss}")
    context = torch.tensor([ds.encode(ds.text[:CTX_LENGTH])], dtype=torch.long, device=DEVICE)
    print(ds.decode(m.sample(context, max_samples=500)[0].tolist()))
