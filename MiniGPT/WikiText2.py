import torch
from torch.utils.data import Dataset, random_split

FOLDER_PATH = "wikitext-2-raw/wiki.raw"

class WikiTextDataset(Dataset):
    def __init__(self, block_size, device="cpu",):
        def getData():
            with open(FOLDER_PATH, mode="r", encoding="utf-8") as f:
                txt = f.read()
            return txt
        
        self.text = getData()

        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        self.block_size = block_size
        
        self.stoi = {ch:i for i,ch in enumerate(self.chars)}
        self.itos = {i:ch for i,ch in enumerate(self.chars)}

        self.encode = lambda s: [self.stoi[ch] for ch in s]
        self.decode = lambda idx: "".join([self.itos[i] for i in idx])

        self.encode_batch = lambda ss: [self.encode(s) for s in ss]
        self.decode_batch = lambda idxs: "".join([self.decode(idx) for idx in idxs])

        self.data = torch.tensor(self.encode(self.text), dtype=torch.long, device=device)
        
    def __getitem__(self, index):
        return self.data[index : index + self.block_size], self.data[index + 1 : index + self.block_size + 1]

    def __getitems__(self, indices):
        x = torch.stack([self[index][0] for index in indices])
        y = torch.stack([self[index][1] for index in indices])
        return x,y

    def __len__(self):
        return len(self.data)-self.block_size

if __name__ == '__main__':
    ds = WikiTextDataset(128)
    train_size = int(0.3 * len(ds))
    val_size = len(ds) - train_size
    g = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(ds, (train_size, val_size), g)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1)

    print(ds.decode(train_ds[4646*64][0].tolist()))
    print(len(train_loader))
    for batch_index, (x, y) in enumerate(train_loader):
        # print(batch_index)
        x_decode = ds.decode_batch(x.tolist())