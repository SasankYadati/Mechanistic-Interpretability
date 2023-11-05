import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

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
        return len(self.data-self.block_size+1)
    

if __name__ == '__main__':
    ds = WikiTextDataset(128)
    train_size = int(0.9 * len(ds))
    val_size = len(ds) - train_size
    indices = list(range(len(ds)))
    train_indices, val_indices = indices[:train_size], indices[train_size:]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(ds, batch_size=1, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(ds, batch_size=1, sampler=val_sampler)

    for batch_index, (x, y) in enumerate(train_loader):
        pass