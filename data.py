import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import tiktoken


def get_tokenizer():
    return tiktoken.get_encoding("gpt2")


def load_tinystories(max_tokens, enc):
    print("loading tinystories...")
    ds  = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    ids = []

    for sample in ds:
        ids.extend(enc.encode_ordinary(sample["text"]))
        if len(ids) >= max_tokens:
            break

    print(f"tokens collected: {len(ids):,}")
    return ids


class TextDataset(Dataset):
    def __init__(self, ids, seq_len):
        self.ids     = torch.tensor(ids, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.ids) - self.seq_len - 1

    def __getitem__(self, i):
        x = self.ids[i     : i + self.seq_len]
        y = self.ids[i + 1 : i + self.seq_len + 1]
        return x, y


def get_loader(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size         = batch_size,
        shuffle            = True,
        num_workers        = 2,
        pin_memory         = True,
        prefetch_factor    = 2,
        persistent_workers = True,
    )
