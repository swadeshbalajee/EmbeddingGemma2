"""Memory-mapped dataset storage for cached frozen embeddings and decoder targets."""
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class CachedDataset(Dataset):
    def __init__(self, directory):
        self.root = Path(directory)
        with (self.root / "metadata.json").open() as f:
            self.meta = json.load(f)
        self.embedding = np.load(self.root / "embeddings.npy", mmap_mode="r")
        self.tokens = np.load(self.root / "tokens.npy", mmap_mode="r")
        self.lengths = np.load(self.root / "lengths.npy", mmap_mode="r")
        assert len(self.embedding) == len(self.tokens) == len(self.lengths) == self.meta["count"]

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, i):
        n = int(self.lengths[i])
        return (torch.from_numpy(np.array(self.embedding[i], dtype=np.float32)),
                torch.from_numpy(np.array(self.tokens[i, :n], dtype=np.int64)))


class LengthBucketBatchSampler:
    """Shuffle records, then group nearby lengths to reduce padding work."""
    def __init__(self, sampler, lengths, batch_size, bucket_size=1024, seed=42):
        if bucket_size < batch_size or bucket_size % batch_size:
            raise ValueError("bucket_size must be a multiple of batch_size")
        self.sampler = sampler
        self.lengths = lengths
        self.batch_size = int(batch_size)
        self.bucket_size = int(bucket_size)
        self.seed = int(seed)

    def __len__(self):
        return len(self.sampler) // self.batch_size

    def __iter__(self):
        indices = list(iter(self.sampler))
        batches = []
        for start in range(0, len(indices), self.bucket_size):
            group = indices[start:start + self.bucket_size]
            group.sort(key=lambda i: int(self.lengths[i]))
            for batch_start in range(0, len(group) - self.batch_size + 1, self.batch_size):
                batches.append(group[batch_start:batch_start + self.batch_size])
        random.Random(self.seed + self.sampler.epoch).shuffle(batches)
        yield from batches


def collate(samples, bos_id, pad_id):
    z = torch.stack([s[0] for s in samples])
    max_t = max(len(s[1]) for s in samples)
    targets = torch.full((len(samples), max_t), -100, dtype=torch.long)
    inputs = torch.full((len(samples), max_t), pad_id, dtype=torch.long)
    mask = torch.zeros((len(samples), max_t), dtype=torch.long)
    for i, (_, target) in enumerate(samples):
        n = len(target)
        targets[i, :n] = target
        inputs[i, 0] = bos_id
        if n > 1:
            inputs[i, 1:n] = target[:-1]
        mask[i, :n] = 1
    return {"embeddings": z, "input_ids": inputs, "labels": targets, "mask": mask}


def different_indices(n, device):
    # Random cyclic permutation with no fixed points, provided records are unique.
    if n < 2:
        raise ValueError("Need batch >=2 for negatives")
    shift = int(torch.randint(1, n, (), device=device).item())
    return (torch.arange(n, device=device) + shift) % n
