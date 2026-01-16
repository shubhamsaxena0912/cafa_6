import torch
from torch.utils.data import Dataset
import math

class ProteinChunkDataset(Dataset):
    def __init__(self,
                data,              # list of dicts (your JSON entries)
                aa_vocab,
                go2idx,
                chunk_size):
        self.data = data
        self.aa_vocab = aa_vocab
        self.go2idx = go2idx
        self.num_labels = len(go2idx)
        self.chunk_size = chunk_size

    def encode_sequence(self, seq):
        return [self.aa_vocab.get(aa, self.aa_vocab["X"]) for aa in seq]

    def encode_labels(self, go_terms):
        labels = torch.zeros(self.num_labels)
        for go in go_terms:
            if go in self.go2idx:
                labels[self.go2idx[go]] = 1.0
        return labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        input_ids = torch.tensor(
            self.encode_sequence(item["sequence"]),
            dtype=torch.long
        )

        attention_mask = torch.ones(len(input_ids))

        # infer number of chunks from original length
        num_chunks = math.ceil(
            item["original_length"] / self.chunk_size
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "protein_id": item["protein_id"],
            "chunk_id": item["chunk_index"],
            "num_chunks": num_chunks,
            "labels": self.encode_labels(item["go_terms"])
        }


def collate_fn(batch):
    batch_size = len(batch)
    max_len = max(x["input_ids"].size(0) for x in batch)

    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len)

    labels = torch.stack([x["labels"] for x in batch])

    protein_ids = []
    chunk_ids = []
    num_chunks = []

    for i, x in enumerate(batch):
        seq_len = x["input_ids"].size(0)
        input_ids[i, :seq_len] = x["input_ids"]
        attention_mask[i, :seq_len] = 1

        protein_ids.append(x["protein_id"])
        chunk_ids.append(x["chunk_id"])
        num_chunks.append(x["num_chunks"])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "protein_id": protein_ids,
        "chunk_id": chunk_ids,
        "num_chunks": num_chunks
    }
