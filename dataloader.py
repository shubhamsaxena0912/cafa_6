import torch
from torch.utils.data import Dataset
import math
import random
from torch.utils.data.sampler import Sampler

# Dataset for protein chunking
class ProteinChunkDataset(Dataset):
    def __init__(self,
                data,              # list of dicts (your JSON entries)
                aa_vocab,
                tokenizer,
                go2idx,
                chunk_size):
        self.data = data
        self.aa_vocab = aa_vocab
        self.tokenizer = tokenizer
        self.go2idx = go2idx
        self.num_labels = len(go2idx)
        self.chunk_size = chunk_size

    # def encode_sequence(self, seq):
    #     return [self.aa_vocab.get(aa, self.aa_vocab["X"]) for aa in seq]

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

        tokens = self.tokenizer(
            item["sequence"],
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_tensors="pt"
        )

        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

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


# Sampler for protein chunking (Protein ID based batching to ensure all chunks in same batch)
class ChunkBudgetBatchSampler(Sampler):
    def __init__(self, protein_to_indices, max_chunks, shuffle=True):
        self.protein_to_indices = protein_to_indices
        self.max_chunks = max_chunks
        self.shuffle = shuffle
        self.protein_ids = list(protein_to_indices.keys())

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.protein_ids)

        batch_indices = []
        current_chunks = 0

        for pid in self.protein_ids:
            indices = self.protein_to_indices[pid]
            n = len(indices)

            if current_chunks + n > self.max_chunks and batch_indices:
                yield batch_indices
                batch_indices = []
                current_chunks = 0

            batch_indices.extend(indices)
            current_chunks += n

        if batch_indices:
            yield batch_indices

    def __len__(self):
        return len(self.protein_ids)



# Collate function for protein chunking (to handle variable sequence lengths)
def collate_fn(batch):
    batch_size = len(batch)
    max_len = max(x["input_ids"].size(0) for x in batch)

    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)

    labels = torch.stack([x["labels"] for x in batch])

    protein_ids = []
    chunk_ids = []
    num_chunks = []

    for i, x in enumerate(batch):
        seq_len = x["input_ids"].size(0)
        input_ids[i, :seq_len] = x["input_ids"]
        attention_mask[i, :seq_len] = x["attention_mask"]

        protein_ids.append(x["protein_id"])
        chunk_ids.append(x["chunk_id"])
        num_chunks.append(x["num_chunks"])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "protein_id": torch.tensor(protein_ids, dtype=torch.long),
        "chunk_id": torch.tensor(chunk_ids, dtype=torch.long),
        "num_chunks": torch.tensor(num_chunks, dtype=torch.long)
    }
