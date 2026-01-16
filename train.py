import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer
from collections import defaultdict
from tqdm import tqdm
import random
import numpy as np
import json
import os

from dataloader import (
    ProteinChunkDataset,
    ChunkBudgetBatchSampler,
    collate_fn
)
from model import ProteinFunctionModel


# --------------------------------------------------
# 1. Reproducibility
# --------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)


# --------------------------------------------------
# 2. Device
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# --------------------------------------------------
# 3. Load tokenizer & encoder
# --------------------------------------------------
ENCODER_NAME = "facebook/esm2_t6_8M_UR50D"
print("Encoder name:", ENCODER_NAME)

tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
encoder = AutoModel.from_pretrained(ENCODER_NAME)

hidden_dim = encoder.config.hidden_size
print("Hidden dimension:", hidden_dim)


# --------------------------------------------------
# 4. Load your data
# --------------------------------------------------
data = json.load(open("data/processed/train_data_chunks.json"))
go2idx = json.load(open("data/Train/go_terms_idx_mapping.json"))
chunk_size = 512

num_labels = len(go2idx)

print("Number of GO terms:", num_labels)
print("Number of data:", len(data))

# --------------------------------------------------
# 5. Protein ID → integer mapping (MANDATORY)
# --------------------------------------------------
all_protein_ids = sorted({item["protein_id"] for item in data})
protein2idx = {pid: i for i, pid in enumerate(all_protein_ids)}

for item in data:
    item["protein_id"] = protein2idx[item["protein_id"]]


# --------------------------------------------------
# 6. Dataset
# --------------------------------------------------
dataset = ProteinChunkDataset(
    data=data,
    tokenizer=tokenizer,
    go2idx=go2idx,
    chunk_size=chunk_size,
    aa_vocab=None  # unused, safe to pass None or remove param
)


# --------------------------------------------------
# 7. protein → chunk indices mapping (CRITICAL)
# --------------------------------------------------
protein_to_indices = defaultdict(list)
for idx, item in enumerate(data):
    protein_to_indices[item["protein_id"]].append(idx)

print("Number of proteins:", len(protein_to_indices))
# --------------------------------------------------
# 8. Chunk-budget BatchSampler
# --------------------------------------------------
MAX_CHUNKS = 512  # tune based on GPU

batch_sampler = ChunkBudgetBatchSampler(
    protein_to_indices=protein_to_indices,
    max_chunks=MAX_CHUNKS,
    shuffle=True
)


# --------------------------------------------------
# 9. DataLoader
# --------------------------------------------------
loader = DataLoader(
    dataset,
    batch_sampler=batch_sampler,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True
)


# --------------------------------------------------
# 10. Model
# --------------------------------------------------
model = ProteinFunctionModel(
    encoder=encoder,
    hidden_dim=hidden_dim,
    num_labels=num_labels,
    dropout=0.3,
    freeze_encoder=True
)

model.to(device)


# --------------------------------------------------
# 11. Optimizer & loss
# --------------------------------------------------
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3,
    weight_decay=1e-4
)

criterion = nn.BCEWithLogitsLoss()


# --------------------------------------------------
# 12. Training loop
# --------------------------------------------------
EPOCHS = 10
GRAD_ACCUM_STEPS = 1  # increase if GPU is small

checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

model.train()

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    epoch_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for step, batch in enumerate(pbar):
        # Move batch to device
        batch["input_ids"] = batch["input_ids"].to(device)
        batch["attention_mask"] = batch["attention_mask"].to(device)
        batch["labels"] = batch["labels"].to(device)
        batch["protein_id"] = batch["protein_id"].to(device)

        # Forward
        logits, protein_labels = model(batch)

        loss = criterion(logits, protein_labels)
        loss = loss / GRAD_ACCUM_STEPS
        loss.backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item()
        pbar.set_postfix(loss=epoch_loss / (step + 1))

    avg_loss = epoch_loss / len(loader)
    print(f"Epoch {epoch+1} | Avg loss: {avg_loss:.4f}")

    # --------------------------------------------------
    # 13. Save checkpoint
    # --------------------------------------------------
    
    
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        },
        checkpoint_path
    )
    print(f"Checkpoint saved to {checkpoint_path}")


print("Training complete.")
