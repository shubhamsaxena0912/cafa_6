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


# ==================================================
# 1. Reproducibility
# ==================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)


# ==================================================
# 2. Device
# ==================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ==================================================
# 3. Load tokenizer & encoder
# ==================================================
ENCODER_NAME = "facebook/esm2_t6_8M_UR50D"
print("Encoder:", ENCODER_NAME)

tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
encoder = AutoModel.from_pretrained(ENCODER_NAME)

hidden_dim = encoder.config.hidden_size
print("Hidden dim:", hidden_dim)


# ==================================================
# 4. Load TRAIN data
# ==================================================
train_data = json.load(open("data/processed/train_data_chunks.json"))
val_data = json.load(open("data/processed/val_data_chunks.json"))

go2idx = json.load(open("data/Train/go_terms_idx_mapping.json"))
chunk_size = 512
num_labels = len(go2idx)

print("GO terms:", num_labels)
print("Train chunks:", len(train_data))
print("Val chunks:", len(val_data))


# ==================================================
# 5. Protein ID → integer mapping (SEPARATE for train & val)
# ==================================================
def remap_protein_ids(data):
    protein_ids = sorted({item["protein_id"] for item in data})
    protein2idx = {pid: i for i, pid in enumerate(protein_ids)}
    for item in data:
        item["protein_id"] = protein2idx[item["protein_id"]]
    return protein2idx


train_protein2idx = remap_protein_ids(train_data)
val_protein2idx = remap_protein_ids(val_data)

print("Train proteins:", len(train_protein2idx))
print("Val proteins:", len(val_protein2idx))


# ==================================================
# 6. Dataset
# ==================================================
train_dataset = ProteinChunkDataset(
    data=train_data,
    tokenizer=tokenizer,
    go2idx=go2idx,
    chunk_size=chunk_size,
    aa_vocab=None
)

val_dataset = ProteinChunkDataset(
    data=val_data,
    tokenizer=tokenizer,
    go2idx=go2idx,
    chunk_size=chunk_size,
    aa_vocab=None
)


# ==================================================
# 7. protein → chunk indices mapping
# ==================================================
def build_protein_to_indices(data):
    mapping = defaultdict(list)
    for idx, item in enumerate(data):
        mapping[item["protein_id"]].append(idx)
    return mapping


train_protein_to_indices = build_protein_to_indices(train_data)
val_protein_to_indices = build_protein_to_indices(val_data)


# ==================================================
# 8. BatchSamplers
# ==================================================
MAX_CHUNKS = 512  # safe for T4

train_batch_sampler = ChunkBudgetBatchSampler(
    protein_to_indices=train_protein_to_indices,
    max_chunks=MAX_CHUNKS,
    shuffle=True
)

val_batch_sampler = ChunkBudgetBatchSampler(
    protein_to_indices=val_protein_to_indices,
    max_chunks=MAX_CHUNKS,
    shuffle=False
)


# ==================================================
# 9. DataLoaders
# ==================================================
train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_batch_sampler,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_sampler=val_batch_sampler,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True
)


# ==================================================
# 10. Model
# ==================================================
model = ProteinFunctionModel(
    encoder=encoder,
    hidden_dim=hidden_dim,
    num_labels=num_labels,
    dropout=0.3,
    freeze_encoder=True
)

model.to(device)


# ==================================================
# 11. Optimizer & loss
# ==================================================
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3,
    weight_decay=1e-4
)

criterion = nn.BCEWithLogitsLoss()


# ==================================================
# 12. Validation function
# ==================================================
@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(loader, desc="Validation", leave=False):
        batch["input_ids"] = batch["input_ids"].to(device)
        batch["attention_mask"] = batch["attention_mask"].to(device)
        batch["labels"] = batch["labels"].to(device)
        batch["protein_id"] = batch["protein_id"].to(device)

        logits, protein_labels = model(batch)
        loss = criterion(logits, protein_labels)

        total_loss += loss.item()
        num_batches += 1

    model.train()
    return total_loss / max(1, num_batches)


# ==================================================
# 13. Training loop
# ==================================================
EPOCHS = 10
GRAD_ACCUM_STEPS = 1

checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    model.train()

    epoch_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc="Training")

    for step, batch in enumerate(pbar):
        batch["input_ids"] = batch["input_ids"].to(device)
        batch["attention_mask"] = batch["attention_mask"].to(device)
        batch["labels"] = batch["labels"].to(device)
        batch["protein_id"] = batch["protein_id"].to(device)

        logits, protein_labels = model(batch)
        loss = criterion(logits, protein_labels) / GRAD_ACCUM_STEPS
        loss.backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item()
        pbar.set_postfix(train_loss=epoch_loss / (step + 1))

    train_loss = epoch_loss / len(train_loader)
    val_loss = validate(model, val_loader, criterion, device)

    print(
        f"Epoch {epoch+1} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )

    # --------------------------------------------------
    # Save checkpoint
    # --------------------------------------------------
    ckpt_path = os.path.join(
        checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt"
    )
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss
        },
        ckpt_path
    )
    print("Saved:", ckpt_path)


print("\nTraining complete.")
