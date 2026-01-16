import torch
import torch.nn as nn

class ProteinFunctionModel(nn.Module):
    def __init__(
        self,
        encoder,          # pretrained protein LM
        hidden_dim,       # encoder hidden size
        num_labels,       # number of GO terms
        dropout=0.3,
        freeze_encoder=True
    ):
        super().__init__()

        self.encoder = encoder

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, batch):
        """
        batch keys:
          input_ids      (N, L)
          attention_mask (N, L)
          protein_id     (N,)
          labels         (N, num_labels)
        """

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        protein_id = batch["protein_id"]
        labels = batch["labels"]

        # -------------------------
        # 1. Encode chunks
        # -------------------------
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # chunk embedding: (N, D)
        chunk_emb = outputs.last_hidden_state.mean(dim=1)

        # -------------------------
        # 2. Group chunks by protein (VECTORISED)
        # -------------------------
        unique_pids, inverse = torch.unique(
            protein_id, return_inverse=True
        )
        num_proteins = unique_pids.size(0)
        D = chunk_emb.size(1)

        # Mean pooling
        sum_emb = torch.zeros(
            num_proteins, D, device=chunk_emb.device
        )
        sum_emb.index_add_(0, inverse, chunk_emb)

        counts = torch.zeros(
            num_proteins, device=chunk_emb.device
        )
        counts.index_add_(
            0, inverse, torch.ones_like(inverse, dtype=torch.float)
        )

        mean_pool = sum_emb / counts.unsqueeze(1)

        # Max pooling
        max_pool = torch.full(
            (num_proteins, D),
            -1e9,
            device=chunk_emb.device
        )
        max_pool.scatter_reduce_(
            0,
            inverse.unsqueeze(1).expand(-1, D),
            chunk_emb,
            reduce="amax"
        )

        # -------------------------
        # 3. Protein embedding
        # -------------------------
        protein_emb = torch.cat([mean_pool, max_pool], dim=1)

        # -------------------------
        # 4. Classifier
        # -------------------------
        logits = self.classifier(protein_emb)

        # One label per protein
        protein_labels = labels[
            torch.unique_consecutive(inverse, return_counts=False)
        ]

        return logits, protein_labels
