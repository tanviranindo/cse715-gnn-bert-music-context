"""BERT text encoder + multi-label tag head (Task 1, PDF S4.1).

    t = BERT_CLS(X_text),   y_hat_k = sigmoid(w_k . t + b_k)

Trained with per-tag binary cross-entropy. The encoder is also reused as the
text branch of the Task 3 fusion model and the text tower in Task 4, so it is
exposed separately from the classification head.
"""

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL = "distilbert-base-uncased"


class BertTagger(nn.Module):
    """DistilBERT/BERT encoder with a linear multi-label head.

    Args:
        n_tags: size of the tag vocabulary (50 for both Task 1 variants).
        model_name: any HuggingFace encoder checkpoint.
        freeze_encoder: train the head only. Much cheaper, and the honest
            setting for a "frozen BERT" ablation, but it underfits badly on
            metadata-style text.
        pooling: 'cls' matches the PDF's BERT_CLS formulation; 'mean' pools
            over non-padding tokens and is usually stronger on short text.
    """

    def __init__(
        self,
        n_tags: int,
        model_name: str = DEFAULT_MODEL,
        freeze_encoder: bool = False,
        dropout: float = 0.1,
        pooling: str = "cls",
    ) -> None:
        super().__init__()
        if pooling not in ("cls", "mean"):
            raise ValueError(f"pooling must be 'cls' or 'mean', got {pooling!r}")
        self.encoder = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_tags)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pooled sentence representation — the vector Task 3 fuses with the GNN."""
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        if self.pooling == "cls":
            return hidden[:, 0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns raw logits; apply sigmoid for probabilities."""
        return self.head(self.dropout(self.encode(input_ids, attention_mask)))


class TagDataset(torch.utils.data.Dataset):
    """Tokenised (text, multi-hot label) pairs."""

    def __init__(
        self,
        records: list[dict],
        vocab: list[str],
        tokenizer,
        max_length: int = 128,
    ) -> None:
        self.records = records
        self.index = {t: i for i, t in enumerate(vocab)}
        self.n_tags = len(vocab)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        r = self.records[i]
        enc = self.tokenizer(
            r["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        y = torch.zeros(self.n_tags)
        for t in r["labels"]:
            if t in self.index:
                y[self.index[t]] = 1.0
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": y,
        }


def load_tokenizer(model_name: str = DEFAULT_MODEL):
    """PDF S3 preprocessing step 4: BERT tokenizer, max length 128-256."""
    return AutoTokenizer.from_pretrained(model_name)


def pos_weight_from_records(records: list[dict], vocab: list[str]) -> torch.Tensor:
    """Per-tag positive weight for BCEWithLogitsLoss.

    Tag prevalence spans roughly 2%-25%, so unweighted BCE drives the rare
    tags to always-negative and silently destroys Macro-F1.
    """
    n = len(records)
    counts = [0] * len(vocab)
    index = {t: i for i, t in enumerate(vocab)}
    for r in records:
        for t in r["labels"]:
            if t in index:
                counts[index[t]] += 1
    return torch.tensor(
        [(n - c) / max(c, 1) for c in counts], dtype=torch.float
    ).clamp(max=50.0)
