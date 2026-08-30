"""GNN-BERT fusion for Task 3 (PDF S4.3).

Cross-attention fusion, exactly as the PDF writes it:

    A = softmax(Q K^T / sqrt(d)),  Q = g W_Q,  K = H_text W_K
    z = CONCAT(g, A H_text),       y_hat = sigmoid(W z)

The graph vector `g` is a single query attending over the BERT token sequence
`H_text`, so the model chooses which words matter given the audio structure.

Four modes are implemented behind one class so the S4.3 ablation
(BERT-only / GNN-only / early concat / cross-attention) is a flag rather
than four diverging code paths that could differ in some unintended way:

    bert      z = t                      (text only, no graph)
    gnn       z = g                      (graph only, no text)
    concat    z = CONCAT(g, t)           (early fusion on the CLS vector)
    crossattn z = CONCAT(g, A H_text)    (the PDF's recommended form)

The emotion head implements the auxiliary term of

    L = L_tags + alpha*||v - v_hat||^2 + beta*||a - a_hat||^2

DEAM is disjoint from the tag corpora, so no batch carries both targets. Per
spec S2.1 the loss is masked: `multitask_loss` zeroes whichever term has no
supervision in the current batch.
"""

import torch
from torch import nn


class CrossAttentionFusion(nn.Module):
    """Single-query cross-attention from the graph vector onto text tokens."""

    def __init__(self, graph_dim: int, text_dim: int, attn_dim: int = 256) -> None:
        super().__init__()
        self.q = nn.Linear(graph_dim, attn_dim)
        self.k = nn.Linear(text_dim, attn_dim)
        self.v = nn.Linear(text_dim, attn_dim)
        self.scale = attn_dim ** 0.5
        self.out_dim = graph_dim + attn_dim

    def forward(self, g, h_text, attention_mask=None):
        """g: (B, Dg) | h_text: (B, L, Dt) -> (z, attention weights)."""
        q = self.q(g).unsqueeze(1)                    # (B, 1, A)
        k = self.k(h_text)                            # (B, L, A)
        v = self.v(h_text)                            # (B, L, A)
        scores = (q @ k.transpose(1, 2)) / self.scale  # (B, 1, L)
        if attention_mask is not None:
            scores = scores.masked_fill(
                attention_mask.unsqueeze(1) == 0, torch.finfo(scores.dtype).min
            )
        attn = torch.softmax(scores, dim=-1)
        context = (attn @ v).squeeze(1)               # (B, A)
        return torch.cat([g, context], dim=-1), attn.squeeze(1)


class GNNBertFusion(nn.Module):
    """End-to-end fusion model with a tag head and an emotion head.

    Args:
        mode: 'bert' | 'gnn' | 'concat' | 'crossattn'.
        freeze_bert: keeps the text tower fixed. Useful when the graph branch
            is still untrained, since a fine-tuning BERT otherwise dominates
            early optimisation and the GNN never learns anything.
    """

    MODES = ("bert", "gnn", "concat", "crossattn")

    def __init__(
        self,
        n_tags: int,
        node_dim: int,
        mode: str = "crossattn",
        bert_name: str = "distilbert-base-uncased",
        hidden_dim: int = 128,
        n_layers: int = 2,
        conv: str = "gat",
        attn_dim: int = 256,
        dropout: float = 0.3,
        freeze_bert: bool = False,
        n_emotion: int = 2,
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.mode = mode

        self.text_encoder = None
        self.graph_encoder = None
        z_dim = 0

        if mode != "gnn":
            from transformers import AutoModel
            self.text_encoder = AutoModel.from_pretrained(bert_name)
            if freeze_bert:
                for p in self.text_encoder.parameters():
                    p.requires_grad = False
            text_dim = self.text_encoder.config.hidden_size
        if mode != "bert":
            from src.gnn_model import GraphEncoder
            self.graph_encoder = GraphEncoder(
                node_dim, hidden_dim=hidden_dim, n_layers=n_layers,
                conv=conv, dropout=dropout,
            )
            graph_dim = self.graph_encoder.out_dim

        if mode == "bert":
            z_dim = text_dim
        elif mode == "gnn":
            z_dim = graph_dim
        elif mode == "concat":
            z_dim = graph_dim + text_dim
        else:
            self.fusion = CrossAttentionFusion(graph_dim, text_dim, attn_dim)
            z_dim = self.fusion.out_dim

        self.z_dim = z_dim
        self.dropout = nn.Dropout(dropout)
        self.tag_head = nn.Linear(z_dim, n_tags)
        self.emotion_head = nn.Linear(z_dim, n_emotion)

    def represent(self, x=None, edge_index=None, batch=None,
                  input_ids=None, attention_mask=None):
        """Fused representation z — the vector t-SNE is run on."""
        attn = None
        if self.mode == "gnn":
            return self.graph_encoder(x, edge_index, batch), None

        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        h_text = out.last_hidden_state
        t = h_text[:, 0]                                   # CLS

        if self.mode == "bert":
            return t, None

        g = self.graph_encoder(x, edge_index, batch)
        if self.mode == "concat":
            return torch.cat([g, t], dim=-1), None
        z, attn = self.fusion(g, h_text, attention_mask)
        return z, attn

    def forward(self, x=None, edge_index=None, batch=None,
                input_ids=None, attention_mask=None):
        z, attn = self.represent(x, edge_index, batch, input_ids, attention_mask)
        z = self.dropout(z)
        return self.tag_head(z), self.emotion_head(z), attn


def multitask_loss(
    tag_logits,
    tag_targets,
    emotion_pred,
    emotion_targets,
    tag_mask,
    emotion_mask,
    alpha: float = 1.0,
    beta: float = 1.0,
    pos_weight=None,
):
    """L = L_tags + alpha*||v-v_hat||^2 + beta*||a-a_hat||^2, masked per sample.

    `tag_mask` / `emotion_mask` are (B,) float tensors marking which samples
    carry that supervision. A batch drawn entirely from DEAM has tag_mask all
    zero, so the tag term contributes nothing rather than training against
    fabricated all-zero labels — which is what silently happens if you feed
    zeros and forget the mask.
    """
    import torch.nn.functional as F

    device = tag_logits.device
    total = torch.zeros((), device=device)
    parts = {}

    if tag_mask.sum() > 0:
        per = F.binary_cross_entropy_with_logits(
            tag_logits, tag_targets, reduction="none",
            pos_weight=pos_weight,
        ).mean(dim=1)
        tag_loss = (per * tag_mask).sum() / tag_mask.sum()
        total = total + tag_loss
        parts["tag"] = float(tag_loss.detach())

    if emotion_mask.sum() > 0:
        se = (emotion_pred - emotion_targets) ** 2      # (B, 2)
        v = (se[:, 0] * emotion_mask).sum() / emotion_mask.sum()
        a = (se[:, 1] * emotion_mask).sum() / emotion_mask.sum()
        total = total + alpha * v + beta * a
        parts["valence_mse"] = float(v.detach())
        parts["arousal_mse"] = float(a.detach())

    return total, parts
