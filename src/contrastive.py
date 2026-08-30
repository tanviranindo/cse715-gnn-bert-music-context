"""Contrastive GNN-BERT dual encoder for Task 4 (PDF S4.4).

    L_NCE = -log( exp(sim(g_i, t_i)/tau) / sum_j exp(sim(g_i, t_j)/tau) )
    sim(u, v) = u.v / (||u|| ||v||)

Two towers project into a shared space: a graph encoder over MusicCaps audio
structure graphs, and BERT over the expert captions. Unlike Tasks 1-3 this is
not classification — nothing is predicted, only aligned — so the metrics are
retrieval recalls in both directions.

The loss is computed symmetrically (caption->audio and audio->caption) and
averaged, which is standard for dual encoders: optimising only one direction
lets the other collapse.
"""

import torch
from torch import nn


class DualEncoder(nn.Module):
    """Graph tower + text tower projected into a shared embedding space.

    Args:
        embed_dim: shared space width. Both towers project to this.
        learn_temperature: if True, tau is learned in log space (as in CLIP).
            A fixed tau is easier to reason about but a learned one usually
            converges faster on small corpora like MusicCaps (~5k pairs).
    """

    def __init__(
        self,
        node_dim: int,
        embed_dim: int = 256,
        bert_name: str = "distilbert-base-uncased",
        hidden_dim: int = 128,
        n_layers: int = 2,
        conv: str = "gat",
        dropout: float = 0.1,
        temperature: float = 0.07,
        learn_temperature: bool = True,
        freeze_bert: bool = False,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        from src.gnn_model import GraphEncoder

        self.graph_encoder = GraphEncoder(
            node_dim, hidden_dim=hidden_dim, n_layers=n_layers,
            conv=conv, dropout=dropout,
        )
        self.text_encoder = AutoModel.from_pretrained(bert_name)
        if freeze_bert:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        self.graph_proj = nn.Linear(self.graph_encoder.out_dim, embed_dim)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, embed_dim)

        init = torch.tensor(float(temperature)).log()
        if learn_temperature:
            self.log_temperature = nn.Parameter(init)
        else:
            self.register_buffer("log_temperature", init)

    def encode_graph(self, x, edge_index, batch):
        g = self.graph_proj(self.graph_encoder(x, edge_index, batch))
        return nn.functional.normalize(g, dim=-1)

    def encode_text(self, input_ids, attention_mask):
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        t = self.text_proj(out.last_hidden_state[:, 0])
        return nn.functional.normalize(t, dim=-1)

    def forward(self, x, edge_index, batch, input_ids, attention_mask):
        return self.encode_graph(x, edge_index, batch), \
               self.encode_text(input_ids, attention_mask)

    @property
    def temperature(self) -> torch.Tensor:
        # clamped so a learned tau cannot collapse to 0 and produce inf logits
        return self.log_temperature.exp().clamp(min=1e-3, max=1.0)


def info_nce(graph_emb, text_emb, temperature) -> tuple[torch.Tensor, dict]:
    """Symmetric InfoNCE over an in-batch-negatives similarity matrix.

    Both embeddings must already be L2-normalised, so `graph_emb @ text_emb.T`
    is the cosine similarity of every pair. The diagonal holds the true pairs.
    """
    logits = graph_emb @ text_emb.t() / temperature
    targets = torch.arange(len(logits), device=logits.device)
    loss_g2t = nn.functional.cross_entropy(logits, targets)
    loss_t2g = nn.functional.cross_entropy(logits.t(), targets)
    loss = (loss_g2t + loss_t2g) / 2
    return loss, {
        "audio_to_caption": float(loss_g2t.detach()),
        "caption_to_audio": float(loss_t2g.detach()),
    }


@torch.no_grad()
def recall_at_k(
    query_emb, gallery_emb, ks: tuple[int, ...] = (1, 5, 10)
) -> dict[str, float]:
    """R@K where row i of `query_emb` matches row i of `gallery_emb`.

    Ranks the whole gallery for each query, so the numbers depend on gallery
    size and are only comparable at equal test-set size.
    """
    sims = query_emb @ gallery_emb.t()
    n = len(sims)
    ranks = (sims > sims.diag().unsqueeze(1)).sum(dim=1)
    return {f"R@{k}": float((ranks < k).float().mean()) for k in ks}


@torch.no_grad()
def median_rank(query_emb, gallery_emb) -> float:
    """Median rank of the true match (1 is perfect). Complements R@K, which
    saturates and hides how badly the failures fail."""
    sims = query_emb @ gallery_emb.t()
    ranks = (sims > sims.diag().unsqueeze(1)).sum(dim=1) + 1
    return float(ranks.float().median())


@torch.no_grad()
def zero_shot_tag_scores(text_emb_tags, graph_emb):
    """Zero-shot tagging: score every clip against embedded tag prompts.

    PDF S4.4 asks for zero-shot tag prediction from captions compared with
    Task 3's supervised model. Tag names are encoded by the text tower and
    scored against clip graph embeddings, so no tag supervision is used.
    """
    return graph_emb @ text_emb_tags.t()
