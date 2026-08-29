"""GraphSAGE / GAT encoders and the CNN baseline for Task 2 (PDF S4.2).

GraphSAGE layer, as written in the PDF:

    h_i^(l+1) = sigma( W^(l) . CONCAT( h_i^(l), MEAN_{j in N(i)} h_j^(l) ) )

`SAGEConv` implements exactly this concat-aggregate form. Readout is mean
pooling over nodes, then a linear classifier:

    g = (1/|V|) sum_i h_i^(L),   y_hat = softmax(W g + b)

The graph encoder is exposed separately from the classifier because Task 3
fuses `g` with the BERT text vector via cross-attention.
"""

import torch
from torch import nn


class GraphEncoder(nn.Module):
    """Stack of GraphSAGE or GAT layers followed by mean pooling.

    Args:
        in_dim: node feature width from `graph_builder.segment_features`.
        hidden_dim: width of each message-passing layer.
        n_layers: 2 by default. On ~20-node graphs, 3+ layers over-smooth —
            every node's receptive field covers the whole graph and the
            pooled vectors collapse toward each other.
        conv: 'sage' or 'gat'.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        conv: str = "sage",
        heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        from torch_geometric.nn import GATConv, SAGEConv

        if conv not in ("sage", "gat"):
            raise ValueError(f"conv must be 'sage' or 'gat', got {conv!r}")
        self.conv_type = conv
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        width = in_dim
        for _ in range(n_layers):
            if conv == "sage":
                self.layers.append(SAGEConv(width, hidden_dim))
                width = hidden_dim
            else:
                self.layers.append(
                    GATConv(width, hidden_dim // heads, heads=heads, dropout=dropout)
                )
                width = hidden_dim
            self.norms.append(nn.BatchNorm1d(width))
        self.out_dim = width
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        """Returns the graph-level vector g, shape (n_graphs, out_dim)."""
        from torch_geometric.nn import global_mean_pool

        for layer, norm in zip(self.layers, self.norms):
            x = layer(x, edge_index)
            x = norm(x)
            x = torch.relu(x)
            x = self.dropout(x)
        return global_mean_pool(x, batch)


class GNNClassifier(nn.Module):
    """GraphEncoder + linear head. Genre is single-label, so logits feed CE."""

    def __init__(self, in_dim: int, n_classes: int, **kwargs) -> None:
        super().__init__()
        self.encoder = GraphEncoder(in_dim, **kwargs)
        self.head = nn.Linear(self.encoder.out_dim, n_classes)

    def forward(self, x, edge_index, batch):
        return self.head(self.encoder(x, edge_index, batch))


class CNNBaseline(nn.Module):
    """B2: 2-D CNN over the log-mel spectrogram — no graph, no text.

    The control that answers the question Task 2 actually asks: does
    relational structure beat treating the spectrogram as an image?
    """

    def __init__(self, n_classes: int, n_mels: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        channels = [1, 32, 64, 128, 128]
        blocks = []
        for i in range(len(channels) - 1):
            blocks += [
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[i + 1]),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(channels[-1], n_classes)

    def forward(self, x):
        """x: (B, 1, n_mels, T) log-mel spectrogram."""
        h = self.pool(self.features(x)).flatten(1)
        return self.head(self.dropout(h))


def count_parameters(model: nn.Module) -> int:
    """Trainable parameter count — reported so the GNN/CNN comparison is fair."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
