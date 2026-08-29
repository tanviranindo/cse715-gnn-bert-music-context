"""Music structure graphs for Task 2 (PDF S3 step 3, S4.2).

The PDF names two graph types:

* **Segment graph** — nodes are time segments, edges are temporal adjacency
  plus cosine similarity of MFCC/chroma above a threshold tau.
* **Chord-transition graph** — nodes are unique chords, edges are observed
  transitions weighted by count.

Both are built here. Segment graphs are the primary input to the GNN because
chord graphs on a 30 s FMA clip typically collapse to a handful of nodes.

**Deviation from the PDF, stated deliberately.** S3 suggests 5-10 s windows.
FMA-small clips are 30 s, so that yields 3-6 nodes — too few for message
passing to do anything, since two GraphSAGE layers would see the whole graph
from every node. We use `segment_seconds=1.5` (=20 nodes on a 30 s clip) and
report the window length as an ablation axis.
"""

import numpy as np

# 12 pitch classes, sharps
PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def segment_features(
    y: np.ndarray, sr: int, segment_seconds: float = 1.5, n_mfcc: int = 20
) -> np.ndarray:
    """Per-segment audio descriptors -> (n_segments, d) node feature matrix.

    Each node carries MFCC mean/std, chroma mean, and spectral
    centroid/rolloff/zero-crossing statistics. Audio only — no text, no
    labels — as S4.2 requires ("audio-only node features").
    """
    import librosa

    window = int(sr * segment_seconds)
    if window <= 0 or len(y) < window:
        return np.zeros((0, 2 * n_mfcc + 12 + 6), dtype=np.float32)

    rows = []
    for start in range(0, len(y) - window + 1, window):
        seg = y[start:start + window]
        mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=n_mfcc)
        chroma = librosa.feature.chroma_stft(y=seg, sr=sr)
        cent = librosa.feature.spectral_centroid(y=seg, sr=sr)
        roll = librosa.feature.spectral_rolloff(y=seg, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(seg)
        rows.append(
            np.concatenate([
                mfcc.mean(axis=1), mfcc.std(axis=1),
                chroma.mean(axis=1),
                [cent.mean(), cent.std(), roll.mean(), roll.std(),
                 zcr.mean(), zcr.std()],
            ])
        )
    return np.asarray(rows, dtype=np.float32)


def cosine_similarity_matrix(x: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity of row vectors."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    unit = x / norms
    return unit @ unit.T


def build_segment_graph(
    features: np.ndarray, tau: float = 0.9, add_adjacency: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Segment graph -> (edge_index [2, E], edge_weight [E]).

    Edges are undirected and stored in both directions, as PyTorch Geometric
    expects. Self-loops are excluded; GNN layers add their own.
    """
    n = len(features)
    if n == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    sim = cosine_similarity_matrix(features)
    pairs: dict[tuple[int, int], float] = {}

    if add_adjacency:
        for i in range(n - 1):
            pairs[(i, i + 1)] = float(max(sim[i, i + 1], 0.0))

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= tau:
                pairs[(i, j)] = float(sim[i, j])

    if not pairs:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    src, dst, w = [], [], []
    for (i, j), weight in pairs.items():
        src += [i, j]
        dst += [j, i]
        w += [weight, weight]
    return (
        np.asarray([src, dst], dtype=np.int64),
        np.asarray(w, dtype=np.float32),
    )


# --------------------------------------------------------------- chord graph

def _chord_templates() -> tuple[np.ndarray, list[str]]:
    """24 binary triad templates: 12 major + 12 minor."""
    templates, names = [], []
    for root in range(12):
        major = np.zeros(12)
        major[[root, (root + 4) % 12, (root + 7) % 12]] = 1
        templates.append(major)
        names.append(f"{PITCHES[root]}:maj")
    for root in range(12):
        minor = np.zeros(12)
        minor[[root, (root + 3) % 12, (root + 7) % 12]] = 1
        templates.append(minor)
        names.append(f"{PITCHES[root]}:min")
    return np.asarray(templates), names


def estimate_chords(chroma: np.ndarray) -> list[str]:
    """Frame-wise chord labels by template matching on a (12, T) chromagram.

    Deliberately simple: the PDF asks for a chord-transition graph, not a
    state-of-the-art chord recogniser, and an HMM here would be unjustified
    complexity for a graph whose only role is edge counts.
    """
    if chroma.size == 0:
        return []
    templates, names = _chord_templates()
    norms = np.linalg.norm(chroma, axis=0, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    scores = templates @ (chroma / norms)
    return [names[i] for i in scores.argmax(axis=0)]


def collapse_repeats(sequence: list[str]) -> list[str]:
    """Drop consecutive duplicates so transitions are real chord changes."""
    out: list[str] = []
    for item in sequence:
        if not out or out[-1] != item:
            out.append(item)
    return out


def build_chord_graph(
    chords: list[str],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Chord-transition graph -> (node_names, edge_index [2, E], counts [E]).

    Nodes are the unique chords observed; a directed edge i->j is weighted by
    how often that transition occurs, exactly as S3 specifies.
    """
    seq = collapse_repeats(chords)
    nodes = sorted(set(seq))
    if len(seq) < 2:
        return nodes, np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    index = {c: i for i, c in enumerate(nodes)}
    counts: dict[tuple[int, int], int] = {}
    for a, b in zip(seq, seq[1:]):
        key = (index[a], index[b])
        counts[key] = counts.get(key, 0) + 1

    src = [k[0] for k in counts]
    dst = [k[1] for k in counts]
    return (
        nodes,
        np.asarray([src, dst], dtype=np.int64),
        np.asarray(list(counts.values()), dtype=np.float32),
    )


def graph_stats(edge_index: np.ndarray, n_nodes: int) -> dict:
    """Summary used to sanity-check a built graph."""
    n_edges = int(edge_index.shape[1])
    return {
        "n_nodes": int(n_nodes),
        "n_edges": n_edges,
        "avg_degree": round(n_edges / n_nodes, 3) if n_nodes else 0.0,
        "density": round(n_edges / (n_nodes * (n_nodes - 1)), 4)
        if n_nodes > 1
        else 0.0,
    }
