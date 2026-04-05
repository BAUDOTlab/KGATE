"""Generate a deterministic small heterogeneous graph for test pipelines.

The generated dataset contains:
- 500 edges
- 5 relation types
- 3 node types
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
GRAPH_PATH = DATA_DIR / "synthetic_test_graph.csv"
METADATA_PATH = DATA_DIR / "synthetic_test_metadata.csv"

SEED = 42
TOTAL_EDGES = 500
NODES_PER_TYPE = 20

NODE_TYPES = ("gene", "disease", "compound")
RELATIONS = (
    ("gene", "associated_with", "disease"),
    ("compound", "treats", "disease"),
    ("gene", "interacts_with", "gene"),
    ("disease", "cooccurs_with", "disease"),
    ("compound", "targets", "gene"),
)


def make_nodes() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for node_type in NODE_TYPES:
        for idx in range(NODES_PER_TYPE):
            rows.append({"id": f"{node_type}_{idx:02d}", "type": node_type})
    return pd.DataFrame(rows)


def sample_edges(metadata: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    nodes_by_type = {
        node_type: metadata.loc[metadata["type"] == node_type, "id"].to_numpy()
        for node_type in NODE_TYPES
    }

    relation_counts = [100, 100, 100, 100, 100]
    rows: list[dict[str, str]] = []

    for (source_type, relation, target_type), count in zip(
        RELATIONS, relation_counts, strict=True
    ):
        heads = rng.choice(nodes_by_type[source_type], size=count, replace=True)
        tails = rng.choice(nodes_by_type[target_type], size=count, replace=True)
        for head, tail in zip(heads, tails, strict=True):
            rows.append({"head": head, "tail": tail, "edge": relation})

    graph_df = pd.DataFrame(rows)
    graph_df = graph_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    if len(graph_df) != TOTAL_EDGES:
        raise RuntimeError(f"Expected {TOTAL_EDGES} edges, got {len(graph_df)}")

    return graph_df


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    metadata_df = make_nodes()
    graph_df = sample_edges(metadata_df)

    metadata_df.to_csv(METADATA_PATH, index=False)
    graph_df.to_csv(GRAPH_PATH, index=False)

    print(f"Wrote graph to {GRAPH_PATH}")
    print(f"Wrote metadata to {METADATA_PATH}")
    print(
        "Summary: "
        f"edges={len(graph_df)}, "
        f"relations={graph_df['edge'].nunique()}, "
        f"node_types={metadata_df['type'].nunique()}"
    )


if __name__ == "__main__":
    main()
