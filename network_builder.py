"""network_builder.py

Construct passing networks per (match, team, window) and compute graph metrics.

Primary APIs
------------
1. build_team_window_graph(df_slice) -> nx.DiGraph
   Given a DataFrame of passes for a single team within a single time window,
   returns a directed weighted graph where edge weight = count of passes
   (passer → receiver).

2. compute_metrics(passes_windowed_df) -> pd.DataFrame
   Accepts the full DataFrame produced by `windowing.assign_windows` and
   returns one row per (matchId, teamId, windowStart, windowEnd) containing
   requested metrics.
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def build_team_window_graph(df_slice: pd.DataFrame) -> nx.DiGraph:
    """Return directed weighted graph of passes in *df_slice*.

    *df_slice* must have columns [passerId, receiverId].
    """
    g = nx.DiGraph()
    # Count edge weights quickly with Counter on tuples
    weights = Counter(zip(df_slice["passerId"], df_slice["receiverId"]))
    for (u, v), w in weights.items():
        g.add_edge(u, v, weight=w)
    return g


def _shannon_entropy(weights: list[int]) -> float:
    if not weights:
        return 0.0
    total = float(sum(weights))
    probs = [w / total for w in weights]
    ent = -sum(p * math.log(p, 2) for p in probs)
    max_ent = math.log(len(weights), 2) if len(weights) > 1 else 1
    return ent / max_ent  # normalised 0-1


def _centralisation(g: nx.DiGraph) -> float:
    if g.number_of_nodes() < 2:
        return 0.0
    degs = dict(g.degree(weight="weight"))
    max_deg = max(degs.values())
    mean_deg = np.mean(list(degs.values()))
    max_possible = (g.number_of_nodes() - 1) * max_deg / max_deg  # simplifies to V-1
    return (max_deg - mean_deg) / max_possible if max_possible else 0.0


def _top_eigen_centrality(g: nx.DiGraph, k: int = 3) -> float:
    if g.number_of_nodes() == 0 or g.number_of_edges() == 0:
        return 0.0
    try:
        # power-iteration variant avoids scipy dependency
        ec = nx.eigenvector_centrality(g, weight="weight", max_iter=1000)
    except (nx.NetworkXException, np.linalg.LinAlgError):
        return 0.0
    top_vals = sorted(ec.values(), reverse=True)[:k]
    return float(np.mean(top_vals))

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_metrics(passes_windowed_df: pd.DataFrame) -> pd.DataFrame:
    """Compute graph metrics for each (match, team, window)."""
    group_cols = ["matchId", "teamId", "windowStart", "windowEnd"]
    records = []
    for keys, grp in tqdm(passes_windowed_df.groupby(group_cols, sort=False), desc="Graphs"):
        g = build_team_window_graph(grp)
        num_nodes = g.number_of_nodes()
        num_edges = g.number_of_edges()
        density = (num_edges / (num_nodes * (num_nodes - 1))) if num_nodes > 1 else 0.0
        clustering = nx.average_clustering(g.to_undirected(), weight="weight") if num_nodes > 1 else 0.0
        entropy = _shannon_entropy([d["weight"] for _, _, d in g.edges(data=True)])
        centralisation = _centralisation(g)
        top_centrality = _top_eigen_centrality(g)
        window_secs = keys[3] - keys[2]
        tempo = len(grp) / (window_secs / 60) if window_secs else 0.0

        records.append(
            dict(
                matchId=keys[0],
                teamId=keys[1],
                windowStart=keys[2],
                windowEnd=keys[3],
                numNodes=num_nodes,
                numEdges=num_edges,
                density=density,
                clustering=clustering,
                entropy=entropy,
                centralisation=centralisation,
                topCentrality=top_centrality,
                tempo=tempo,
            )
        )
    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Compute passing-network metrics from windowed passes parquet")
    ap.add_argument("passes_windowed", type=Path, help="passes_windowed.parquet")
    ap.add_argument("--out", type=Path, default=Path("metrics.parquet"))
    ns = ap.parse_args()

    df_pw = pd.read_parquet(ns.passes_windowed)
    df_metrics = compute_metrics(df_pw)
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    df_metrics.to_parquet(ns.out, index=False)
    print(f"Wrote {len(df_metrics)} rows of metrics to {ns.out}")
