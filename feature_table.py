"""feature_table.py

Build master feature table combining graph metrics, pressure labels, pass-volume
counts and match metadata.

Usage
-----
python feature_table.py \
  metrics_pressure.parquet passes_windowed.parquet Data/Metadata \
  --out features.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _aggregate_pass_volume(passes_df: pd.DataFrame) -> pd.DataFrame:
    """Return per-window pass volume features."""
    keys = ["matchId", "teamId", "windowStart", "windowEnd"]

    def _num_edges_fast(df: pd.DataFrame) -> int:
        return pd.MultiIndex.from_frame(df[["passerId", "receiverId"]]).nunique()

    agg_df = (
        passes_df.groupby(keys, as_index=False)
        .agg(
            numPasses=("passerId", "size"),
            numPassers=("passerId", "nunique"),
            numReceivers=("receiverId", "nunique"),
        )
    )
    # compute numEdges separately
    edge_counts = (
        passes_df.groupby(keys)
        .apply(_num_edges_fast)
        .rename("numEdges")
        .reset_index()
    )
    agg_df = agg_df.merge(edge_counts, on=keys, how="left")
    return agg_df


def _load_metadata(meta_dir: Path) -> Dict[str, Dict]:
    """Return mapping matchId -> metadata dict with opponent/team info."""
    mapping: Dict[str, Dict] = {}
    for p in meta_dir.glob("*.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))[0]
        except Exception:
            continue
        mid = str(obj["id"])
        mapping[mid] = {
            "matchDate": obj.get("matchDate"),
            "stadiumName": obj.get("stadiumName"),
            "homeTeamId": int(obj.get("homeTeamId") or obj.get("homeTeam", {}).get("id", 0)),
            "awayTeamId": int(obj.get("awayTeamId") or obj.get("awayTeam", {}).get("id", 0)),
            "round": "group" if "group" in str(obj.get("competitionStage", "")).lower() else "knockout",
        }
    return mapping


def build_feature_table(metrics_path: Path, passes_path: Path, meta_dir: Path) -> pd.DataFrame:
    """Return combined feature table DataFrame."""
    metrics = pd.read_parquet(metrics_path)
    passes = pd.read_parquet(passes_path)

    # ensure matchId is string for consistent merges
    metrics["matchId"] = metrics["matchId"].astype(str)
    passes["matchId"] = passes["matchId"].astype(str)

    volume_df = _aggregate_pass_volume(passes)
    df = metrics.merge(volume_df, on=["matchId", "teamId", "windowStart", "windowEnd"], how="left")

    meta_map = _load_metadata(meta_dir)
    meta_df = pd.DataFrame.from_dict(meta_map, orient="index").reset_index(names="matchId")
    meta_df["matchId"] = meta_df["matchId"].astype(str)
    df = df.merge(meta_df, on="matchId", how="left")

    # opponent id per row
    def _opp(row):
        if row["teamId"] == row.get("homeTeamId"):
            return row.get("awayTeamId")
        if row["teamId"] == row.get("awayTeamId"):
            return row.get("homeTeamId")
        return None

    df["opponentTeamId"] = df.apply(_opp, axis=1)

    # reorder columns roughly
    desired_order = [
        "matchId",
        "teamId",
        "opponentTeamId",
        "windowStart",
        "windowEnd",
        "minute",
        "pressure",
        "scoreDiff",
        "round",
    ]
    first_cols = [c for c in desired_order if c in df.columns]
    other_cols = [c for c in df.columns if c not in first_cols]
    df = df[first_cols + other_cols]
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build master feature table for passing network analysis")
    ap.add_argument("metrics_pressure", type=Path, help="metrics_pressure.parquet from pressure_label.py")
    ap.add_argument("passes_windowed", type=Path, help="passes_windowed.parquet from windowing.py")
    ap.add_argument("metadata_dir", type=Path, help="directory with match Metadata JSON files")
    ap.add_argument("--out", type=Path, default=Path("features.parquet"))
    ns = ap.parse_args()

    feat_df = build_feature_table(ns.metrics_pressure, ns.passes_windowed, ns.metadata_dir)
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_parquet(ns.out, index=False)
    # also write CSV for convenience
    csv_path = ns.out.with_suffix(".csv")
    feat_df.to_csv(csv_path, index=False)
    print(f"Wrote {len(feat_df)} rows to {ns.out} and {csv_path}")
