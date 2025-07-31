"""windowing.py
Assign passes (or any event rows) into time windows within a match.

Core entry point
----------------
assign_windows(passes_df, metadata_dir, width=900, stride=900) -> DataFrame
    Returns *a copy* of `passes_df` with added columns:
        - matchSeconds: time from kick-off (float, seconds)
        - windowStart: window start time (seconds from kick-off)
        - windowEnd:   window end time (seconds from kick-off)

The function relies on Metadata JSON files produced by PFF. Only first- and
second-half offsets are handled (extra-time is ignored for 2022 World Cup).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_metadata(metadata_dir: Path) -> Dict[str, dict]:
    """Return dict mapping matchId → metadata dict."""
    meta = {}
    for p in metadata_dir.glob("*.json"):
        with p.open("r", encoding="utf-8") as fh:
            obj = json.load(fh)[0]  # each file is a 1-element list
            meta[obj["id"]] = obj
    return meta


def _compute_match_seconds(row: pd.Series, meta: dict) -> float:
    """Convert eventTime into seconds from kick-off (0 at 00:00)."""
    period = int(row["period"])
    evt = float(row["eventTime"])
    sp1 = meta.get("startPeriod1") or 0.0
    sp2 = meta.get("startPeriod2")
    # Fallback for missing second-half start: use halftime length if provided
    if sp2 is None:
        halftime = float(meta.get("halfPeriod") or 900)  # default 15-min break
        sp2 = float(sp1) + 45 * 60 + halftime
    if period == 1:
        return evt - float(sp1)
    if period == 2:
        secs_first_half = 45 * 60
        return secs_first_half + (evt - float(sp2))
    # Extra time / other periods – use raw timestamp offset by sp1
    return evt - float(sp1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assign_windows(
    passes_df: pd.DataFrame,
    metadata_dir: str | Path,
    *,
    width: int = 900,
    stride: int = 900,
) -> pd.DataFrame:
    """Add window assignment columns to *passes_df*.

    Parameters
    ----------
    passes_df : DataFrame
        Output from `parse_passes.extract_passes` (or concatenation thereof).
    metadata_dir : Path-like
        Directory containing per-match metadata JSON files.
    width : int, default 900
        Window width in seconds. 900 s = 15 min.
    stride : int, default 900
        Step size between window starts. For non-overlapping bins, set equal to
        *width*. For rolling windows, set *stride* < *width* (e.g., stride=300).
    """
    if stride <= 0 or width <= 0:
        raise ValueError("`width` and `stride` must be positive seconds")

    metadata_dir = Path(metadata_dir)
    meta_map = _load_metadata(metadata_dir)

    # Compute matchSeconds for each row.
    def _ms(row):
        mid = str(row["matchId"])
        return _compute_match_seconds(row, meta_map[mid])

    passes_df = passes_df.copy()
    passes_df["matchSeconds"] = passes_df.apply(_ms, axis=1)

    # Window index based on stride
    passes_df["_win_idx"] = (passes_df["matchSeconds"] // stride).astype(int)
    passes_df["windowStart"] = passes_df["_win_idx"] * stride
    passes_df["windowEnd"] = passes_df["windowStart"] + width
    passes_df.drop(columns="_win_idx", inplace=True)

    return passes_df


# CLI utility for quick testing ------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Assign time windows to passes parquet")
    ap.add_argument("passes", type=Path, help="passes.parquet file")
    ap.add_argument("metadata_dir", type=Path, default=Path("Data/Metadata"))
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--stride", type=int, default=900)
    ap.add_argument("--out", type=Path, default=Path("passes_windowed.parquet"))
    ns = ap.parse_args()

    df_passes = pd.read_parquet(ns.passes)
    df_win = assign_windows(df_passes, ns.metadata_dir, width=ns.width, stride=ns.stride)
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    df_win.to_parquet(ns.out, index=False)
    print(f"Wrote {len(df_win)} rows with window assignment to {ns.out}")
