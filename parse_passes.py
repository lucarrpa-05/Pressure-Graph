"""parse_passes.py
Phase-A utility for extracting completed passes from a single match’s event file.

Usage (CLI):
    python parse_passes.py --events_dir Data/Event\ Data --metadata_dir Data/Metadata --rosters_dir Data/Rosters \
                           --competitions Data/competitions.csv --out passes.parquet

The core logic is exposed via `extract_passes`, enabling reuse in notebooks / pipelines.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd
import ijson  # streaming JSON parser
from tqdm import tqdm

FIELDS = [
    "matchId",
    "period",
    "eventTime",
    "passerId",
    "receiverId",
    "teamId",
]

def _stream_completed_passes(event_file: Path) -> List[dict]:
    """Yield dictionaries with required fields for each completed pass in *event_file*."""
    records: List[dict] = []
    with event_file.open("rb") as fh:
        # iterate over the list elements (each is an event dict)
        for obj in ijson.items(fh, "item"):
            pe = obj.get("possessionEvents") or {}
            if pe.get("possessionEventType") != "PA" or pe.get("passOutcomeType") != "C":
                continue  # not a completed pass

            # Fallback when receiverId missing: use targetPlayerId
            receiver_id = pe.get("receiverPlayerId") or pe.get("targetPlayerId")

            rec = {
                "matchId": obj.get("gameId"),
                "period": obj.get("gameEvents", {}).get("period"),
                "eventTime": obj.get("eventTime"),
                "passerId": pe.get("passerPlayerId"),
                "receiverId": receiver_id,
                "teamId": obj.get("gameEvents", {}).get("teamId"),
            }
            # Ensure all required keys present
            if None not in rec.values():
                records.append(rec)
    return records


def extract_passes(event_path: str | Path, metadata_path: str | Path | None = None, roster_path: str | Path | None = None) -> pd.DataFrame:
    """Return DataFrame of completed passes for one match.

    Parameters
    ----------
    event_path : str | Path
        Path to the match event JSON file.
    metadata_path : str | Path | None, optional
        Not used in current extraction but reserved for future enhancements (e.g., half offsets).
    roster_path : str | Path | None, optional
        Not used for extraction, kept for interface compatibility.
    """
    event_path = Path(event_path)
    passes = _stream_completed_passes(event_path)
    return pd.DataFrame(passes, columns=FIELDS)


def _iter_match_ids(competitions_csv: Path) -> List[str]:
    comp_df = pd.read_csv(competitions_csv)
    # "games" column contains a stringified list-of-dicts → eval safe via json.loads after replace
    games_raw = comp_df.loc[0, "games"]
    if isinstance(games_raw, str):
        game_dicts = json.loads(games_raw.replace("'", '"'))
        return [item["id"] for item in game_dicts]
    raise ValueError("Unexpected competitions.csv format")


def cli():
    p = argparse.ArgumentParser(description="Extract completed passes for all matches listed in competitions.csv")
    p.add_argument("--events_dir", type=Path, default=Path("Data/Event Data"))
    p.add_argument("--metadata_dir", type=Path, default=Path("Data/Metadata"))
    p.add_argument("--rosters_dir", type=Path, default=Path("Data/Rosters"))
    p.add_argument("--competitions", type=Path, default=Path("Data/competitions.csv"))
    p.add_argument("--out", type=Path, default=Path("passes.parquet"))
    args = p.parse_args()

    match_ids = _iter_match_ids(args.competitions)
    all_frames: List[pd.DataFrame] = []

    for mid in tqdm(match_ids, desc="Matches"):
        event_path = args.events_dir / f"{mid}.json"
        if not event_path.exists():
            print(f"Warning: event file {event_path} missing; skipping")
            continue
        df = extract_passes(event_path)
        all_frames.append(df)

    full_df = pd.concat(all_frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_parquet(args.out, index=False)
    print(f"Wrote {len(full_df)} pass records to {args.out}")


if __name__ == "__main__":
    cli()
