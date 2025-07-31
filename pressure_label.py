"""pressure_label.py

Assign categorical pressure labels to each team-window based on score, time,
and competition round, according to the taxonomy provided in the project
specification.

Usage (CLI)
-----------
python pressure_label.py metrics.parquet Data/Event\ Data Data/Metadata --out metrics_labeled.parquet

Dependencies: pandas, tqdm (already in requirements).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm
import ijson

# ---------------------------------------------------------------------------
# Helpers to extract goals & round information
# ---------------------------------------------------------------------------

def _collect_goals(events_dir: Path, metadata_dir: Path) -> pd.DataFrame:
    meta_map = {}
    for mp in metadata_dir.glob("*.json"):
        try:
            obj = json.loads(mp.read_text(encoding="utf-8"))[0]
            meta_map[str(obj["id"])] = {
                "home": int(obj.get("homeTeamId") or obj.get("homeTeam", {}).get("id", 0)),
                "away": int(obj.get("awayTeamId") or obj.get("awayTeam", {}).get("id", 0)),
            }
        except Exception:
            pass

    rows = []
    for f in tqdm(list(events_dir.glob("*.json")), desc="Streaming goals"):
        match_id = f.stem
        with f.open("r", encoding="utf-8") as fh:
            first = fh.read(1); fh.seek(0)
            events_iter = ijson.items(fh, "item") if first == "[" else (json.loads(l) for l in fh if l.strip())

            for ev in events_iter:
                period = int(ev.get("period", 1))

                # ----- gameEvents path (OUT + H/A) -----
                ge = ev.get("gameEvents")
                if isinstance(ge, dict):
                    if ge.get("gameEventType") == "OUT" and ge.get("outType") in ("H", "A"):
                        side = "home" if ge["outType"] == "H" else "away"
                        tid = meta_map.get(match_id, {}).get(side)
                        if tid:
                            rows.append(dict(
                                matchId=match_id,
                                teamId=tid,
                                period=period,
                                matchSeconds=float(ge.get("startGameClock", ge.get("gameClock", 0.0)))
                            ))

                # ----- possessionEvents path (SH + shotOutcomeType == G) -----
                pe = ev.get("possessionEvents") or {}
                pe_iter = pe.values() if isinstance(pe, dict) else pe if isinstance(pe, list) else []
                for sub in pe_iter:
                    if (
                        isinstance(sub, dict)
                        and str(sub.get("possessionEventType", "")).upper() == "SH"
                        and str(sub.get("shotOutcomeType", "")).upper().startswith("G")
                    ):
                        rows.append(dict(
                            matchId=match_id,
                            teamId=sub.get("teamId", ev.get("teamId")),
                            period=period,
                            matchSeconds=float(sub.get("gameClock", ev.get("gameClock", 0.0)))
                        ))

    return pd.DataFrame(rows)


def _event_time_to_match_seconds(row: pd.Series) -> float:
    """Approximate seconds from kickoff given period and eventTime.

    For simplicity assume:
    - first half kickoff at 0
    - second half kickoff at 45*60 = 2700
    Extra-time not handled rigorously.
    """
    if row["period"] == 1:
        return row["eventTime"]
    if row["period"] == 2:
        return 45 * 60 + row["eventTime"]
    return row["eventTime"]


def _tag_round(meta_dir: Path) -> Dict[str, str]:
    """Return mapping matchId -> round ('group'/'knockout')."""
    mapping: Dict[str, str] = {}
    for p in meta_dir.glob("*.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))[0]
        except Exception:
            continue
        stage = str(obj.get("competitionStage", "")).lower()
        match_id = str(obj["id"])
        if "group" in stage:
            mapping[match_id] = "group"
        else:
            mapping[match_id] = "knockout"
    return mapping

# ---------------------------------------------------------------------------
# Core labelling logic
# ---------------------------------------------------------------------------

def _pressure_label(row: pd.Series) -> str:
    """Return HIGH/MEDIUM/LOW for a row with needed columns."""
    # Unpack
    round_ = row["round"]
    score = row["scoreDiff"]
    minute = row["minute"]

    # Leading by 2+ goals
    if score >= 2:
        return "LOW"
    # Early group-stage (first 30 mins)
    if round_ == "group" and minute < 30:
        return "LOW"
    # Up 1 goal with >30 mins left
    if score == 1 and minute < 60:
        return "LOW"

    # High pressure conditions
    if round_ == "knockout" and score < 0:
        return "HIGH"
    if minute >= 75:
        return "HIGH"
    if score == 0 and minute >= 75:
        return "HIGH"

    # Medium cases
    if round_ == "knockout":
        return "MEDIUM"
    if score == 0:  # tied group not final15
        return "MEDIUM"

    # default
    return "MEDIUM"


def label_pressure(
    metrics_df: pd.DataFrame,
    events_dir: str | Path,
    metadata_dir: str | Path,
) -> pd.DataFrame:
    """Return metrics_df merged with pressure variables."""
    events_dir = Path(events_dir)
    metadata_dir = Path(metadata_dir)

    goals_df = _collect_goals(events_dir, metadata_dir)
    if goals_df.empty:
        raise RuntimeError("No goals found – check event data directory")

    goals_df["matchSeconds"] = goals_df["matchSeconds"].astype(float)

    # cumulative goals per team up to each time
    metrics_df = metrics_df.copy()
    metrics_df["minute"] = (metrics_df["windowStart"] + metrics_df["windowEnd"]) / 120

    score_diff_list = []
    for (_, team, start, _), grp in metrics_df.groupby(["matchId", "teamId", "windowStart", "windowEnd"]):
        match_id = grp.iloc[0]["matchId"]
        team_id = team
        window_mid_sec = (start + grp.iloc[0]["windowEnd"]) / 2
        goals_for = goals_df[(goals_df["matchId"] == match_id) & (goals_df["teamId"] == team_id) & (goals_df["matchSeconds"] <= window_mid_sec)].shape[0]
        goals_against = goals_df[(goals_df["matchId"] == match_id) & (goals_df["teamId"] != team_id) & (goals_df["matchSeconds"] <= window_mid_sec)].shape[0]
        score_diff_list.append(goals_for - goals_against)

    metrics_df["scoreDiff"] = score_diff_list

    round_map = _tag_round(metadata_dir)
    metrics_df["round"] = metrics_df["matchId"].map(round_map).fillna("group")

    metrics_df["pressure"] = metrics_df.apply(_pressure_label, axis=1)

    return metrics_df[
        [
            "matchId",
            "teamId",
            "windowStart",
            "windowEnd",
            "scoreDiff",
            "round",
            "minute",
            "pressure",
        ]
    ]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Label pressure for each team-window and merge with metrics")
    ap.add_argument("metrics", type=Path, help="metrics.parquet file")
    ap.add_argument("events_dir", type=Path, help="directory with Event Data JSONs")
    ap.add_argument("metadata_dir", type=Path, help="directory with Metadata JSONs")
    ap.add_argument("--out", type=Path, default=Path("metrics_pressure.parquet"))
    ns = ap.parse_args()

    df_metrics = pd.read_parquet(ns.metrics)
    df_press = label_pressure(df_metrics, ns.events_dir, ns.metadata_dir)
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    df_press.to_parquet(ns.out, index=False)
    print(f"Wrote {len(df_press)} rows with pressure labels to {ns.out}")
