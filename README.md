# FIFA 2022 Passing Network Analysis

End-to-end pipeline to extract completed passes from FIFA World Cup 2022 event data, construct team-window passing graphs, compute network metrics, and label tactical pressure.

## Quick Start

```bash
# Setup environment
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# Run full pipeline (produces outputs/features_master.csv)
python src/parse_passes.py --events_dir "Data/Event Data" --metadata_dir Data/Metadata --out passes.parquet
python src/windowing.py passes.parquet Data/Metadata --out passes_windowed.parquet
python src/network_builder.py passes_windowed.parquet --out metrics.parquet
python src/pressure_label.py metrics.parquet "Data/Event Data" Data/Metadata --out metrics_pressure.parquet
python src/feature_table.py metrics_pressure.parquet passes_windowed.parquet Data/Metadata --out features.parquet
```

## Output

The final dataset (`outputs/features_master.csv`) contains **919 rows** representing team-window combinations with:

- **Graph metrics**: density, clustering, entropy, centralisation, tempo
- **Pressure labels**: HIGH/MEDIUM/LOW based on score, time, and competition round
- **Match context**: opponent, stadium, round, score difference
- **Volume features**: pass counts, unique passers/receivers

## Dataset Summary

- **64 matches** from FIFA World Cup 2022
- **55,727 completed passes** across all matches
- **Score differences** range from -7 to +7 goals
- **Pressure distribution**: 42% MEDIUM, 30% HIGH, 28% LOW

## Directory Structure

```
├── src/                    # Pipeline modules
│   ├── parse_passes.py     # Extract passes from event JSON
│   ├── windowing.py        # Assign 15-minute time windows
│   ├── network_builder.py  # Build graphs and compute metrics
│   ├── pressure_label.py   # Label pressure based on game state
│   └── feature_table.py    # Combine all features
├── outputs/                # Final results
│   ├── features_master.csv # Main dataset for analysis
│   └── features_master.parquet
├── Data/                   # Raw FIFA data (not included)
└── requirements.txt        # Python dependencies
```

## Citation

If you use this pipeline or dataset, please cite:

```
[Your paper citation here]
```

## License

[Your license here]
