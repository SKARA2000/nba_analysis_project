# NBA performance estimation

Start with [the dataset exploration notebook](notebooks/01_dataset_exploration.ipynb), which follows [the project proposal](NBA_Project_Proposal_Revised.md).

## Run

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m jupyterlab
```

Open `notebooks/01_dataset_exploration.ipynb`, select the environment's Python kernel, and run all cells. Internet access is needed for the first download. The notebook caches the six season Parquet files under `data/raw/<revision>/` and records their URLs and checksums. Subsequent runs can use that cache offline.

## What is included

- A dictionary for all 26 documented fields and checks against the observed Parquet schemas.
- Missingness, duplicate keys, box-score consistency, season coverage, and matchup audits.
- Training-period distributions and player-history plots.
- Strictly lagged pregame features and a validation-only comparison of the proposal's historical-average baselines.
- Chronological train/validation/test boundaries, with the partial latest season reserved for monitoring.

## Initial source inspection

The [Hugging Face dataset](https://huggingface.co/datasets/cbratkovics/nba-game-logs) was inspected on October 28, 2026. The notebook pins revision `d70d239b7b82e3e4071d2c7caab2d0e39db70389` and downloads only `game_logs/game_logs_*.parquet`; the repository's general viewer currently fails with a schema mismatch.

The source documents regular-season player appearances, omits DNPs, and reports seven missing games in 2024–25. Season files cover 2021–22 through a partial 2026–27. Consequently, this initial analysis targets scoring conditional on an appearance; predicting availability in a team's next game requires additional data.

**Execution limitation:** the real-data cells have not been executed here. Python could not resolve the external download hostname, and the local environment lacks `pyarrow`. Source documentation was inspected, notebook code was syntax-checked, and downstream analysis was checked with a temporary synthetic fixture. Those checks are not empirical dataset findings; real statistics and baseline scores will appear after running the notebook with the dependencies and network access.

## HTTPS certificate errors

The downloader supplements Python's default trust store with certifi's CA bundle. If an existing kernel reports `CERTIFICATE_VERIFY_FAILED`, run `%pip install --upgrade certifi` in a notebook cell, then rerun the download cell. Certificate and hostname verification remain enabled. Download errors also include the underlying cause rather than only a generic network message.
