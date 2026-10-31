# NBA performance estimation

This project explores per-game NBA player statistics and prepares scoring forecasts following [the project proposal](NBA_Project_Proposal_Revised.md).

## Environment setup

Run these commands from the project root. Create the environment only if `.venv` does not already exist.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Exploration script

[scripts/explore_data.py](scripts/explore_data.py) inspects locally cached Parquet files without executing a notebook. It prints field names, types, missing values, season coverage, player/game key checks, and individual game records for an example player. It also calculates a season summary from those records to illustrate the difference between game-level observations and season aggregates.

```sh
.venv/bin/python scripts/explore_data.py
.venv/bin/python scripts/explore_data.py --season 2024-25 --player "LeBron James" --rows 8
.venv/bin/python scripts/explore_data.py --season 2024-25 --player 2544
.venv/bin/python scripts/explore_data.py --data-dir /path/to/season-files
.venv/bin/python scripts/explore_data.py --help
```

By default, the script reads `data/raw/d70d239b7b82e3e4071d2c7caab2d0e39db70389/game_logs_*.parquet`. `--season` limits the inspection to one season; `--player` accepts a player ID or case-insensitive name substring; `--rows` controls the number of example game records. Without `--player`, the script chooses a player-season with the most observed games. It does not download data or modify files. If the cache is empty, obtain the season files using a notebook's download section or supply a populated directory with `--data-dir`.

## Notebooks

[notebooks/dataset_exploration.ipynb](notebooks/dataset_exploration.ipynb) is the focused data exploration notebook. It includes dataset loading, a dictionary for the 26 documented fields, missingness and box-score consistency checks, season coverage, and matchup audits. Section 6 follows [plans/data_exploration.md](plans/data_exploration.md): player season counts, participation masks, gaps, retention, first observations, plots, and implications for future train/test splits. First-observed players are not automatically confirmed NBA rookies.

Sections 7–11 add the plan's basic multiple linear regression (MLR) setup: lagged player-history features; four expanding-history experiments testing 2022–23 through 2025–26; player-indexed split audits; separate PPM, minutes, and direct-points models; a factorized points comparison; and error/coefficient diagnostics. Each experiment uses an approximate 80/20 shared-date validation split within its historical training seasons, followed by the specified whole-season test. Individual player ratios are reported rather than forced to 80/20. Starting/bench status is omitted because the dataset does not supply it.

The MLR comparison uses four nested feature sets: `core` (12 inputs), `role_defense` (18), `shooting` (24), and `full_history` (72), before missingness indicators. Additional predictors are strictly lagged assists, rebounds, steals, blocks, turnovers, fouls, shooting counts, offensive/defensive rebounds, and plus/minus. Validation compares the same player-games across sets and reports training/validation error, design rank, condition number, correlated predictors, and coefficient sensitivity to assess when extra features stop helping OLS. Model and prediction dictionaries use `(experiment, feature_set)` keys.

The new MLR cells are unexecuted. Install the updated requirements, open the notebook with the command below, and review sections 7–9 before running the validation cells. Test execution is gated by `RUN_TEST_EXPERIMENTS = False`; first fill `SELECTED_FEATURE_SETS` with one feature set per experiment based on validation, then set the flag to `True` when ready to evaluate the fixed design. The later partial 2026–27 season is excluded from these experiments. Comparing factorized and direct predictions does not by itself establish low conditional covariance; residual dependence is reported as an exploratory diagnostic.

Open it with:

```sh
.venv/bin/python -m jupyterlab notebooks/dataset_exploration.ipynb
```

[notebooks/01_dataset_exploration.ipynb](notebooks/01_dataset_exploration.ipynb) is the earlier starter notebook. It contains the loading and quality checks plus a provisional season-based split, training-period outcome plots, lagged pregame features, and a validation-only comparison of historical-average baselines. Its section 6 is a placeholder; use `dataset_exploration.ipynb` for the completed player-persistence analysis. Reconsider the starter's split using that analysis before developing further models.

Open it with:

```sh
.venv/bin/python -m jupyterlab notebooks/01_dataset_exploration.ipynb
```

These commands open the notebooks without executing them. Select the `.venv` Python kernel, then run the desired cells in order when ready. Internet access is needed for an uncached first download. Both notebooks cache the six season Parquet files under `data/raw/<revision>/` and record their URLs and checksums; subsequent runs can reuse the cache offline.

## Initial source inspection

The [Hugging Face dataset](https://huggingface.co/datasets/cbratkovics/nba-game-logs) was inspected on October 28, 2026. The notebook pins revision `d70d239b7b82e3e4071d2c7caab2d0e39db70389` and downloads only `game_logs/game_logs_*.parquet`; the repository's general viewer currently fails with a schema mismatch.

The source documents regular-season player appearances, omits DNPs, and reports seven missing games in 2024–25. Season files cover 2021–22 through a partial 2026–27. Consequently, this initial analysis targets scoring conditional on an appearance; predicting availability in a team's next game requires additional data.

**Verification status:** section 6 of `dataset_exploration.ipynb` has saved outputs from the cached real dataset. The standalone script has also been checked against all cached seasons and a specific player-season. These checks confirm per-game player records with no duplicate `(player_id, game_id)` keys in the cached snapshot. They do not validate every modeling result in the earlier starter notebook.

## HTTPS certificate errors

The downloader supplements Python's default trust store with certifi's CA bundle. If an existing kernel reports `CERTIFICATE_VERIFY_FAILED`, run `%pip install --upgrade certifi` in a notebook cell, then rerun the download cell. Certificate and hostname verification remain enabled. Download errors also include the underlying cause rather than only a generic network message.
