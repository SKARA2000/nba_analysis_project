"""Inspect locally cached NBA Parquet files without executing a notebook.

Run from any directory with the project's Python environment:
    .venv/bin/python scripts/explore_data.py
    .venv/bin/python scripts/explore_data.py --season 2024-25 --player "LeBron James"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "raw" / "d70d239b7b82e3e4071d2c7caab2d0e39db70389"


def show(title: str, value: pd.DataFrame | pd.Series) -> None:
    print(f"\n{title}\n{'-' * len(title)}")
    print(value.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA,
                        help="Directory containing game_logs_*.parquet (no downloads).")
    parser.add_argument("--season", help="Inspect one season, for example 2024-25.")
    parser.add_argument("--player", help="Player ID or case-insensitive name substring.")
    parser.add_argument("--rows", type=int, default=10, help="Example game rows to print (default: 10).")
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("--rows must be positive.")

    files = sorted(args.data_dir.glob("game_logs_*.parquet"))
    if not files:
        parser.error(f"No game_logs_*.parquet files in {args.data_dir}. Supply --data-dir.")
    frames = []
    for path in files:
        part = pd.read_parquet(path, engine="pyarrow")
        if args.season:
            part = part.loc[part["season"].eq(args.season)]
        if not part.empty:
            frames.append(part)
    if not frames:
        parser.error(f"No records for season {args.season!r}.")
    df = pd.concat(frames, ignore_index=True)
    required = {"player_id", "player_name", "season", "game_id", "game_date", "pts", "minutes"}
    if missing := required - set(df):
        parser.error(f"Expected fields missing: {sorted(missing)}")
    for column in ["player_id", "game_id", "season"]:
        df[column] = df[column].astype("string")
    df["game_date"] = pd.to_datetime(df["game_date"], errors="raise", utc=True)
    df = df.sort_values(["player_id", "game_date", "game_id"])

    print(f"Source: {args.data_dir.resolve()}")
    print(f"Read {len(df):,} rows, {len(df.columns)} fields from {len(frames)} nonempty season files.")
    print("This script only reads local data and prints an inspection report.")
    show("Fields, types, and missing values", pd.DataFrame({
        "dtype": df.dtypes.astype(str), "missing": df.isna().sum(),
    }))
    show("Season coverage", df.groupby("season").agg(
        rows=("game_id", "size"), unique_games=("game_id", "nunique"),
        unique_players=("player_id", "nunique"),
        first_game=("game_date", "min"), last_game=("game_date", "max"),
    ))

    player_seasons = df.groupby(["player_id", "season"]).agg(
        rows=("game_id", "size"), distinct_games=("game_id", "nunique"),
        distinct_dates=("game_date", "nunique"),
    )
    duplicates = int(df.duplicated(["player_id", "game_id"]).sum())
    missing_keys = int(df[["player_id", "game_id"]].isna().any(axis=1).sum())
    multi_game = int(player_seasons["distinct_games"].gt(1).sum())
    print("\nObservation granularity")
    print(f"Duplicate (player_id, game_id) keys: {duplicates:,}; rows with missing keys: {missing_keys:,}")
    print(f"Player-seasons containing multiple distinct games: {multi_game:,}/{len(player_seasons):,}")
    if multi_game and not duplicates and not missing_keys:
        print("RESULT: Per-game player statistics: one row per player appearance in a game.")
        print("Season files contain many game records, not one season-wide statistical row per player.")
    elif multi_game:
        print("RESULT: Multiple game records per player-season, but key issues need inspection.")
    else:
        print("RESULT: This selection does not establish game-level granularity; inspect a larger sample.")
    show("Observed games per player-season", player_seasons["distinct_games"].describe())

    if args.player:
        names = df["player_name"].astype("string")
        matched = df.loc[df["player_id"].eq(args.player) | names.str.contains(
            args.player, case=False, regex=False, na=False
        )]
        candidates = matched[["player_id", "player_name"]].drop_duplicates()
        if matched.empty:
            parser.error(f"No player matches {args.player!r} in the selected seasons.")
        if matched["player_id"].nunique() > 1:
            parser.error(f"Multiple players match; rerun with a player ID:\n{candidates.to_string(index=False)}")
        selected_id = matched["player_id"].iloc[0]
        selected_season = matched.groupby("season").size().idxmax()
    else:
        # A player-season with many rows makes the data's granularity easy to see.
        selected_id, selected_season = player_seasons["distinct_games"].idxmax()

    example = df.loc[df["player_id"].eq(selected_id) & df["season"].eq(selected_season)]
    print(f"\nExample: {example['player_name'].iloc[0]} (ID {selected_id}), {selected_season}")
    print(f"{len(example):,} rows, {example['game_id'].nunique():,} distinct games in this player-season.")
    columns = [c for c in ["player_id", "player_name", "season", "game_id", "game_date",
                           "team", "opponent", "home", "minutes", "pts", "reb", "ast"] if c in df]
    show(f"First {min(args.rows, len(example))} individual game records", example[columns].head(args.rows).reset_index(drop=True))
    numeric_stats = [c for c in ["minutes", "pts", "reb", "ast"] if c in example]
    show("Season summary COMPUTED from these game rows (not an original record)",
         example[numeric_stats].agg(["count", "mean", "sum"]).T)
    print("\nInterpretation: pts, minutes, reb, and ast describe that row's game.")
    print("Season averages/totals must be calculated by grouping game rows.")
    print("DNPs are excluded by the source; an absent player-game row is not a recorded zero.")
    print("For future-game prediction, use lagged statistics, not outcomes from the game being predicted.")


if __name__ == "__main__":
    main()
