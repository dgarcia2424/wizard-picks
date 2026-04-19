"""
odds_combine.py
===============
Merges all odds sources into one clean parquet per year.

Sources merged (in deduplication priority order):
  1. odds_historical_{year}.parquet  (OddsPortal — highest quality)
  2. odds_current_{YYYY_MM_DD}.parquet files (Odds API / ActionNetwork daily pulls)

For each year:
  - Concatenates all available sources
  - Deduplicates on (game_date, home_team, away_team), preferring OddsPortal > Odds API > ActionNetwork
  - Joins with mlb_schedule_{year}.parquet to fill in game_pk where missing
  - Outputs odds_combined_{year}.parquet

Usage:
    python odds_combine.py             # process all years
    python odds_combine.py --year 2026 # process single year
    python odds_combine.py --summary   # print coverage summary only
"""

import argparse
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("./data/statcast")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2023, 2024, 2025, 2026]

# Dedup priority (lower number = higher priority, kept when duplicates found)
SOURCE_PRIORITY = {
    "oddsportal": 0,
    "odds_api": 1,
    "actionnetwork": 2,
}
DEFAULT_PRIORITY = 99  # unknown sources rank lowest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / "odds_combine.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

SCHEMA_COLS = [
    "game_date", "game_pk", "home_team", "away_team",
    "open_ml_home", "close_ml_home", "open_ml_away", "close_ml_away",
    "open_total", "close_total", "runline_home", "runline_home_odds",
    "public_pct_home", "public_pct_over", "source", "pull_timestamp",
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_historical(year: int) -> pd.DataFrame:
    """Load OddsPortal historical parquet for a year."""
    path = OUTPUT_DIR / f"odds_historical_{year}.parquet"
    if not path.exists():
        log.info("No historical file for year %d: %s", year, path)
        return pd.DataFrame(columns=SCHEMA_COLS)
    df = pd.read_parquet(path, engine="pyarrow")
    log.info("Loaded historical %d: %d rows from %s", year, len(df), path.name)
    return df


def load_current_files(year: int) -> pd.DataFrame:
    """
    Load all odds_current_YYYY_MM_DD.parquet files matching the given year
    from OUTPUT_DIR.
    """
    pattern = re.compile(rf"odds_current_{year}_\d{{2}}_\d{{2}}\.parquet$")
    matching = sorted(OUTPUT_DIR.glob(f"odds_current_{year}_*.parquet"))

    frames = []
    for path in matching:
        if not pattern.match(path.name):
            log.debug("Skipping non-matching file: %s", path.name)
            continue
        try:
            df = pd.read_parquet(path, engine="pyarrow")
            frames.append(df)
            log.debug("Loaded current file: %s (%d rows)", path.name, len(df))
        except Exception as exc:
            log.warning("Failed to load %s: %s", path.name, exc)

    if not frames:
        log.info("No odds_current files found for year %d", year)
        return pd.DataFrame(columns=SCHEMA_COLS)

    combined = pd.concat(frames, ignore_index=True)
    log.info("Loaded %d odds_current files for year %d: %d total rows", len(frames), year, len(combined))
    return combined


def load_schedule(year: int) -> Optional[pd.DataFrame]:
    """Load schedule file for the year to obtain game_pk mappings."""
    candidates = [
        OUTPUT_DIR / f"mlb_schedule_{year}.parquet",
        OUTPUT_DIR / f"schedule_all_{year}.parquet",
    ]
    for path in candidates:
        if path.exists():
            try:
                df = pd.read_parquet(path, engine="pyarrow")
                log.info("Loaded schedule %d from %s: %d rows", year, path.name, len(df))
                return df
            except Exception as exc:
                log.warning("Failed to load schedule %s: %s", path.name, exc)
    log.info("No schedule file found for year %d — game_pk fill will be skipped", year)
    return None


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------
def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Add any missing schema columns and reorder."""
    for col in SCHEMA_COLS:
        if col not in df.columns:
            df[col] = None
    return df[SCHEMA_COLS].copy()


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Apply standard type coercions to an odds DataFrame."""
    df = df.copy()
    df["game_date"] = df["game_date"].astype(str).str[:10]
    df["home_team"] = df["home_team"].astype(str)
    df["away_team"] = df["away_team"].astype(str)

    for int_col in ["game_pk", "open_ml_home", "close_ml_home", "open_ml_away",
                    "close_ml_away", "runline_home_odds"]:
        df[int_col] = pd.to_numeric(df[int_col], errors="coerce").astype("Int64")

    for float_col in ["open_total", "close_total", "runline_home",
                      "public_pct_home", "public_pct_over"]:
        df[float_col] = pd.to_numeric(df[float_col], errors="coerce")

    df["pull_timestamp"] = pd.to_datetime(df["pull_timestamp"], errors="coerce")
    df["source"] = df["source"].fillna("unknown").astype(str)
    return df


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
def deduplicate_by_source_priority(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate on (game_date, home_team, away_team).
    When multiple rows exist for the same game, keep the row from the
    highest-priority source: oddsportal > odds_api > actionnetwork.

    If multiple rows share the same source, keep the most recently pulled one.
    """
    if df.empty:
        return df

    df = df.copy()
    df["_priority"] = df["source"].map(lambda s: SOURCE_PRIORITY.get(str(s), DEFAULT_PRIORITY))

    # Sort by priority ASC, then pull_timestamp DESC (newest first within priority)
    df = df.sort_values(
        ["game_date", "home_team", "away_team", "_priority", "pull_timestamp"],
        ascending=[True, True, True, True, False],
        na_position="last",
    )

    # Keep first occurrence per game (= highest priority, most recent pull)
    deduped = df.drop_duplicates(subset=["game_date", "home_team", "away_team"], keep="first")
    deduped = deduped.drop(columns=["_priority"])

    n_removed = len(df) - len(deduped)
    if n_removed > 0:
        log.info("Deduplication removed %d duplicate rows", n_removed)

    return deduped


# ---------------------------------------------------------------------------
# Schedule join (fill game_pk)
# ---------------------------------------------------------------------------
def _normalise_schedule_teams(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise team columns in a schedule DataFrame."""
    # Try to locate home/away columns
    home_candidates = [c for c in df.columns if "home" in c.lower() and "team" in c.lower()]
    away_candidates = [c for c in df.columns if "away" in c.lower() and "team" in c.lower()]
    date_candidates = [c for c in df.columns if "date" in c.lower() or c.lower() in ("game_date",)]
    pk_candidates = [c for c in df.columns if "pk" in c.lower() or "id" in c.lower()]

    home_col = home_candidates[0] if home_candidates else "home_team"
    away_col = away_candidates[0] if away_candidates else "away_team"
    date_col = date_candidates[0] if date_candidates else "game_date"
    pk_col = pk_candidates[0] if pk_candidates else "game_pk"

    sched = df[[date_col, home_col, away_col, pk_col]].copy()
    sched.columns = ["game_date", "home_team", "away_team", "game_pk"]
    sched["game_date"] = sched["game_date"].astype(str).str[:10]
    sched["home_team"] = sched["home_team"].astype(str).str.strip().str.upper()
    sched["away_team"] = sched["away_team"].astype(str).str.strip().str.upper()
    return sched


def fill_game_pk_from_schedule(odds_df: pd.DataFrame, schedule_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join odds DataFrame with schedule to fill in missing game_pk values.
    Only fills rows where game_pk is currently null.
    """
    if schedule_df is None or schedule_df.empty:
        return odds_df

    sched = _normalise_schedule_teams(schedule_df)
    sched = sched.dropna(subset=["game_pk"])
    sched["game_pk"] = pd.to_numeric(sched["game_pk"], errors="coerce").astype("Int64")
    sched = sched.drop_duplicates(subset=["game_date", "home_team", "away_team"])

    # Normalise odds team columns to uppercase for join
    odds_df = odds_df.copy()
    odds_df["_home_upper"] = odds_df["home_team"].str.upper()
    odds_df["_away_upper"] = odds_df["away_team"].str.upper()

    # Merge
    merged = odds_df.merge(
        sched.rename(columns={"game_pk": "_sched_pk"}),
        left_on=["game_date", "_home_upper", "_away_upper"],
        right_on=["game_date", "home_team", "away_team"],
        how="left",
        suffixes=("", "_sched"),
    )

    # Fill game_pk where missing
    needs_fill = merged["game_pk"].isna() & merged["_sched_pk"].notna()
    n_filled = needs_fill.sum()
    merged.loc[needs_fill, "game_pk"] = merged.loc[needs_fill, "_sched_pk"]

    if n_filled > 0:
        log.info("Filled game_pk for %d games from schedule", n_filled)

    # Drop helper columns
    drop_cols = ["_home_upper", "_away_upper", "_sched_pk", "home_team_sched", "away_team_sched"]
    merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])

    return merged


# ---------------------------------------------------------------------------
# Per-year processing
# ---------------------------------------------------------------------------
def process_year(year: int) -> Optional[pd.DataFrame]:
    """
    Combine all odds sources for a single year.
    Returns the combined DataFrame, or None if no data found.
    """
    log.info("=== Processing year %d ===", year)

    frames = []

    # Source 1: OddsPortal historical
    hist_df = load_historical(year)
    if not hist_df.empty:
        frames.append(_ensure_schema(hist_df))

    # Source 2: Daily current files (Odds API + ActionNetwork)
    current_df = load_current_files(year)
    if not current_df.empty:
        frames.append(_ensure_schema(current_df))

    if not frames:
        log.warning("No odds data found for year %d — skipping", year)
        return None

    combined = pd.concat(frames, ignore_index=True)
    combined = _coerce_types(combined)

    log.info("Year %d — total rows before dedup: %d", year, len(combined))

    combined = deduplicate_by_source_priority(combined)

    # Fill game_pk from schedule
    sched = load_schedule(year)
    if sched is not None:
        combined = fill_game_pk_from_schedule(combined, sched)
        combined = _coerce_types(combined)  # re-coerce after merge

    combined = combined.sort_values(["game_date", "home_team"]).reset_index(drop=True)

    output_path = OUTPUT_DIR / f"odds_combined_{year}.parquet"
    combined.to_parquet(output_path, engine="pyarrow", index=False)
    print(f"Saved {output_path.name} — shape: {combined.shape}")
    log.info("Saved %s — shape: %s", output_path.name, combined.shape)

    return combined


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------
def _load_schedule_game_count(year: int) -> int:
    """Load total expected game count for the year from schedule."""
    sched = load_schedule(year)
    if sched is None or sched.empty:
        return 0
    return len(sched)


def _count_by_source(df: pd.DataFrame) -> dict[str, int]:
    """Return count of games per source."""
    if df.empty or "source" not in df.columns:
        return {}
    return df["source"].value_counts().to_dict()


def coverage_report_row(year: int, df: Optional[pd.DataFrame]) -> dict:
    """Build a coverage summary row for a single year."""
    if df is None or df.empty:
        return {
            "year": year,
            "total_games": 0,
            "with_lines": 0,
            "coverage_pct": 0.0,
            "sources": {},
            "schedule_total": _load_schedule_game_count(year),
        }

    total = len(df)
    with_lines = df["close_ml_home"].notna().sum()
    coverage = (with_lines / total * 100) if total > 0 else 0.0
    sources = _count_by_source(df)

    return {
        "year": year,
        "total_games": total,
        "with_lines": int(with_lines),
        "coverage_pct": round(coverage, 1),
        "sources": sources,
        "schedule_total": _load_schedule_game_count(year),
    }


def odds_combine_summary(year_results: dict[int, Optional[pd.DataFrame]]) -> None:
    """
    Print a formatted coverage summary table.

    Example output:
    Year | Total Games | With Lines | Coverage | Sources
    2024 |        2521 |       2450 |    97.2% | oddsportal: 2310, odds_api: 140, actionnetwork: 0
    """
    print()
    print("=" * 80)
    print("ODDS COVERAGE SUMMARY")
    print("=" * 80)
    header = f"{'Year':<6} | {'Sched':>7} | {'Combined':>8} | {'w/ Lines':>8} | {'Coverage':>8} | Sources"
    print(header)
    print("-" * 80)

    for year in sorted(year_results.keys()):
        df = year_results[year]
        row = coverage_report_row(year, df)

        sched_total = row["schedule_total"]
        total = row["total_games"]
        with_lines = row["with_lines"]
        coverage = row["coverage_pct"]
        sources = row["sources"]

        source_str = ", ".join(
            f"{s}: {n}" for s, n in sorted(sources.items(), key=lambda x: SOURCE_PRIORITY.get(x[0], 99))
        ) if sources else "—"

        print(
            f"{year:<6} | {sched_total:>7,} | {total:>8,} | {with_lines:>8,} | "
            f"{coverage:>7.1f}% | {source_str}"
        )

    print("=" * 80)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Combine all MLB odds sources per year")
    parser.add_argument("--year", type=int, default=None, help="Process only this year (default: all)")
    parser.add_argument("--summary", action="store_true", help="Print coverage summary only (no re-processing)")
    args = parser.parse_args()

    target_years = [args.year] if args.year else YEARS

    year_results: dict[int, Optional[pd.DataFrame]] = {}

    if args.summary:
        # Load existing combined files for summary only
        for year in target_years:
            path = OUTPUT_DIR / f"odds_combined_{year}.parquet"
            if path.exists():
                df = pd.read_parquet(path, engine="pyarrow")
                year_results[year] = df
                log.info("Loaded existing combined %d: %d rows", year, len(df))
            else:
                year_results[year] = None
    else:
        for year in target_years:
            df = process_year(year)
            year_results[year] = df

    odds_combine_summary(year_results)


if __name__ == "__main__":
    main()
