"""
data_catalog.py
===============
Validation script — loads all parquets in ./statcast_data/, prints a detailed
summary, and flags issues (missing files, empty files, high null rates).

Usage:
    python data_catalog.py                         # scan default ./statcast_data/
    python data_catalog.py --dir /path/to/data     # custom directory
    python data_catalog.py --years 2024 2025       # override baseline years
    python data_catalog.py --no-color              # disable ANSI coloring

Output:
    FILE STATUS section  — per-file rows with shape, date range, null warnings
    COVERAGE SUMMARY     — grid of OK / -- per data category and year
"""

import argparse
import os
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("./statcast_data")
BASELINE_YEARS = [2023, 2024, 2025, 2026]

# Key columns to check null % for each file pattern
NULL_CHECK_COLS: dict[str, list[str]] = {
    "statcast": ["pitch_type", "release_speed", "events"],
    "batting_fg": ["wRC_plus", "wOBA", "PA"],
    "pitching_fg": ["ERA", "FIP", "IP"],
    "pitching_fg_full": ["ERA", "FIP", "IP"],
    "bullpen_fg": ["ERA", "FIP", "IP"],
    "batter_xstats": ["xBA", "xSLG", "xwOBA"],
    "pitcher_xstats": ["xERA", "xFIP", "xwOBA"],
    "batter_exitvelo": ["avg_exit_velocity", "launch_angle"],
    "pitcher_exitvelo": ["avg_exit_velocity", "launch_angle"],
    "batter_percentiles": ["percentile_rank"],
    "pitcher_percentiles": ["percentile_rank"],
    "sprint_speed": ["sprint_speed"],
    "pitch_arsenal": ["avg_speed", "avg_spin"],
    "pitcher_active_spin": ["active_spin"],
    "schedule_all": ["game_date", "home_team", "away_team"],
    "standings": ["W", "L", "W-L%"],
    "mlb_schedule": ["game_date", "home_team", "away_team", "game_pk"],
    "park_factors": ["park_factor"],
    "weather": ["temp_f", "wind_speed_mph", "wind_bearing"],
    "odds_historical": ["close_ml_home", "close_ml_away"],
    "odds_combined": ["close_ml_home", "game_date"],
    "pitcher_handedness": ["throws"],
    "lineups_today": ["player_name", "team"],
}

NULL_WARN_THRESHOLD = 0.50  # 50% nulls in a key column triggers WARN

# ANSI colors
_ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "green": "\033[32m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "dim": "\033[2m",
}

USE_COLOR = True  # set by --no-color flag


def _c(text: str, *styles: str) -> str:
    """Apply ANSI color/style codes if color is enabled."""
    if not USE_COLOR:
        return text
    codes = "".join(_ANSI.get(s, "") for s in styles)
    return f"{codes}{text}{_ANSI['reset']}"


# ---------------------------------------------------------------------------
# Expected files definition
# ---------------------------------------------------------------------------
def build_expected_files(years: list[int]) -> list[str]:
    """Return the full expected file list for the given baseline years."""
    per_year_templates = [
        "statcast_{year}.parquet",
        "batting_fg_{year}.parquet",
        "pitching_fg_{year}.parquet",
        "pitching_fg_full_{year}.parquet",
        "bullpen_fg_{year}.parquet",
        "batter_xstats_{year}.parquet",
        "pitcher_xstats_{year}.parquet",
        "batter_exitvelo_{year}.parquet",
        "pitcher_exitvelo_{year}.parquet",
        "batter_percentiles_{year}.parquet",
        "pitcher_percentiles_{year}.parquet",
        "sprint_speed_{year}.parquet",
        "pitch_arsenal_{year}.parquet",
        "pitcher_active_spin_{year}.parquet",
        "schedule_all_{year}.parquet",
        "standings_{year}.parquet",
        "mlb_schedule_{year}.parquet",
        "park_factors_{year}.parquet",
        "weather_{year}.parquet",
        "odds_historical_{year}.parquet",
        "odds_combined_{year}.parquet",
        "backtest_games_{year}.csv",
    ]
    non_year_files = [
        "pitcher_handedness.parquet",
        "lineups_today.parquet",
    ]

    result = []
    for template in per_year_templates:
        for yr in years:
            result.append(template.format(year=yr))
    result.extend(non_year_files)
    return result


# ---------------------------------------------------------------------------
# File analysis helpers
# ---------------------------------------------------------------------------
def _detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """Return the first column that looks like a date column."""
    priority = ["game_date", "date", "game_Date", "Date"]
    for col in priority:
        if col in df.columns:
            return col
    for col in df.columns:
        if "date" in col.lower():
            return col
    return None


def _date_range_str(df: pd.DataFrame, date_col: str) -> str:
    """Return 'YYYY-MM-DD -> YYYY-MM-DD' range string."""
    try:
        series = pd.to_datetime(df[date_col], errors="coerce").dropna()
        if series.empty:
            return "no valid dates"
        mn = series.min().strftime("%Y-%m-%d")
        mx = series.max().strftime("%Y-%m-%d")
        return f"{mn} \u2192 {mx}"
    except Exception:
        return "?"


def _key_null_pct(df: pd.DataFrame, filename: str) -> list[tuple[str, float]]:
    """
    Return list of (column_name, null_pct) for key columns defined for this file pattern.
    Matches by checking if any key in NULL_CHECK_COLS appears in the filename.
    """
    results = []
    cols_to_check = []
    for pattern_key, cols in NULL_CHECK_COLS.items():
        if pattern_key in filename.lower():
            cols_to_check = cols
            break

    for col in cols_to_check:
        # Try to find the column case-insensitively
        match = next((c for c in df.columns if c.lower() == col.lower()), None)
        if match:
            pct = df[match].isna().mean()
            results.append((col, pct))
    return results


def analyse_file(path: Path) -> dict:
    """
    Load a parquet (or CSV) file and return analysis dict.
    Keys: status, filename, rows, cols, date_range, null_warnings, error
    """
    result = {
        "status": "OK",
        "filename": path.name,
        "rows": None,
        "cols": None,
        "date_range": None,
        "null_warnings": [],
        "error": None,
    }

    try:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path, engine="pyarrow")
        elif path.suffix == ".csv":
            df = pd.read_csv(path, low_memory=False)
        else:
            result["status"] = "SKIP"
            result["error"] = "unsupported format"
            return result
    except Exception as exc:
        result["status"] = "ERR"
        result["error"] = str(exc)
        return result

    result["rows"] = len(df)
    result["cols"] = len(df.columns)

    if len(df) == 0:
        result["status"] = "WARN"
        result["null_warnings"].append("0 rows")
        return result

    # Date range
    date_col = _detect_date_column(df)
    if date_col:
        result["date_range"] = _date_range_str(df, date_col)

    # Null checks
    null_info = _key_null_pct(df, path.name)
    for col, pct in null_info:
        if pct > NULL_WARN_THRESHOLD:
            result["null_warnings"].append(f"{pct*100:.1f}% null {col}")
            result["status"] = "WARN"

    return result


# ---------------------------------------------------------------------------
# Coverage grid
# ---------------------------------------------------------------------------
# Map file prefix patterns to category labels used in coverage grid
COVERAGE_CATEGORIES = [
    ("statcast", "Statcast"),
    ("batting_fg", "FanGraphs Bat"),
    ("pitching_fg_full", "FanGraphs Pit Full"),
    ("pitching_fg", "FanGraphs Pit"),
    ("bullpen_fg", "Bullpen"),
    ("batter_xstats", "xStats Bat"),
    ("pitcher_xstats", "xStats Pit"),
    ("batter_exitvelo", "ExitVelo Bat"),
    ("pitcher_exitvelo", "ExitVelo Pit"),
    ("sprint_speed", "Sprint Speed"),
    ("pitch_arsenal", "Arsenal"),
    ("schedule_all", "Schedule"),
    ("mlb_schedule", "MLB Sched"),
    ("standings", "Standings"),
    ("park_factors", "Park Factors"),
    ("weather", "Weather"),
    ("odds_historical", "Odds Hist"),
    ("odds_combined", "Odds Combined"),
]


def build_coverage_grid(scan_dir: Path, years: list[int]) -> dict[str, dict[int, str]]:
    """
    For each (category, year) pair, check whether the corresponding file exists.
    Returns nested dict: category_label -> {year: "OK" | "--"}
    """
    grid: dict[str, dict[int, str]] = {}
    for pattern, label in COVERAGE_CATEGORIES:
        grid[label] = {}
        for yr in years:
            # Match files like statcast_2024.parquet
            candidates = list(scan_dir.glob(f"{pattern}_{yr}.*"))
            grid[label][yr] = "OK" if candidates else "--"
    return grid


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
def print_file_status(analyses: list[dict], expected: set[str]) -> None:
    """Print the FILE STATUS section."""
    print(_c("\nFILE STATUS", "bold"))
    print(_c("-" * 70, "dim"))

    # Collect all filenames seen
    seen_names = {a["filename"] for a in analyses}

    # Print found files
    for a in sorted(analyses, key=lambda x: x["filename"]):
        status = a["status"]
        fname = a["filename"]
        rows = a["rows"]
        cols = a["cols"]
        date_range = a["date_range"]
        warns = a["null_warnings"]
        error = a["error"]

        # Status tag with color
        if status == "OK":
            tag = _c("OK  ", "green")
        elif status == "WARN":
            tag = _c("WARN", "yellow")
        elif status == "ERR":
            tag = _c("ERR ", "red")
        else:
            tag = _c("SKIP", "dim")

        # Build info string
        if rows is not None and cols is not None:
            shape_str = f"{rows:>10,} rows | {cols:>3} cols"
        else:
            shape_str = ""

        date_str = f" | {date_range}" if date_range else ""
        warn_str = ""
        if warns:
            warn_str = " | " + _c("; ".join(warns), "yellow")

        err_str = f" | {_c(error, 'red')}" if error else ""

        line = f"{tag}  {fname:<50} | {shape_str}{date_str}{warn_str}{err_str}"
        print(line)

    # Print missing expected files
    for fname in sorted(expected):
        if fname not in seen_names:
            tag = _c("MISS", "red")
            print(f"{tag}  {_c(fname, 'dim'):<50} | {_c('NOT FOUND', 'red')}")

    print()


def print_coverage_summary(grid: dict[str, dict[int, str]], years: list[int]) -> None:
    """Print the COVERAGE SUMMARY grid."""
    print(_c("COVERAGE SUMMARY", "bold"))
    print(_c("-" * 70, "dim"))

    # Header row
    year_headers = "  ".join(f"{yr}" for yr in years)
    print(f"{'Category':<22} {year_headers}")
    print(_c("-" * (22 + len(years) * 7), "dim"))

    for label, year_map in grid.items():
        cells = []
        for yr in years:
            status = year_map.get(yr, "--")
            if status == "OK":
                cells.append(_c("OK ", "green"))
            else:
                cells.append(_c("-- ", "dim"))
        print(f"{label:<22} {'  '.join(cells)}")

    print()


def print_header(scan_dir: Path) -> None:
    """Print the catalog header."""
    today = date.today().strftime("%Y-%m-%d")
    print()
    print(_c("=" * 70, "bold"))
    print(_c("=== MLB DATA CATALOG ===", "bold", "cyan"))
    print(f"Scanned : {scan_dir.resolve()}")
    print(f"Date    : {today}")
    print(_c("=" * 70, "bold"))


def print_scan_summary(analyses: list[dict], expected_count: int) -> None:
    """Print a brief overall scan summary."""
    total_files = len(analyses)
    ok_count = sum(1 for a in analyses if a["status"] == "OK")
    warn_count = sum(1 for a in analyses if a["status"] == "WARN")
    err_count = sum(1 for a in analyses if a["status"] == "ERR")
    total_rows = sum(a["rows"] for a in analyses if a["rows"] is not None)

    print()
    print(_c("SCAN SUMMARY", "bold"))
    print(_c("-" * 50, "dim"))
    print(f"  Files found   : {total_files}")
    print(f"  Expected files: {expected_count}")
    print(f"  {_c('OK', 'green')}            : {ok_count}")
    print(f"  {_c('WARN', 'yellow')}          : {warn_count}")
    print(f"  {_c('ERR', 'red')}           : {err_count}")
    print(f"  Total rows    : {total_rows:,}")
    print()


# ---------------------------------------------------------------------------
# Main scan logic
# ---------------------------------------------------------------------------
def scan_directory(scan_dir: Path, years: list[int]) -> None:
    """Full scan: analyse all files, print status, print coverage grid."""
    print_header(scan_dir)

    if not scan_dir.exists():
        print(_c(f"\nERROR: Directory not found: {scan_dir.resolve()}", "red", "bold"))
        sys.exit(1)

    # Collect all parquet and csv files (non-recursive, one level)
    all_files = sorted(
        [p for p in scan_dir.iterdir() if p.suffix in (".parquet", ".csv") and p.is_file()],
        key=lambda p: p.name,
    )

    # Also check subdirectories one level deep (e.g., odds_api_cache)
    for subdir in scan_dir.iterdir():
        if subdir.is_dir():
            all_files.extend(
                sorted(p for p in subdir.iterdir() if p.suffix in (".parquet", ".csv") and p.is_file())
            )

    if not all_files:
        print(_c("\nNo parquet or CSV files found.", "yellow"))
        return

    # Analyse each file
    analyses = []
    for path in all_files:
        analysis = analyse_file(path)
        analyses.append(analysis)

    # Expected files
    expected_names = set(build_expected_files(years))
    # Only flag expected files that are truly at the top level of scan_dir
    top_level_names = {a["filename"] for a in analyses if (scan_dir / a["filename"]).parent == scan_dir}

    print_scan_summary(analyses, len(expected_names))
    print_file_status(analyses, expected_names)

    # Coverage grid (only years that appear in any found file)
    grid = build_coverage_grid(scan_dir, years)
    print_coverage_summary(grid, years)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    global USE_COLOR

    parser = argparse.ArgumentParser(
        description="MLB Data Catalog — validate parquet files in statcast_data/"
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Directory to scan (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=BASELINE_YEARS,
        help=f"Years to include in coverage check (default: {BASELINE_YEARS})",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color output",
    )
    args = parser.parse_args()

    if args.no_color or not sys.stdout.isatty():
        USE_COLOR = False

    scan_directory(args.dir, args.years)


if __name__ == "__main__":
    main()
