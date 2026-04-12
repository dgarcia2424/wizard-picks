"""
build_lineup_quality.py
-----------------------
Compute a lineup-weighted wRC+ score for each team in today's games.

This replaces the season-average RS/G offensive factor in the Monte Carlo
simulation with a score derived from today's actual confirmed starters,
implicitly capturing injuries, rest days, and day-to-day roster changes.

Usage (programmatic):
    from build_lineup_quality import build
    lq = build("2026-04-12")          # {(game_pk, team_abbr): wrc_plus}
    lq = build("2026-04-12", verbose=True)

Output:
    data/statcast/lineup_quality_today.parquet
    Columns: game_pk, team, lineup_wrc_plus, n_matched, game_date
"""

import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("data/statcast")
LINEUP_LONG_PATH = OUTPUT_DIR / "lineups_today_long.parquet"
LINEUP_QUALITY_PATH = OUTPUT_DIR / "lineup_quality_today.parquet"
FANGRAPHS_PATH = Path("data/raw/fangraphs_batters.csv")

LEAGUE_WRC_PLUS = 100.0

# Staleness threshold: if the long parquet is older than this, refresh it
STALE_HOURS = 4


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------

def _norm_name(name: str) -> str:
    """Normalize player name: strip accents, uppercase, remove Jr/Sr/II/III."""
    if not isinstance(name, str):
        return ""
    name = "".join(
        c for c in unicodedata.normalize("NFD", name)
        if unicodedata.category(c) != "Mn"
    ).upper().strip()
    # Remove common suffixes
    for suffix in [" JR.", " SR.", " II", " III", " IV", " JR", " SR"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)].strip()
    return name


# ---------------------------------------------------------------------------
# wRC+ lookup builder
# ---------------------------------------------------------------------------

def _build_wrc_lookup() -> dict:
    """
    Build a name → wRC+ dict from fangraphs_batters.csv.

    Priority: 2026 (≥10 PA) > 2025 (≥50 PA) > 2024 (≥50 PA).
    Later years overwrite earlier entries in the dict, so 2026 wins.
    """
    fg = pd.read_csv(FANGRAPHS_PATH, encoding="utf-8-sig")
    fg["year"] = pd.to_numeric(fg["year"], errors="coerce")
    fg["wrc_plus"] = pd.to_numeric(fg["wRC+"], errors="coerce")
    fg["PA"] = pd.to_numeric(fg["PA"], errors="coerce")
    fg["name_norm"] = fg["Name"].apply(_norm_name)

    wrc_lookup: dict[str, float] = {}
    for year in [2024, 2025, 2026]:
        min_pa = 10 if year == 2026 else 50
        sub = fg[
            (fg["year"] == year) & (fg["PA"] >= min_pa) & fg["wrc_plus"].notna()
        ]
        for _, row in sub.iterrows():
            wrc_lookup[row["name_norm"]] = float(row["wrc_plus"])
        # Later years overwrite earlier — so 2026 takes priority

    return wrc_lookup


# ---------------------------------------------------------------------------
# Per-lineup wRC+ computation
# ---------------------------------------------------------------------------

def lineup_wrc_plus(
    player_names: list[str], wrc_lookup: dict
) -> tuple[float, int]:
    """
    Return (avg_wrc_plus, n_matched) for a list of batter names.

    Falls back to LEAGUE_WRC_PLUS if fewer than 3 names can be matched.
    """
    scores = []
    for name in player_names:
        norm = _norm_name(name)
        if norm in wrc_lookup:
            scores.append(wrc_lookup[norm])
    if len(scores) < 3:  # not enough matches — return league average
        return LEAGUE_WRC_PLUS, len(scores)
    return sum(scores) / len(scores), len(scores)


# ---------------------------------------------------------------------------
# Lineup refresh helper
# ---------------------------------------------------------------------------

def _refresh_lineups(date_str: str, verbose: bool = True) -> None:
    """
    Pull fresh lineup data for date_str by calling lineup_pull functions
    directly (no subprocess).
    """
    from lineup_pull import (
        ensure_dirs,
        pull_date,
        records_to_wide,
        records_to_long,
        save_today_outputs,
    )

    ensure_dirs()
    if verbose:
        print(f"  Refreshing lineups for {date_str} ...")
    records = pull_date(date_str, verbose=verbose)
    if not records:
        if verbose:
            print("  [WARN] No lineup records returned from MLB Stats API.")
        return
    wide_df = records_to_wide(records)
    long_df = records_to_long(records)
    save_today_outputs(wide_df, long_df, date_str=date_str)


# ---------------------------------------------------------------------------
# Staleness check
# ---------------------------------------------------------------------------

def _is_stale(path: Path, max_hours: float = STALE_HOURS) -> bool:
    """Return True if the file doesn't exist, is empty, or is older than max_hours."""
    if not path.exists():
        return True
    age_hours = (datetime.now().timestamp() - path.stat().st_mtime) / 3600
    if age_hours > max_hours:
        return True
    # Check if the parquet is effectively empty
    try:
        df = pd.read_parquet(path, engine="pyarrow")
        return df.empty
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build(date_str: str, verbose: bool = True) -> dict:
    """
    Compute lineup wRC+ for every (game_pk, team) playing on date_str.

    Steps:
      1. Load (or refresh) lineups long parquet for date_str.
         Prefers date-stamped file (lineups_{date_str}_long.parquet);
         falls back to lineups_today_long.parquet when date_str == today.
      2. Build wRC+ lookup from fangraphs_batters.csv.
      3. For each game+team, compute average wRC+ of confirmed starters.
      4. Save lineup_quality_{date_str}.parquet (and lineup_quality_today.parquet
         when date_str == today).
      5. Return dict keyed by (game_pk, team_abbr) → lineup_wrc_plus.

    Falls back gracefully: returns {} if no lineup data is available.
    """
    import datetime as _dt
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    today_str = _dt.date.today().isoformat()
    is_today  = (date_str == today_str)

    # --- Step 1: Load or refresh lineup data ---------------------------------
    # Prefer date-stamped long file; fall back to canonical today file
    dated_long_path = OUTPUT_DIR / f"lineups_{date_str}_long.parquet"
    long_path = dated_long_path if dated_long_path.exists() else LINEUP_LONG_PATH

    if _is_stale(long_path):
        try:
            _refresh_lineups(date_str, verbose=verbose)
            # After refresh, prefer the newly written dated file
            if dated_long_path.exists():
                long_path = dated_long_path
        except Exception as exc:
            if verbose:
                print(f"  [WARN] Could not refresh lineups: {exc}")

    try:
        long_df = pd.read_parquet(long_path, engine="pyarrow")
    except Exception as exc:
        if verbose:
            print(f"  [WARN] Could not read {long_path}: {exc}")
        return {}

    if long_df.empty:
        if verbose:
            print(f"  [INFO] {long_path.name} is empty — no lineup quality scores available.")
        return {}

    # Filter to the requested date if game_date column is present
    if "game_date" in long_df.columns:
        long_df = long_df[long_df["game_date"].astype(str).str.startswith(date_str[:10])]
        if long_df.empty:
            if verbose:
                print(f"  [INFO] No lineup rows found for {date_str}.")
            return {}

    # --- Step 2: Build wRC+ lookup -------------------------------------------
    try:
        wrc_lookup = _build_wrc_lookup()
        if verbose:
            print(f"  wRC+ lookup built: {len(wrc_lookup):,} players")
    except Exception as exc:
        if verbose:
            print(f"  [WARN] Could not build wRC+ lookup: {exc}")
        return {}

    # --- Step 3: Compute lineup wRC+ per game+team ---------------------------
    quality_rows = []
    result_dict: dict = {}

    grouped = long_df.groupby(["game_pk", "team"])
    for (game_pk, team), group in grouped:
        player_names = group["player_name"].dropna().tolist()
        avg_wrc, n_matched = lineup_wrc_plus(player_names, wrc_lookup)

        quality_rows.append({
            "game_pk":         game_pk,
            "team":            team,
            "lineup_wrc_plus": round(avg_wrc, 2),
            "n_matched":       n_matched,
            "game_date":       date_str,
        })
        result_dict[(game_pk, team)] = avg_wrc

        if verbose:
            print(
                f"  {team:>4s}  game_pk={game_pk}  "
                f"lineup_wrc+={avg_wrc:.1f}  matched={n_matched}/{len(player_names)}"
            )

    # --- Step 4: Save quality parquet (dated + canonical today) --------------
    if quality_rows:
        quality_df = pd.DataFrame(quality_rows)
        # Always save date-stamped file
        dated_out = OUTPUT_DIR / f"lineup_quality_{date_str}.parquet"
        try:
            quality_df.to_parquet(dated_out, engine="pyarrow", index=False)
            if verbose:
                print(f"  Saved {dated_out}  shape={quality_df.shape}")
        except Exception as exc:
            if verbose:
                print(f"  [WARN] Could not save {dated_out}: {exc}")
        # Also save canonical today file when date_str == today
        if is_today:
            try:
                quality_df.to_parquet(LINEUP_QUALITY_PATH, engine="pyarrow", index=False)
                if verbose:
                    print(f"  Saved {LINEUP_QUALITY_PATH}  shape={quality_df.shape}")
            except Exception as exc:
                if verbose:
                    print(f"  [WARN] Could not save {LINEUP_QUALITY_PATH}: {exc}")

    return result_dict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from datetime import date

    parser = argparse.ArgumentParser(
        description="Build lineup wRC+ quality scores for today's MLB games."
    )
    parser.add_argument(
        "--date", type=str, default=date.today().strftime("%Y-%m-%d"),
        metavar="YYYY-MM-DD", help="Date to compute lineup quality for.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress verbose output.",
    )
    args = parser.parse_args()

    result = build(args.date, verbose=not args.quiet)
    print(f"\nDone. {len(result)} (game_pk, team) entries computed.")
