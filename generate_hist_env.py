"""
generate_hist_env.py
--------------------
Point-in-time backfill of the grounded environment signal used by the
Wizard thermostat. For every unique (date, home_park_id) in the batter
training matrix, replay get_dynamic_bias() as it would have fired on the
MORNING of that game — i.e. using only residual rows with date < that day.

Outputs:
    data/logs/historical_env_lookup.csv
        columns: date, home_park_id, bias_offset, projected_total_adj,
                 bias_source, park_blend, park_n, n_recent

Honesty note: the residual log (data/logs/model_residuals.csv) only
contains dates in the 2026 season. For any training row whose game_date
predates the earliest residual + BIAS_WINDOW_DAYS, get_dynamic_bias()
legitimately returns GLOBAL_BIAS_FALLBACK. We do not fabricate history.

projected_total_adj is reconstructed as LEAGUE_BASELINE + bias_offset.
LEAGUE_BASELINE is fixed at 8.80 runs/game (the MLB full-season average
used elsewhere in the pipeline as a prior). The relative signal across
parks/dates is what the stacker consumes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import score_run_dist_today as srd  # noqa: E402

MATRIX_FILE = Path("data/batter_features/final_training_matrix.parquet")
RESIDUAL_LOG = Path("data/logs/model_residuals.csv")
OUT_FILE = Path("data/logs/historical_env_lookup.csv")

LEAGUE_BASELINE = 8.80


def _pit_bias(as_of: pd.Timestamp, home_park_id: str,
              full_log: pd.DataFrame) -> tuple[float, dict, int]:
    """Replay get_dynamic_bias() using only rows with date < as_of."""
    pit = full_log[full_log["date"] < as_of]
    meta: dict = {"source": "fallback", "park_blend": "fallback",
                  "global_residual": None, "park_residual": None,
                  "damping": srd.BIAS_DAMPING}
    if pit.empty:
        return srd.GLOBAL_BIAS_FALLBACK, meta, 0

    cutoff = pit["date"].max() - pd.Timedelta(days=srd.BIAS_WINDOW_DAYS)
    window = pit[pit["date"] >= cutoff].sort_values("date")
    recent = window.tail(srd.BIAS_MAX_ROWS) if len(window) > srd.BIAS_MAX_ROWS else window
    if len(recent) < srd.BIAS_MIN_ROWS:
        return srd.GLOBAL_BIAS_FALLBACK, meta, len(recent)

    global_residual = float(recent["residual"].mean())
    meta["source"] = "dynamic"
    meta["global_residual"] = global_residual

    park_rows = window[window["home_park_id"] == home_park_id]
    if len(park_rows) >= srd.BIAS_PARK_MIN_ROWS:
        park_residual = float(park_rows["residual"].mean())
        meta["park_residual"] = park_residual
        meta["park_n"] = int(len(park_rows))
        if global_residual < 0 and park_residual > 0:
            meta["park_blend"] = "safety_valve"
            offset = srd._clamp_offset(-park_residual * srd.BIAS_DAMPING, meta)
            return offset, meta, len(recent)
        blended = (1.0 - srd.BIAS_PARK_BLEND_WEIGHT) * global_residual \
                  + srd.BIAS_PARK_BLEND_WEIGHT * park_residual
        meta["park_blend"] = "blend"
        offset = srd._clamp_offset(-blended * srd.BIAS_DAMPING, meta)
        return offset, meta, len(recent)

    meta["park_blend"] = "global"
    offset = srd._clamp_offset(-global_residual * srd.BIAS_DAMPING, meta)
    return offset, meta, len(recent)


def main():
    if not MATRIX_FILE.exists():
        raise SystemExit(f"Missing {MATRIX_FILE}. Build it first.")
    if not RESIDUAL_LOG.exists():
        raise SystemExit(f"Missing {RESIDUAL_LOG}.")

    log = pd.read_csv(RESIDUAL_LOG)
    log["date"] = pd.to_datetime(log["date"], errors="coerce")
    log = log.dropna(subset=["date", "residual"])
    print(f"Residual log: {len(log):,} rows | "
          f"{log['date'].min().date()} .. {log['date'].max().date()}")

    mat = pd.read_parquet(MATRIX_FILE, columns=["game_date", "home_park_id"])
    mat["date"] = pd.to_datetime(mat["game_date"], errors="coerce")
    mat = mat.dropna(subset=["date", "home_park_id"])
    pairs = (mat[["date", "home_park_id"]]
             .drop_duplicates()
             .sort_values(["date", "home_park_id"])
             .reset_index(drop=True))
    print(f"Unique (date, park) pairs to backfill: {len(pairs):,}")

    out_rows = []
    for i, (d, park) in enumerate(zip(pairs["date"], pairs["home_park_id"]), 1):
        offset, meta, n_recent = _pit_bias(d, park, log)
        out_rows.append({
            "date": d.date().isoformat(),
            "home_park_id": park,
            "bias_offset": round(offset, 3),
            "projected_total_adj": round(LEAGUE_BASELINE + offset, 3),
            "bias_source": meta.get("source", "fallback"),
            "park_blend": meta.get("park_blend", "fallback"),
            "park_residual": (round(meta["park_residual"], 3)
                               if meta.get("park_residual") is not None else None),
            "park_n": meta.get("park_n", 0),
            "global_residual": (round(meta["global_residual"], 3)
                                 if meta.get("global_residual") is not None else None),
            "n_recent": int(n_recent),
        })
        if i % 500 == 0:
            print(f"  [{i}/{len(pairs)}]")

    out = pd.DataFrame(out_rows)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_FILE, index=False)
    print(f"\nWrote {len(out):,} rows -> {OUT_FILE}")

    print("\n" + "="*60)
    print("BACKFILL AUDIT")
    print("="*60)
    print("bias_source breakdown:")
    print(out["bias_source"].value_counts().to_string())
    print("\npark_blend breakdown:")
    print(out["park_blend"].value_counts().to_string())
    dyn = out[out["bias_source"] == "dynamic"]
    print(f"\nDynamic coverage: {len(dyn):,} / {len(out):,} "
          f"({len(dyn)/max(len(out),1)*100:.1f}%)")
    if len(dyn):
        print(f"  projected_total_adj range: "
              f"{dyn['projected_total_adj'].min():.3f} .. "
              f"{dyn['projected_total_adj'].max():.3f}")
        print(f"  mean: {dyn['projected_total_adj'].mean():.3f}")


if __name__ == "__main__":
    main()
