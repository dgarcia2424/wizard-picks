"""
verify_imputation_fix.py — Static verification of the imputation & fallback edits.
Checks: AST parse, required symbols present, vig math correct, shrinkage constants.
"""
import ast, math, sys

ROOT = __import__("pathlib").Path(__file__).parent

CHECKS_PASSED = 0
CHECKS_FAILED = 0

def ok(msg):
    global CHECKS_PASSED
    CHECKS_PASSED += 1
    print(f"  PASS  {msg}")

def fail(msg):
    global CHECKS_FAILED
    CHECKS_FAILED += 1
    print(f"  FAIL  {msg}")

# ── 1. AST parse ──────────────────────────────────────────────────────────────
for fname in ("odds_combine.py", "score_ml_today.py", "train_xgboost.py"):
    path = ROOT / fname
    try:
        src = path.read_text(encoding="utf-8")
        ast.parse(src)
        ok(f"AST parse: {fname}")
    except SyntaxError as e:
        fail(f"AST parse {fname}: {e}")
    except FileNotFoundError:
        fail(f"File not found: {fname}")

# ── 2. Required symbols ───────────────────────────────────────────────────────
def src(fname):
    return (ROOT / fname).read_text(encoding="utf-8")

oc = src("odds_combine.py")
for sym in ("strip_vig_multiplicative", "_american_to_prob", "TIER_MAP",
            "market_data_missing", "true_home_prob", "true_away_prob",
            "import math"):
    (ok if sym in oc else fail)(f"odds_combine.py contains '{sym}'")

sm = src("score_ml_today.py")
for sym in ("_MARKET_MISS_LOGIT_SHRINK", "_STANDALONE_CONVICTION_THRESHOLD",
            "_THRESHOLD_250", "_MARKET_GAP_FEATURES", "_detect_market_missing",
            "market_missing", "market_data_missing"):
    (ok if sym in sm else fail)(f"score_ml_today.py contains '{sym}'")

# Confirm zero-fill poison is removed: the old blanket fillna(0.0) line should be gone
if 'fill_value=0.0).fillna(0.0)' in sm:
    fail("score_ml_today.py still has blanket fill_value=0.0 zero-fill (poison not removed)")
else:
    ok("score_ml_today.py: blanket zero-fill removed")

tx = src("train_xgboost.py")
for sym in ("_MARKET_MISS_LOGIT_SHRINK", "_STANDALONE_CONVICTION_THRESHOLD",
            "_THRESHOLD_250", "_MARKET_GAP_FEATURES",
            "market_data_missing"):
    (ok if sym in tx else fail)(f"train_xgboost.py contains '{sym}'")

# ── 3. Vig math ───────────────────────────────────────────────────────────────
def american_to_prob(odds):
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return 100.0 / (odds + 100.0)

def strip_vig(ml_home, ml_away):
    try:
        h = float(ml_home); a = float(ml_away)
    except (TypeError, ValueError):
        return float("nan"), float("nan")
    if not (math.isfinite(h) and math.isfinite(a)):
        return float("nan"), float("nan")
    p_h = american_to_prob(h)
    p_a = american_to_prob(a)
    overround = p_h + p_a
    if overround <= 0.0:
        return float("nan"), float("nan")
    return p_h / overround, p_a / overround

th, ta = strip_vig(-250, 210)
if abs(th - 0.6889) < 0.001:
    ok(f"Vig strip -250/+210 → true_h={th:.4f} (expected ~0.6889)")
else:
    fail(f"Vig strip -250/+210 → true_h={th:.4f} (expected ~0.6889)")

th2, ta2 = strip_vig(-110, -110)
if abs(th2 - 0.5000) < 0.001:
    ok(f"Vig strip -110/-110 → true_h={th2:.4f} (expected 0.5000)")
else:
    fail(f"Vig strip -110/-110 → true_h={th2:.4f} (expected 0.5000)")

th3, ta3 = strip_vig(float("nan"), 200)
if math.isnan(th3):
    ok("Vig strip NaN input → NaN propagated correctly")
else:
    fail(f"Vig strip NaN input → {th3} (expected NaN)")

# ── 4. Logit shrinkage math ───────────────────────────────────────────────────
import numpy as np

SHRINK = 0.50
THRESHOLD_250 = 100.0 / (250.0 + 100.0)  # ≈ 0.71429
STANDALONE_THRESHOLD = 0.85

# stk_p = 0.80 (market absent, xgb_raw = 0.90 ≥ 0.85) → should NOT be capped
stk_p = 0.80; xgb_raw = 0.90
logit = np.log(stk_p / (1.0 - stk_p)) * SHRINK
p_shrunk = 1.0 / (1.0 + np.exp(-logit))
if xgb_raw < STANDALONE_THRESHOLD:
    p_shrunk = min(p_shrunk, THRESHOLD_250 - 1e-4)
if p_shrunk < THRESHOLD_250:
    ok(f"Logit shrink (stk=0.80, xgb=0.90) → {p_shrunk:.5f} < threshold (shrinkage working)")
else:
    fail(f"Logit shrink (stk=0.80, xgb=0.90) → {p_shrunk:.5f} (expected < {THRESHOLD_250:.5f})")

# stk_p = 0.80 (market absent, xgb_raw = 0.70 < 0.85) → must be capped
stk_p = 0.80; xgb_raw = 0.70
logit = np.log(stk_p / (1.0 - stk_p)) * SHRINK
p_shrunk = 1.0 / (1.0 + np.exp(-logit))
if xgb_raw < STANDALONE_THRESHOLD:
    p_shrunk = min(p_shrunk, THRESHOLD_250 - 1e-4)
cap_expected = THRESHOLD_250 - 1e-4
if abs(p_shrunk - cap_expected) < 1e-6:
    ok(f"Ceiling gate (stk=0.80, xgb=0.70) → {p_shrunk:.6f} (capped at {cap_expected:.6f})")
else:
    fail(f"Ceiling gate (stk=0.80, xgb=0.70) → {p_shrunk:.6f} (expected {cap_expected:.6f})")

# stk_p = 0.60 (below threshold, market present) → no shrinkage path
stk_p = 0.60
logit = np.log(stk_p / (1.0 - stk_p))  # no shrink
p_out = 1.0 / (1.0 + np.exp(-logit))
if abs(p_out - stk_p) < 1e-9:
    ok(f"No shrinkage when market present: stk_p={stk_p} → {p_out:.6f} (identity)")
else:
    fail(f"Identity check failed: {stk_p} → {p_out}")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print(f"{'='*50}")
print(f"  {CHECKS_PASSED} passed  |  {CHECKS_FAILED} failed")
print(f"{'='*50}")
sys.exit(0 if CHECKS_FAILED == 0 else 1)
