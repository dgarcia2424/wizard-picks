import pandas as pd, numpy as np, json, warnings, sys
warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8","utf_8"):
    import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, log_loss

df  = pd.read_parquet("feature_matrix.parquet")
tr  = df[df["split"]=="train"].dropna(subset=["home_covers_rl"])
val = df[df["split"]=="val"].dropna(subset=["home_covers_rl"])
y_tr = tr["home_covers_rl"].astype(float)
y_v  = val["home_covers_rl"].astype(float)

XP = dict(tree_method="hist", device="cuda", random_state=42,
          learning_rate=0.04, max_depth=5, min_child_weight=20,
          subsample=0.80, colsample_bytree=0.75,
          reg_alpha=0.5, reg_lambda=2.0, n_estimators=600, verbosity=0)

# v1 baseline
v1 = [c for c in json.load(open("models/feature_cols_v1.json")) if c in df.columns]
print(f"Training v1 baseline ({len(v1)} features)...", flush=True)
m1 = XGBClassifier(**XP)
m1.fit(tr[v1].astype(float).fillna(-999), y_tr)
p1 = m1.predict_proba(val[v1].astype(float).fillna(-999))[:,1]
a1_rl = roc_auc_score(y_v, p1)
l1_rl = log_loss(y_v, p1)

# v1 ML
y_tr_ml = tr["actual_home_win"].fillna(0).astype(float)
y_v_ml  = val["actual_home_win"].fillna(0).astype(float)
m1ml = XGBClassifier(**XP)
m1ml.fit(tr[v1].astype(float).fillna(-999), y_tr_ml)
p1ml = m1ml.predict_proba(val[v1].astype(float).fillna(-999))[:,1]
a1_ml = roc_auc_score(y_v_ml, p1ml)

# new model from saved NCV predictions
vp   = pd.read_csv("xgb_val_predictions.csv").dropna(subset=["home_covers_rl","rl_prob"])
a2_rl = roc_auc_score(vp["home_covers_rl"], vp["rl_prob"])
l2_rl = log_loss(vp["home_covers_rl"], vp["rl_prob"])
vp2  = pd.read_csv("xgb_val_predictions.csv").dropna(subset=["actual_home_win","ml_prob"])
a2_ml = roc_auc_score(vp2["actual_home_win"], vp2["ml_prob"])

print()
print("=" * 58)
print("  BEFORE vs AFTER  (val=2025, run-line model)")
print("=" * 58)
print(f"  {'Model':<28} {'Feats':>6}  {'AUC':>7}  {'Logloss':>9}")
print(f"  {'-'*54}")
print(f"  {'v1 original':<28} {len(v1):>6}  {a1_rl:.4f}   {l1_rl:.4f}")
print(f"  {'new (Phase2 + QW + NCV)':<28} {140:>6}  {a2_rl:.4f}   {l2_rl:.4f}")
print(f"  {'delta':<28} {'':>6}  {a2_rl-a1_rl:+.4f}   {l2_rl-l1_rl:+.4f}")
print()
print(f"  Moneyline model:")
print(f"  {'v1 original':<28} {len(v1):>6}  {a1_ml:.4f}")
print(f"  {'new (Phase2 + QW + NCV)':<28} {140:>6}  {a2_ml:.4f}")
print(f"  {'delta':<28} {'':>6}  {a2_ml-a1_ml:+.4f}")
print()
print("  NCV 2-fold mean RL AUC (new model): 0.5996")
print("  (v1 NCV baseline not saved -- single-fold comparison above)")
