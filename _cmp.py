
import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings("ignore")
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, log_loss

df   = pd.read_parquet("feature_matrix.parquet")
tr   = df[df["split"]=="train"].dropna(subset=["home_covers_rl"])
val  = df[df["split"]=="val"].dropna(subset=["home_covers_rl"])
y_tr = tr["home_covers_rl"].astype(float)
y_v  = val["home_covers_rl"].astype(float)

XP = dict(tree_method="hist",device="cuda",random_state=42,
          learning_rate=0.04,max_depth=5,min_child_weight=20,
          subsample=0.80,colsample_bytree=0.75,
          reg_alpha=0.5,reg_lambda=2.0,n_estimators=600,verbosity=0)

v1 = [c for c in json.load(open("models/feature_cols_v1.json")) if c in df.columns]
m1 = XGBClassifier(**XP)
m1.fit(tr[v1].fillna(-999), y_tr)
p1 = m1.predict_proba(val[v1].fillna(-999))[:,1]
a1 = roc_auc_score(y_v, p1)
l1 = log_loss(y_v, p1)

vp = pd.read_csv("xgb_val_predictions.csv").dropna(subset=["home_covers_rl","rl_prob"])
a2 = roc_auc_score(vp["home_covers_rl"], vp["rl_prob"])
l2 = log_loss(vp["home_covers_rl"], vp["rl_prob"])

vp2 = pd.read_csv("xgb_val_predictions.csv").dropna(subset=["actual_home_win","ml_prob"])
a2ml = roc_auc_score(vp2["actual_home_win"], vp2["ml_prob"])

print(f"v1 ({len(v1)} feats) RL AUC: {a1:.4f}  logloss: {l1:.4f}")
print(f"new (140 feats) RL AUC: {a2:.4f}  logloss: {l2:.4f}")
print(f"delta RL: {a2-a1:+.4f}  logloss: {l2-l1:+.4f}")
print(f"new (140 feats) ML AUC: {a2ml:.4f}")
