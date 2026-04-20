"""
generate_pdf_v8.py
==================
MLB Pipeline Technical Documentation — Shadow Ensemble v5.1 (April 2026 update)
Covers all new changes: F5 model, enriched feature matrix v2, K-prop tracker,
blend tracker, lock optimizer, catcher framing, updated scheduler.
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\garci\OneDrive\Documents\Claude\mlb_model_pipeline_dan")
OUTPUT = BASE / "MLB_Pipeline_Technical_Documentation.pdf"

# ── chart paths ───────────────────────────────────────────────────────────────
CHARTS = {}

# ── colours ───────────────────────────────────────────────────────────────────
BG      = "#0a0e1a"
NAVY_M  = "#1F497D"
BLUE_M  = "#2E74B5"
TEAL_M  = "#1A7070"
GREEN_M = "#1A7A3A"
GOLD_M  = "#C09010"
RED_M   = "#8B1A1A"
GRAY_M  = "#556070"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — CHARTS
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  MLB Pipeline Documentation v8 — generate_pdf_v8.py")
print("=" * 65)
print("\n[1/2] Generating charts...\n")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": "#0d1221",
    "axes.edgecolor": "#2a3a5a", "axes.labelcolor": "#c0d0e8",
    "xtick.color": "#8090b0", "ytick.color": "#8090b0",
    "text.color": "#c0d0e8", "grid.color": "#1e2d4a",
    "grid.alpha": 0.6, "font.family": "DejaVu Sans",
})

DPI = 150

def save_chart(key, fig, name):
    p = BASE / name
    fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    CHARTS[key] = p
    print(f"    Saved {name}")


# ── 1. AUC curves — Full-Game RL + ML ────────────────────────────────────────
try:
    print("  [1/9] AUC curves (RL + ML)...")
    from sklearn.metrics import roc_curve, auc as sk_auc, roc_auc_score
    ep = pd.read_csv(BASE / "eval_predictions.csv")
    ep = ep[ep["_eval_mode"] == "current"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("ROC Curves — Full-Game Models (Val 2025)", color="white", fontsize=13, fontweight="bold", y=1.01)

    # RL panel
    ax = axes[0]
    ax.set_facecolor("#0d1221")
    rl_cfg = [("rl_raw","home_covers_rl","XGB Raw","#5590ff"),
              ("rl_cal","home_covers_rl","XGB Platt","#44ddaa"),
              ("rl_stacked","home_covers_rl","Bayesian Stacked","#ffcc44")]
    for col, lbl_col, name, c in rl_cfg:
        sub = ep.dropna(subset=[lbl_col, col])
        yt, yp = sub[lbl_col].astype(float), sub[col].astype(float)
        fpr, tpr, _ = roc_curve(yt, yp)
        a = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC={a:.4f})")
    ax.plot([0,1],[0,1],"--", color="#445566", lw=1.2)
    ax.set_title("Run Line (+0.5)", color="white"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(fontsize=8.5, facecolor="#0d1221", edgecolor="#2a3a5a", labelcolor="white")
    ax.grid(True, alpha=0.4)

    # ML panel
    ax = axes[1]
    ax.set_facecolor("#0d1221")
    ml_cfg = [("ml_raw","actual_home_win","XGB Raw","#5590ff"),
              ("ml_cal","actual_home_win","XGB Platt","#44ddaa")]
    for col, lbl_col, name, c in ml_cfg:
        sub = ep.dropna(subset=[lbl_col, col])
        yt, yp = sub[lbl_col].astype(float), sub[col].astype(float)
        fpr, tpr, _ = roc_curve(yt, yp)
        a = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC={a:.4f})")
    ax.plot([0,1],[0,1],"--", color="#445566", lw=1.2)
    ax.set_title("Money Line", color="white"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(fontsize=8.5, facecolor="#0d1221", edgecolor="#2a3a5a", labelcolor="white")
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    save_chart("auc_rl", fig, "chart_auc_rl_v8.png")
except Exception as e:
    print(f"    [WARN] AUC RL chart failed: {e}")

# ── 2. F5 AUC curves ─────────────────────────────────────────────────────────
try:
    print("  [2/9] F5 AUC curves...")
    from sklearn.metrics import roc_curve, auc as sk_auc
    f5 = pd.read_csv(BASE / "f5_val_predictions.csv")
    f5c = f5.dropna(subset=["f5_home_cover","xgb_raw_f5_cover"])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_facecolor("#0d1221")
    fig.patch.set_facecolor(BG)
    f5_cfg = [("xgb_raw_f5_cover","XGB Raw F5","#5590ff"),
              ("xgb_cal_f5_cover","XGB Platt F5","#44ddaa"),
              ("stacker_f5_cover","Bayesian Stacked F5","#ffcc44")]
    yt = f5c["f5_home_cover"].astype(float)
    for col, name, c in f5_cfg:
        yp = f5c[col].astype(float)
        fpr, tpr, _ = roc_curve(yt, yp)
        a = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, color=c, lw=2.2, label=f"{name} (AUC={a:.4f})")
    ax.plot([0,1],[0,1],"--", color="#445566", lw=1.2)
    ax.set_title("ROC Curves — F5 Model (+0.5 Cover, Val 2025)\nTie push rate=15.9%  |  F5 cover rate=61.5%",
                 color="white", fontsize=11)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(fontsize=9, facecolor="#0d1221", edgecolor="#2a3a5a", labelcolor="white")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    save_chart("auc_f5", fig, "chart_auc_f5_v8.png")
except Exception as e:
    print(f"    [WARN] F5 AUC chart failed: {e}")

# ── 3. Monthly AUC ────────────────────────────────────────────────────────────
try:
    print("  [3/9] Monthly AUC...")
    months  = ["Mar","Apr","May","Jun","Jul","Aug","Sep"]
    aucs    = [0.5367, 0.6133, 0.5857, 0.5426, 0.5602, 0.6281, 0.6058]
    ns      = [65,     384,    403,    394,    362,    418,    372]
    colors  = ["#44dd88" if a>=0.58 else ("#ffcc44" if a>=0.54 else "#667788") for a in aucs]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor("#0d1221"); fig.patch.set_facecolor(BG)
    bars = ax.bar(months, aucs, color=colors, width=0.6, edgecolor="#2a3a5a", linewidth=0.8)
    ax.axhline(0.50, color="#ff6655", lw=1.5, ls="--", label="Random (0.50)")
    ax.axhline(np.mean(aucs), color="#88aaff", lw=1.5, ls=":", label=f"Mean ({np.mean(aucs):.4f})")
    for bar, a, n in zip(bars, aucs, ns):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                f"{a:.4f}\nn={n}", ha="center", va="bottom", fontsize=8, color="#c0d0e8")
    ax.set_ylim(0.48, 0.65)
    ax.set_title("Monthly RL AUC — 2025 Validation Season", color="white", fontsize=12, fontweight="bold")
    ax.set_ylabel("AUC-ROC")
    ax.legend(fontsize=9, facecolor="#0d1221", edgecolor="#2a3a5a", labelcolor="white")
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    save_chart("monthly_auc", fig, "chart_monthly_auc_v8.png")
except Exception as e:
    print(f"    [WARN] Monthly AUC chart failed: {e}")

# ── 4. Calibration curves ─────────────────────────────────────────────────────
try:
    print("  [4/9] Calibration curves...")
    ep = pd.read_csv(BASE / "eval_predictions.csv")
    ep = ep[ep["_eval_mode"]=="current"].dropna(subset=["home_covers_rl","rl_raw","rl_stacked"])
    yt = ep["home_covers_rl"].astype(float)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor("#0d1221"); fig.patch.set_facecolor(BG)
    bins = np.linspace(0, 1, 11)
    for col, label, c in [("rl_raw","XGB Raw","#5590ff"),("rl_stacked","Bayesian Stacked","#ffcc44")]:
        yp = ep[col].astype(float)
        mids, rates = [], []
        for i in range(len(bins)-1):
            mask = (yp>=bins[i]) & (yp<bins[i+1])
            if mask.sum() < 5: continue
            mids.append(yp[mask].mean())
            rates.append(yt[mask].mean())
        ax.plot(mids, rates, "o-", color=c, lw=2, ms=6, label=label)
    ax.plot([0,1],[0,1],"--", color="#88aaff", lw=1.5, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Actual Cover Rate")
    ax.set_title("Calibration Curve — RL Model (Val 2025)", color="white", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, facecolor="#0d1221", edgecolor="#2a3a5a", labelcolor="white")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    save_chart("calibration", fig, "chart_calibration_v8.png")
except Exception as e:
    print(f"    [WARN] Calibration chart failed: {e}")

# ── 5. Feature importance ─────────────────────────────────────────────────────
try:
    print("  [5/9] Feature importance...")
    fi = pd.read_csv(BASE / "xgb_feature_importance.csv")
    rl = fi[fi["model"]=="run_line"].sort_values("gain", ascending=False).head(20)

    def feat_color(name):
        n = name.lower()
        if "true_" in n or "prob" in n: return "#ff9944"
        if "bp_" in n or "bullpen" in n: return "#44cccc"
        if "sp_" in n or "xwoba" in n or "k_pct" in n or "velo" in n or "xera" in n: return "#5599ff"
        if "bat" in n or "matchup" in n: return "#55dd88"
        if "circadian" in n or "cluster" in n or "ump" in n or "park" in n: return "#cc88ff"
        return "#778899"

    colors = [feat_color(f) for f in rl["feature"]]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor("#0d1221"); fig.patch.set_facecolor(BG)
    bars = ax.barh(rl["feature"][::-1], rl["gain"][::-1], color=colors[::-1],
                   edgecolor="#2a3a5a", linewidth=0.6)
    ax.set_xlabel("Gain (information gain per split)")
    ax.set_title("Top 20 RL Features by XGBoost Gain", color="white", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.4)
    legend_items = [
        mpatches.Patch(color="#ff9944", label="Vegas/Market"),
        mpatches.Patch(color="#5599ff", label="SP / Statcast"),
        mpatches.Patch(color="#44cccc", label="Bullpen"),
        mpatches.Patch(color="#55dd88", label="Batting/Matchup"),
        mpatches.Patch(color="#cc88ff", label="Park/Ump/Circadian"),
    ]
    ax.legend(handles=legend_items, fontsize=8.5, facecolor="#0d1221",
              edgecolor="#2a3a5a", labelcolor="white", loc="lower right")
    plt.tight_layout()
    save_chart("feat_imp", fig, "chart_feature_importance_v8.png")
except Exception as e:
    print(f"    [WARN] Feature importance chart failed: {e}")

# ── 6. ROI by year ────────────────────────────────────────────────────────────
try:
    print("  [6/9] ROI by year...")
    years  = ["2023","2024","2025","2026 YTD"]
    roi_all= [-1.0,  -0.2,   5.6,  31.1]
    roi_s1 = [-3.6,  -0.6,   2.6,  29.8]
    roi_s2 = [ 1.9,   0.3,   9.2,  32.2]
    n_all  = [1213,  1159,  1119,   166]
    n_s1   = [ 632,   632,   612,    75]
    n_s2   = [ 581,   527,   507,    91]

    x = np.arange(len(years)); w = 0.25
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_facecolor("#0d1221"); fig.patch.set_facecolor(BG)
    b1 = ax.bar(x - w, roi_all, w, label="All Bets",  color="#4466aa", edgecolor="#2a3a5a")
    b2 = ax.bar(x,     roi_s1,  w, label="★ Tier 1",  color="#5599ee", edgecolor="#2a3a5a")
    b3 = ax.bar(x + w, roi_s2,  w, label="★★ Tier 2", color="#ffcc44", edgecolor="#2a3a5a")
    ax.axhline(0, color="#ff6655", lw=1.5, ls="--")

    for bars, ns in [(b1, n_all),(b2, n_s1),(b3, n_s2)]:
        for bar, n in zip(bars, ns):
            h = bar.get_height()
            ypos = h + 0.4 if h >= 0 else h - 2.5
            ax.text(bar.get_x()+bar.get_width()/2, ypos,
                    f"n={n}", ha="center", fontsize=7, color="#8090b0")

    ax.set_xticks(x); ax.set_xticklabels(years, fontsize=10)
    ax.set_ylabel("ROI (%)")
    ax.set_title("Backtest ROI by Year & Tier (−110 juice assumed)", color="white", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, facecolor="#0d1221", edgecolor="#2a3a5a", labelcolor="white")
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    save_chart("roi", fig, "chart_roi_v8.png")
except Exception as e:
    print(f"    [WARN] ROI chart failed: {e}")

# ── 7. NCV results ────────────────────────────────────────────────────────────
try:
    print("  [7/9] NCV results...")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Nested Cross-Validation Results", color="white", fontsize=12, fontweight="bold")

    # AUC panel
    ax = axes[0]
    ax.set_facecolor("#0d1221")
    x = np.arange(2); w = 0.35
    rl_aucs = [0.6056, 0.5869]; ml_aucs = [0.6111, 0.5970]
    ax.bar(x-w/2, rl_aucs, w, color="#5599ff", label="Run Line AUC", edgecolor="#2a3a5a")
    ax.bar(x+w/2, ml_aucs, w, color="#44ddaa", label="Money Line AUC", edgecolor="#2a3a5a")
    ax.axhline(0.50, color="#ff6655", lw=1.2, ls="--")
    for i, (rl, ml) in enumerate(zip(rl_aucs, ml_aucs)):
        ax.text(i-w/2, rl+0.002, f"{rl:.4f}", ha="center", fontsize=9, color="#c0d0e8")
        ax.text(i+w/2, ml+0.002, f"{ml:.4f}", ha="center", fontsize=9, color="#c0d0e8")
    ax.set_xticks(x)
    ax.set_xticklabels(["Fold 1\nTrain 2023/Val 2024","Fold 2\nTrain 2023+24/Val 2025"], fontsize=8.5)
    ax.set_ylabel("AUC-ROC"); ax.set_ylim(0.48, 0.64)
    ax.legend(fontsize=9, facecolor="#0d1221", edgecolor="#2a3a5a", labelcolor="white")
    ax.grid(axis="y", alpha=0.4)

    # MAE panel
    ax = axes[1]
    ax.set_facecolor("#0d1221")
    maes = [3.4247, 3.5778]; rmses = [4.3105, 4.5533]
    ax.bar(x-w/2, maes,  w, color="#cc88ff", label="MAE",  edgecolor="#2a3a5a")
    ax.bar(x+w/2, rmses, w, color="#ff8855", label="RMSE", edgecolor="#2a3a5a")
    for i, (m, r) in enumerate(zip(maes, rmses)):
        ax.text(i-w/2, m+0.02, f"{m:.4f}", ha="center", fontsize=9, color="#c0d0e8")
        ax.text(i+w/2, r+0.02, f"{r:.4f}", ha="center", fontsize=9, color="#c0d0e8")
    ax.set_xticks(x)
    ax.set_xticklabels(["Fold 1","Fold 2"], fontsize=8.5)
    ax.set_ylabel("Runs"); ax.set_title("Total Runs Regression Error", color="white")
    ax.legend(fontsize=9, facecolor="#0d1221", edgecolor="#2a3a5a", labelcolor="white")
    ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    save_chart("ncv", fig, "chart_ncv_v8.png")
except Exception as e:
    print(f"    [WARN] NCV chart failed: {e}")

# ── 8. F5 vs Full-Game comparison ─────────────────────────────────────────────
try:
    print("  [8/9] F5 vs Full-Game comparison...")
    labels = ["RL Raw","RL Platt","RL Stacked\n(Bayesian)","F5 Raw","F5 Platt","F5 Stacked\n(Bayesian)"]
    aucs   = [0.5793, 0.5793, 0.5760, 0.5944, 0.5944, 0.5986]
    colors = ["#4466aa","#4488cc","#ffcc44","#2a6a3a","#44aa55","#88ee44"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor("#0d1221"); fig.patch.set_facecolor(BG)
    bars = ax.bar(labels, aucs, color=colors, edgecolor="#2a3a5a", linewidth=0.8, width=0.6)
    ax.axhline(0.50, color="#ff6655", lw=1.5, ls="--", label="Random baseline")
    ax.axvline(2.5, color="#667788", lw=1.2, ls=":")
    ax.text(0.95, 0.95, "Full-Game Models", transform=ax.transAxes, ha="right",
            color="#8090b0", fontsize=9, style="italic")
    ax.text(0.97, 0.95, "", transform=ax.transAxes)
    for bar, a in zip(bars, aucs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0008,
                f"{a:.4f}", ha="center", fontsize=9.5, color="white", fontweight="bold")
    ax.set_ylim(0.545, 0.615)
    ax.set_title("F5 Model vs Full-Game RL Model — AUC Comparison (Val 2025)",
                 color="white", fontsize=12, fontweight="bold")
    ax.set_ylabel("AUC-ROC")
    ax.annotate("Full-Game RL", xy=(1, 0.552), xycoords=("data","data"),
                color="#7090b0", fontsize=9)
    ax.annotate("F5 Models", xy=(3.2, 0.552), xycoords=("data","data"),
                color="#7090b0", fontsize=9)
    ax.legend(fontsize=9, facecolor="#0d1221", edgecolor="#2a3a5a", labelcolor="white")
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    save_chart("f5_vs_rl", fig, "chart_f5_vs_rl_v8.png")
except Exception as e:
    print(f"    [WARN] F5 vs RL chart failed: {e}")

# ── 9. K-prop tracker ─────────────────────────────────────────────────────────
try:
    print("  [9/9] K-prop tracker...")
    kp = pd.read_csv(BASE / "kprop_tracker_2026.csv")
    kp_clean = kp.dropna(subset=["model_correct"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("K-Prop Tracker — 2026 YTD (N=53)", color="white", fontsize=12, fontweight="bold")

    # Accuracy donut
    ax = axes[0]
    ax.set_facecolor("#0d1221")
    correct = int(kp_clean["model_correct"].sum())
    total   = len(kp_clean)
    wrong   = total - correct
    acc     = correct / total
    wedges, _ = ax.pie([correct, wrong],
                        colors=["#44dd88","#446688"],
                        startangle=90, counterclock=False,
                        wedgeprops=dict(width=0.55, edgecolor="#0d1221", linewidth=2))
    ax.text(0, 0, f"{acc:.1%}\nAccuracy", ha="center", va="center",
            fontsize=13, fontweight="bold", color="white")
    ax.set_title(f"Model Accuracy\n({correct}/{total} correct)", color="white", fontsize=10)

    # Edge distribution
    ax = axes[1]
    ax.set_facecolor("#0d1221")
    if "edge" in kp_clean.columns:
        edges = kp_clean["edge"].dropna()
        ax.hist(edges, bins=15, color="#5599ff", edgecolor="#0d1221", linewidth=0.6)
        ax.axvline(edges.mean(), color="#ffcc44", lw=2, ls="--",
                   label=f"Mean edge = {edges.mean():.3f}")
        ax.set_xlabel("Model Edge (P_model − P_implied)")
        ax.set_ylabel("Count")
        ax.set_title("K-Prop Edge Distribution", color="white", fontsize=10)
        ax.legend(fontsize=9, facecolor="#0d1221", edgecolor="#2a3a5a", labelcolor="white")
        ax.grid(True, alpha=0.4)

    plt.tight_layout()
    save_chart("kprop", fig, "chart_kprop_v8.png")
except Exception as e:
    print(f"    [WARN] K-prop chart failed: {e}")

print(f"\n  Charts done: {len(CHARTS)} generated\n")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — PDF
# ══════════════════════════════════════════════════════════════════════════════
print("[2/2] Building PDF...\n")

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether, Image,
)
from reportlab.lib.colors import HexColor, white, black
from PIL import Image as PILImage

NAVY   = HexColor("#1F497D")
BLUE   = HexColor("#2E74B5")
TEAL   = HexColor("#1A7070")
GREEN  = HexColor("#17602E")
RED    = HexColor("#8B0000")
AMBER  = HexColor("#C07800")
LGRAY  = HexColor("#F2F4F7")
MGRAY  = HexColor("#CCCCCC")
DGRAY  = HexColor("#444444")
GOLD   = HexColor("#B08800")
WHITE  = white
BLACK  = black

PAGE_W, PAGE_H = letter
M_L = M_R = 0.85 * inch
M_T = M_B = 0.75 * inch
CW  = PAGE_W - M_L - M_R   # ~6.8 in

# ── styles ────────────────────────────────────────────────────────────────────
_ss = getSampleStyleSheet()
styles = {}

def S(name, **kw):
    s = ParagraphStyle(name, **kw)
    styles[name] = s
    return s

S("h1",    fontName="Helvetica-Bold",  fontSize=14, textColor=NAVY,  spaceAfter=6,  spaceBefore=14, leading=18)
S("h2",    fontName="Helvetica-Bold",  fontSize=12, textColor=BLUE,  spaceAfter=4,  spaceBefore=10, leading=16)
S("h3",    fontName="Helvetica-Bold",  fontSize=10, textColor=TEAL,  spaceAfter=3,  spaceBefore=8,  leading=14)
S("h3g",   fontName="Helvetica-Bold",  fontSize=10, textColor=GREEN, spaceAfter=3,  spaceBefore=8,  leading=14)
S("body",  fontName="Helvetica",       fontSize=9,  textColor=DGRAY, spaceAfter=4,  leading=13)
S("mono",  fontName="Courier",         fontSize=8,  textColor=DGRAY, spaceAfter=3,  leading=12)
S("caption", fontName="Helvetica-Oblique", fontSize=8, textColor=HexColor("#666666"), spaceAfter=2, alignment=TA_CENTER)
S("new_badge", fontName="Helvetica-Bold", fontSize=8, textColor=GREEN, spaceAfter=0)
S("cover_title", fontName="Helvetica-Bold", fontSize=28, textColor=NAVY, alignment=TA_CENTER, leading=34)
S("cover_sub",   fontName="Helvetica-Bold", fontSize=14, textColor=BLUE, alignment=TA_CENTER, spaceAfter=6)
S("cover_body",  fontName="Helvetica",      fontSize=10, textColor=DGRAY, alignment=TA_CENTER, spaceAfter=4)

# ── helpers ───────────────────────────────────────────────────────────────────
def P(text, style="body"): return Paragraph(text, styles[style])
def SP(h=6): return Spacer(1, h)
def HR(): return HRFlowable(width=CW, thickness=0.5, color=MGRAY, spaceAfter=4)
def PB(): return PageBreak()
def NEW(): return Paragraph("&#9679; NEW", styles["new_badge"])

def chart(key, max_w=None, max_h=None, caption=None):
    path = CHARTS.get(key)
    if not path or not path.exists():
        return [P(f"[Chart '{key}' not found]")]
    with PILImage.open(path) as im:
        px_w, px_h = im.size
    asp = px_h / px_w
    if max_w is None and max_h is None: max_w = CW
    if max_w and not max_h:
        w = min(max_w, CW); h = w * asp
    elif max_h and not max_w:
        h = max_h; w = h / asp
    else:
        w = min(max_w, CW); h = w * asp
        if h > max_h: h = max_h; w = h / asp
    elems = [SP(4), Image(str(path), width=w, height=h, hAlign="CENTER")]
    if caption:
        elems += [SP(3), P(caption, "caption")]
    elems.append(SP(8))
    return elems

def make_table(headers, rows, col_widths=None, row_bg=True):
    data = [[Paragraph(f"<b>{h}</b>", ParagraphStyle("th", fontName="Helvetica-Bold",
             fontSize=8.5, textColor=WHITE)) for h in headers]] + \
           [[Paragraph(str(c), ParagraphStyle("td", fontName="Helvetica",
             fontSize=8, textColor=DGRAY, leading=11)) for c in row] for row in rows]
    tbl  = Table(data, colWidths=col_widths, repeatRows=1)
    cmds = [
        ("BACKGROUND",  (0,0), (-1,0), NAVY),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [LGRAY, WHITE] if row_bg else [WHITE]),
        ("GRID",        (0,0), (-1,-1), 0.35, MGRAY),
        ("VALIGN",      (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING",  (0,0), (-1,-1), 5),
        ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING",   (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
    ]
    tbl.setStyle(TableStyle(cmds))
    return tbl

# ── page callback ─────────────────────────────────────────────────────────────
def on_page(canvas, doc):
    w, h = letter
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, h - 0.42*inch, w, 0.42*inch, fill=1, stroke=0)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.setFillColor(WHITE)
    canvas.drawString(0.4*inch, h - 0.28*inch, "THE WIZARD REPORT  |  MLB Pipeline Technical Documentation  |  Shadow Ensemble v5.1")
    canvas.drawRightString(w - 0.4*inch, h - 0.28*inch, f"Page {doc.page}")
    canvas.setStrokeColor(MGRAY)
    canvas.setLineWidth(0.4)
    canvas.line(0.4*inch, 0.45*inch, w - 0.4*inch, 0.45*inch)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(HexColor("#888888"))
    canvas.drawString(0.4*inch, 0.3*inch, "Internal use only  |  garcia.dan24@gmail.com  |  April 2026")
    canvas.restoreState()

# ══════════════════════════════════════════════════════════════════════════════
# CONTENT
# ══════════════════════════════════════════════════════════════════════════════
story = []

# ── COVER ─────────────────────────────────────────────────────────────────────
story += [
    SP(60),
    P("MLB WIZARD PIPELINE", "cover_title"),
    SP(8),
    P("Shadow Ensemble v5.1 — Technical Documentation", "cover_sub"),
    SP(4),
    P("April 20, 2026  ·  Updated with all recent methodology changes", "cover_body"),
    SP(24),
    HR(),
    SP(12),
]

cover_stats = [
    ["9", "Models"],["20", "Pipeline Steps"],["97+","Features"],
    ["50K","MC Trials"],["3-Shot","Scheduler"],["RTX 5080","GPU"],
]
cov_tbl = Table(
    [[Paragraph(f"<b><font size=18 color='#1F497D'>{v}</font></b><br/>"
                f"<font size=8 color='#666666'>{l}</font>", ParagraphStyle("cv", alignment=TA_CENTER))
      for v,l in cover_stats]],
    colWidths=[CW/6]*6
)
cov_tbl.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
story += [cov_tbl, SP(20), HR(), SP(8)]

story += [
    P("<b>What's new in this update (April 2026):</b>", "h3"),
    P("&#9679; <b>F5 (First 5 Innings) Model</b> — complete new XGBoost + Bayesian Hierarchical Stacker for the F5 +0.5 run line market", "body"),
    P("&#9679; <b>Enriched Feature Matrix v2</b> — 8 new feature groups: Lineup wRC+, SP FIP/LOB%, Barrel%, Run Diff, Team Metrics, DER, Elo Ratings, Catcher Framing", "body"),
    P("&#9679; <b>K-Prop Tracker</b> — strikeout prediction logging vs PrizePicks lines", "body"),
    P("&#9679; <b>Blend Tracker + Lock Optimizer</b> — rolling grid-search of optimal MC/XGB weights and gate constants", "body"),
    P("&#9679; <b>Expanded Scheduler</b> — now fires at 10:00 AM ET (moved from 11 AM), 20 pipeline steps (was 9)", "body"),
    P("&#9679; <b>Catcher Framing</b> — official Baseball Savant framing run values integrated as feature", "body"),
    PB(),
]

# ── SECTION 1: Executive Summary ──────────────────────────────────────────────
story += [
    P("1. Executive Summary", "h1"), HR(),
    P("The Wizard Report is a pitcher-first MLB prediction pipeline that blends a 50,000-trial Poisson-LogNormal "
      "Monte Carlo simulation with a GPU-accelerated three-model shadow ensemble (XGBoost + LightGBM + CatBoost) "
      "and a Bayesian Hierarchical Level-2 stacker. As of April 2026, the pipeline now runs a dedicated "
      "<b>First 5 Innings (F5)</b> model in addition to the full-game run line and money line markets.", "body"),
    P("<b>Key Performance Metrics (Val 2025 / 2026 Live):</b>", "h3"),
]
story.append(make_table(
    ["Metric","Full-Game RL","Full-Game ML","F5 Model"],
    [["AUC (XGB Raw)","0.5793","0.5642","0.5944"],
     ["AUC (Bayesian Stacked)","0.5760","—","0.5986"],
     ["Brier Score (Stacked)","0.2259","0.2453","0.2308"],
     ["NCV Fold 1 AUC","0.6056","0.6111","—"],
     ["NCV Fold 2 AUC","0.5869","0.5970","—"],
     ["2026 YTD ROI (★★)","+ 32.2%  (91 bets)","—","—"],
     ["2026 YTD Win Rate","68.7%  (166 bets)","—","—"]],
    [2.6*inch, 1.4*inch, 1.4*inch, 1.4*inch]
))
story += [
    SP(6),
    P("<b>Key insight:</b> The F5 model (AUC=0.5986) outperforms the full-game RL model (AUC=0.5760) because "
      "F5 outcomes are almost entirely determined by the two starting pitchers — the primary signal the pipeline "
      "models best. Bullpen variance and late-game luck are eliminated.", "body"),
    PB(),
]

# ── SECTION 2: Architecture Overview ──────────────────────────────────────────
story += [
    P("2. Shadow Ensemble Architecture — Overview", "h1"), HR(),
    P("The pipeline trains three tree-based algorithms on the same feature matrix and temporal splits. "
      "Only XGBoost feeds the official Level-2 Bayesian stacker. LightGBM and CatBoost serve as shadow models "
      "whose disagreement with XGBoost measures uncertainty.", "body"),
    SP(4),
    make_table(
        ["Model","Algorithm","Role","GPU","Output File","Size"],
        [["XGBoost RL","Gradient Boosted Trees","OFFICIAL — feeds stacker","CUDA / RTX 5080","models/xgb_rl.json","204 KB"],
         ["XGBoost ML","Gradient Boosted Trees","OFFICIAL — moneyline","CUDA / RTX 5080","models/xgb_ml.json","115 KB"],
         ["XGBoost Total","Quantile Regression","OFFICIAL — runs (3 quantiles)","CUDA / RTX 5080","models/xgb_total.json","248 KB"],
         ["XGBoost F5","Gradient Boosted Trees","OFFICIAL — F5 +0.5 cover","CPU (hist)","models/xgb_f5.json","984 KB"],
         ["LightGBM","Leaf-wise Trees","Shadow — uncertainty","GPU (device=gpu)","models/lgb_shadow.json","1242 KB"],
         ["CatBoost","Symmetric Trees","Shadow — uncertainty","GPU (task_type=GPU)","models/cat_shadow.cbm","26 KB"],
         ["Bayesian Stacker RL","NUTS MCMC (NumPyro/JAX)","Level-2 — official bet signal","JAX GPU","models/stacking_lr_rl.pkl","—"],
         ["Bayesian Stacker F5","NUTS MCMC (NumPyro/JAX)","Level-2 — F5 bet signal","JAX GPU","models/stacking_lr_f5.pkl","—"],
         ["Team F5 XGBoost","Gradient Boosted Trees","F5 team-perspective mirror","CPU (hist)","models/team_f5_model.json","1073 KB"]],
        [1.4*inch, 1.5*inch, 1.6*inch, 0.9*inch, 1.3*inch, 0.6*inch]
    ),
    SP(8),
    P("<b>model_spread</b> = max(xgb_raw, lgbm_raw, cat_raw) − min(xgb_raw, lgbm_raw, cat_raw). "
      "A wide spread (>0.08) signals high model disagreement and typically causes the Three-Part Lock Gate "
      "to reject the bet. A narrow spread (<0.03) indicates all three algorithms agree and boosts conviction.", "body"),
    PB(),
]

# ── SECTION 3: Data Sources ───────────────────────────────────────────────────
story += [
    P("3. Data Sources & Ingestion", "h1"), HR(),
    make_table(
        ["Source","Data","Script","Frequency"],
        [["MLB Stats API","Schedule, lineups, probable SPs","lineup_pull.py","2× daily (today + tomorrow)"],
         ["Statcast (pybaseball)","Pitch-level xwOBA, EV, barrel, spin","statcast_pull_2026.py","Daily append"],
         ["The Odds API","Pinnacle moneylines + run lines","odds_current_pull.py","3× daily"],
         ["Weather API (Open-Meteo)","Temperature, wind speed/direction","weather_pull.py","Daily"],
         ["Baseball Savant","SP exit velo, framing runs","statcast_framing_pull.py","Daily"],
         ["PrizePicks API","K-prop lines, player props","prizepicks_pull.py","Daily — NEW"],
         ["FanGraphs (manual)","Team vs LHP/RHP splits","data/raw/ CSVs","Weekly refresh"],
         ["Baseball Reference","Park factors, umpire history","build_ump_stats.py","Seasonal"]],
        [1.6*inch, 2.0*inch, 1.6*inch, 1.55*inch]
    ),
    SP(8),
    P("All fetched data is cached as <b>Parquet files</b> under <i>data/statcast/</i> with date-stamped "
      "filenames (e.g. <i>odds_current_2026_04_20.parquet</i>). A <b>data_watchdog.py</b> script monitors "
      "file age, row counts, and null rates — alerting if any critical source is stale before the model runs.", "body"),
    PB(),
]

# ── SECTION 4: Enriched Feature Matrix v2 ─────────────────────────────────────
story += [
    P("4. Feature Engineering — Enriched Feature Matrix v2", "h1"), HR(),
    P("The base feature matrix (<i>feature_matrix.parquet</i>) contains the original 97 features across "
      "9 groups. <b>enrich_feature_matrix_v2.py</b> adds 8 new feature groups on top, producing "
      "<i>feature_matrix_enriched_v2.parquet</i> (7.6 MB). All new features are leakage-safe: "
      "prior-year or rolling pre-game statistics only.", "body"),
    SP(6),
    P("<b>Base Feature Groups (original 97 features):</b>", "h3"),
    make_table(
        ["Group","Count","Key Features"],
        [["Calendar","2","game_month, game_day_of_week"],
         ["SP EWMA (6 spans)","38","k_pct, bb_pct, xwOBA, gb_pct, xrv, velo, arm_angle, whiff_pctl, xera_pctl"],
         ["SP 10-day Trailing","6","k_pct_10d, bb_pct_10d, xwOBA_10d (home + away)"],
         ["SP Differentials","9","k_pct_diff, xwOBA_diff, xrv_diff, velo_diff, kminusbb_diff, 10d diffs"],
         ["Team Batting","16","xwOBA vs LHP/RHP, K%, BB%, GB%, FB% (home + away, 10d)"],
         ["Matchup Edges","6","bat_vs_sp, matchup_edge, matchup_edge_10d (home + away + diff)"],
         ["Bullpen","13","bp_ERA, K/9, BB/9, HR/9, WHIP, GB% (home + away + diff)"],
         ["Park + Ump","5","park_factor, ump_k_above_avg, ump_bb_above_avg, ump_rpg_above_avg"],
         ["Market + Circadian","7","true_home_prob, line_move, sharp_flag, circadian_edge, days_rest"],
         ["Clusters + IL","6","off_cluster, bp_cluster (KMeans), il_return_flag, starts_since_il"],
         ["MC Residual","1","mc_expected_runs (from Monte Carlo — NaN in historical training)"]],
        [1.8*inch, 0.7*inch, 3.8*inch]
    ),
    SP(10),
    P("NEW  ·  <b>Enriched Feature Groups (v2 additions):</b>", "h3"),
]

v2_groups = [
    ("Group 1","Lineup wRC+",
     "Per-game actual lineup quality scores from confirmed batting orders.\n"
     "home_lineup_wrc_plus, away_lineup_wrc_plus, lineup_wrc_plus_diff\n"
     "home_lineup_xwoba_vs_rhp/lhp, away_lineup_xwoba_vs_rhp/lhp\n"
     "Source: lineup_quality_YYYY-MM-DD.parquet (daily) + annual backfill.\n"
     "Fill: season-median per team for games where lineup not yet confirmed."),
    ("Group 2","SP FIP and LOB%",
     "Fielding Independent Pitching and Left-On-Base% computed from Statcast pitch events.\n"
     "FIP = (13×HR + 3×(BB+HBP) − 2×K) / IP + 3.15\n"
     "LOB% = (H + BB + HBP − R) / (H + BB + HBP − 1.4×HR)\n"
     "home_sp_fip, away_sp_fip, sp_fip_diff, home_sp_lob_pct, away_sp_lob_pct\n"
     "Leakage-safe: 2025 game uses 2024 FIP stats. Min 5.0 IP to qualify."),
    ("Group 3","SP Barrel% Against",
     "Hard contact rate allowed by each starting pitcher.\n"
     "home_sp_barrel_pct, away_sp_barrel_pct, sp_barrel_diff\n"
     "Source: pitcher_exitvelo_YYYY.parquet (Baseball Savant exit velo leaderboard).\n"
     "Higher barrel% = worse pitcher quality (more solid contact allowed)."),
    ("Group 4","Rolling Run Diff + Pythagorean Win%",
     "Team-level 30-game rolling run quality metrics.\n"
     "home_run_diff_30d, away_run_diff_30d, run_diff_30d_diff\n"
     "home_pyth_win_pct, away_pyth_win_pct, pyth_win_pct_diff\n"
     "Pythagorean Win% = RS² / (RS² + RA²) — better signal than win% for true team quality."),
    ("Group 5","Team Rolling Offensive Metrics",
     "30-day trailing team-level Statcast offensive output.\n"
     "home_team_xwoba_30d, away_team_xwoba_30d, team_xwoba_30d_diff\n"
     "home_team_barrel_pct_30d, away_team_barrel_pct_30d\n"
     "Captures hot/cold offensive streaks and lineup composition changes."),
    ("Group 6","DER — Defensive Efficiency Rate",
     "Fraction of balls in play converted to outs.\n"
     "DER = (BIP − Hits_on_BIP) / BIP\n"
     "home_der, away_der, der_diff\n"
     "Computed from Statcast events, 30-game rolling window, look-ahead free.\n"
     "Higher DER = better defense; directly impacts pitcher ERA-vs-FIP gap."),
    ("Group 7","Elo Ratings",
     "Standard Elo rating system updated after every game result.\n"
     "home_elo, away_elo, elo_diff\n"
     "K-factor=20, starting rating=1500 for all teams.\n"
     "Pre-game rating used for features (no leakage). Tracks relative team strength."),
    ("Group 8","Catcher Framing",
     "Official Baseball Savant catcher framing run values (Extra Strikes × Run Value).\n"
     "home_framing_runs, away_framing_runs, framing_diff\n"
     "Source: statcast_framing_pull.py → catcher_framing_YYYY.parquet\n"
     "Team-level aggregate (min 50 innings caught). Framing is stable, SP-independent."),
]

for gnum, gname, gdesc in v2_groups:
    story += [
        P(f"<b>{gnum}: {gname}</b>", "h3g"),
        P(gdesc.replace("\n","<br/>"), "body"),
        SP(2),
    ]

story.append(PB())

# ── SECTION 5: Monte Carlo ────────────────────────────────────────────────────
story += [
    P("5. Monte Carlo Simulation", "h1"), HR(),
    P("The Monte Carlo engine (<i>monte_carlo_runline.py</i>) runs 50,000 independent game simulations "
      "per matchup using a <b>Poisson-LogNormal copula</b> model. GPU acceleration via CuPy on the "
      "RTX 5080 provides ~40× speedup vs NumPy; the code transparently falls back to NumPy when CuPy "
      "is unavailable.", "body"),
    SP(6),
    P("<b>Bivariate Scoring Model:</b>", "h3"),
    P("H | V_h ~ Poisson(μ_h · V_h) &nbsp;&nbsp; A | V_a ~ Poisson(μ_a · V_a)", "mono"),
    P("[log V_h, log V_a] ~ BVN(−σ²/2 · 1, &nbsp; σ² · [[1, ρ], [ρ, 1]])", "mono"),
    SP(4),
    make_table(
        ["Parameter","Value","Interpretation"],
        [["N_SIMS","50,000","Independent trials per game"],
         ["XWOBA_INTERCEPT / SLOPE","−3.0 / +24.0","R/G_allowed = −3.0 + 24×xwOBA (clipped 0–20)"],
         ["XGB_BLEND_WEIGHT","0.40 (MC) / 0.60 (XGB)","Final P = 0.40×mc_prob + 0.60×xgb_prob"],
         ["σ_NB (log-normal SD)","0.50","Overdispersion equivalent to NB dispersion r≈4"],
         ["ρ_COPULA","0.14","Gaussian copula → run-level Corr(H,A) ≈ 0.07"],
         ["INNINGS_SP","5.5","Expected SP innings before bullpen entry"],
         ["Air density multiplier","1 + (elev_ft/5200)×0.35","Coors Field (5200 ft) ≈ +35% run scoring"],
         ["Temperature factor","0.3%/°F above 72°F","Heat increases run scoring"],
         ["KPROP_NB_R","8","NB dispersion for K-prop Gamma-Poisson mixture"]],
        [2.0*inch, 1.6*inch, 2.7*inch]
    ),
    SP(8),
    P("<b>F5 Monte Carlo Output Features</b> (new, fed into F5 model as physics baseline):", "h3"),
    make_table(
        ["Feature","Definition"],
        [["mc_f5_home_win_pct","P(home team leads after exactly 5 innings)"],
         ["mc_f5_away_win_pct","P(away team leads after exactly 5 innings)"],
         ["mc_f5_tie_pct","P(tied after 5 innings) — a tie is a push on −0.5, a cover on +0.5"],
         ["mc_f5_expected_total","E[total runs scored through 5 innings]"],
         ["mc_f5_home_cover_pct","P(home covers +0.5) = mc_f5_home_win_pct + mc_f5_tie_pct"]],
        [2.4*inch, 3.9*inch]
    ),
    PB(),
]

# ── SECTION 6: XGBoost Training ───────────────────────────────────────────────
story += [
    P("6. XGBoost Training", "h1"), HR(),
    P("XGBoost is trained on the enriched feature matrix using CUDA on the RTX 5080 "
      "(<i>device='cuda'</i>, <i>tree_method='hist'</i>). Three separate models are trained: "
      "Run Line classifier, Moneyline classifier, and Total Runs multi-quantile regressor.", "body"),
    SP(6),
    make_table(
        ["Parameter","XGB Full-Game","XGB F5"],
        [["tree_method","hist","hist"],
         ["device","cuda (RTX 5080)","cpu (hist)"],
         ["learning_rate","0.04","0.04"],
         ["max_depth","5","4  (shallower — less data)"],
         ["min_child_weight","20","15"],
         ["subsample","0.80","0.80"],
         ["colsample_bytree","0.75","0.70"],
         ["reg_alpha","0.50","0.10"],
         ["reg_lambda","2.0","1.50"],
         ["gamma","0 (default)","0.05"],
         ["n_estimators","600","600"],
         ["early_stopping_rounds","40 (on AUC)","40 (on logloss)"]],
        [2.4*inch, 2.2*inch, 2.2*inch]
    ),
    SP(8),
    P("<b>Sample Weighting:</b> Two multipliers stack multiplicatively.", "h3"),
    make_table(
        ["Layer","Factor","Rationale"],
        [["Year-decay 2023","0.70×","Old roster identities (BAL, PIT, LAA regression)"],
         ["Year-decay 2024","1.00×","Baseline"],
         ["Year-decay 2025","1.50×","Recent form dominates true team quality signal"],
         ["Year-decay 2026 (F5)","2.00×","Live-season data highly relevant for F5 model"],
         ["ABS Statcast rows","1.35×","Rows with whiff_pctl + xera_pctl upweighted"],
         ["2025 ABS row (combined)","2.025×","1.50 × 1.35 = maximum influence"]],
        [2.0*inch, 0.9*inch, 3.4*inch]
    ),
    SP(8),
    P("<b>Multi-Quantile Total Runs Regression:</b> The total runs model produces three outputs per "
      "game: floor (10th percentile), median (50th percentile), and ceiling (90th percentile). "
      "Objective: <i>reg:quantileerror</i>, <i>multi_strategy='multi_output_tree'</i> "
      "— GPU-efficient (one tree per round outputs all 3 quantiles simultaneously).", "body"),
    PB(),
]

# ── SECTION 7: Shadow Models ──────────────────────────────────────────────────
story += [
    P("7. Shadow Models — LightGBM & CatBoost", "h1"), HR(),
    P("LightGBM and CatBoost are trained with the same splits and sample weights as XGBoost but serve "
      "only as variance estimators. Their raw probabilities are never fed into the official stacker "
      "or used for final bet sizing.", "body"),
    SP(6),
    make_table(
        ["Framework","Key Params","GPU Flag","Output"],
        [["LightGBM","n_estimators=600, lr=0.05, num_leaves=63, min_child=20",
          "device='gpu'","lgb_shadow.json (1242 KB), lgbm_rl/ml/total.pkl"],
         ["CatBoost","iterations=600, lr=0.05, depth=6, l2_leaf_reg=3.0",
          "task_type='GPU'","cat_shadow.cbm (26 KB), cat_rl/ml/total.pkl"]],
        [1.1*inch, 3.2*inch, 0.9*inch, 2.1*inch]
    ),
    SP(8),
    P("<b>Uncertainty signal:</b> <i>model_spread = max(xgb_raw, lgbm_raw, cat_raw) − min(...)</i>. "
      "Recorded in each daily card and used qualitatively — bets with spread > 0.08 are flagged as "
      "high-uncertainty and typically fail the sanity gate.", "body"),
    PB(),
]

# ── SECTION 8: Platt Calibration ──────────────────────────────────────────────
story += [
    P("8. Platt (Sigmoid) Calibration", "h1"), HR(),
    P("Raw XGBoost probabilities are systematically over-confident (compressed toward 0.5). "
      "Isotonic regression was tested but overfits on the ~2400-game validation set. "
      "A <b>sigmoid (Platt) calibrator</b> with only 2 parameters avoids overfit while "
      "substantially reducing Brier score.", "body"),
    SP(6),
    make_table(
        ["Metric","RL Raw","RL Platt","Improvement"],
        [["Brier Score","0.2429","0.2255","−7.2%"],
         ["Log-Loss","0.6789","0.6428","−5.3%"],
         ["Accuracy@50%","0.5771","0.6418","+11.2%"],
         ["AUC (unchanged)","0.5793","0.5793","rank-order preserved"]],
        [2.0*inch, 1.4*inch, 1.4*inch, 1.5*inch]
    ),
    SP(6),
    P("Calibrator: <i>LogisticRegression(C=1e10, solver='lbfgs', max_iter=500)</i> "
      "fit on pooled OOF raw probabilities. Stored as <i>models/calibrator_rl.pkl</i> "
      "and <i>models/calibrator_ml.pkl</i>.", "body"),
    PB(),
]

# ── SECTION 9: Bayesian Hierarchical Stacker ──────────────────────────────────
story += [
    P("9. Bayesian Hierarchical Level-2 Stacker", "h1"), HR(),
    P("The Level-2 model blends the calibrated XGBoost probability with 12 domain features using "
      "a <b>Bayesian hierarchical model fitted via NUTS MCMC</b> (NumPyro / JAX). The model uses "
      "partial pooling across 4 SP handedness segments — rare matchups (LvL) borrow strength from "
      "the global mean rather than overfitting on small samples.", "body"),
    SP(6),
    P("<b>Generative model:</b>", "h3"),
    P("y_i ~ Bernoulli( σ(α + β·logit(p_xgb,i) + δ_{j(i)} + γᵀ·x_i) )", "mono"),
    SP(4),
    make_table(
        ["Parameter","Prior","Interpretation"],
        [["α (intercept)","Normal(0, 1)","Global log-odds offset"],
         ["β (XGB weight)","Normal(1, 0.5)","Trust XGB by default; data can shift ±1 unit"],
         ["σ_δ (segment scale)","HalfCauchy(0, 1)","Polson-Scott hyperprior; allows large group effects"],
         ["δ_j (segment offset)","Normal(0, σ_δ) [partial pooling]","Per-handedness log-odds adjustment"],
         ["γ (domain weights)","Normal(0, 0.3) per feature","Weakly regularised; σ≈3× typical effect size"]],
        [0.9*inch, 1.8*inch, 3.6*inch]
    ),
    SP(8),
    make_table(
        ["Segment j","Encoding","Description","Approx. Freq."],
        [["0 = LvL","home_R=0, away_R=0","Both LHP starters","~4%"],
         ["1 = LvR","home_R=0, away_R=1","Home LHP vs Away RHP","~18%"],
         ["2 = RvL","home_R=1, away_R=0","Home RHP vs Away LHP","~18%"],
         ["3 = RvR","home_R=1, away_R=1","Both RHP starters (most common)","~60%"]],
        [0.9*inch, 1.5*inch, 2.5*inch, 1.4*inch]
    ),
    SP(8),
    P("<b>STACKING_FEATURES (12 inputs to γᵀ·x):</b>", "h3"),
    P("sp_k_pct_diff, sp_xwoba_diff, sp_kminusbb_diff, bp_era_diff, bp_whip_diff, "
      "batting_matchup_edge, home_sp_il_return_flag, away_sp_il_return_flag, "
      "sp_k_pct_10d_diff, sp_xwoba_10d_diff, batting_matchup_edge_10d, "
      "<b>ml_model_vs_vegas_gap</b>  (logit(xgb_ml_raw) − logit(true_home_prob))", "body"),
    P("Posterior means stored as closed-form coefficients in <i>stacking_lr_rl.pkl</i> "
      "(inference = one dot-product, no MCMC at test time). Full posterior trace in "
      "<i>stacking_lr_rl.npz</i> for credible interval analysis.", "body"),
    PB(),
]

# ── SECTION 10: F5 Model ─────────────────────────────────────────────────────
story += [
    P("10. F5 (First 5 Innings) Model  ★ NEW", "h1"), HR(),
    P("The F5 model predicts <b>P(home team covers the +0.5 F5 run line)</b>, i.e. home leads or ties "
      "after exactly 5 innings. A tie is a push on −0.5 but a <i>win</i> on +0.5 — this asymmetry "
      "makes the +0.5 line the preferred F5 market.", "body"),
    SP(6),
    P("<b>Why F5 outperforms full-game RL (AUC 0.5986 vs 0.5760):</b>", "h3"),
    P("Full-game outcomes are contaminated by bullpen performance, manager decisions, and late-inning "
      "luck. F5 outcomes are almost entirely determined by the two starting pitchers — exactly the "
      "signal XGBoost models best. <b>Bullpen features are deliberately excluded</b> from the F5 "
      "model: relief pitchers account for only 5–8% of F5 runs, and in 67% of games both SPs "
      "go the full 5 innings with zero bullpen involvement.", "body"),
    SP(6),
]
story += chart("auc_f5", max_w=5.2*inch,
               caption="Figure: F5 Model ROC Curves — Val 2025 (N=2,436 games). "
                       "Bayesian Stacked AUC=0.5986, tie push rate=15.9%")
story += chart("f5_vs_rl", max_w=CW,
               caption="Figure: F5 vs Full-Game RL AUC comparison — F5 Bayesian Stacker (0.5986) "
                       "outperforms full-game RL equivalent (0.5760) by +226 basis points")
story += [
    SP(4),
    P("<b>F5 Stacking Features</b> (13 domain inputs to BayesianStackerF5):", "h3"),
    make_table(
        ["Feature","Full-Game Equivalent","Change"],
        [["sp_k_pct_diff","Same","—"],
         ["sp_xwoba_diff","Same","—"],
         ["sp_kminusbb_diff","Same","—"],
         ["batting_matchup_edge","Same","—"],
         ["batting_matchup_edge_10d","Same","—"],
         ["home/away_sp_il_return_flag","Same","—"],
         ["sp_k_pct_10d_diff / sp_xwoba_10d_diff","Same","—"],
         ["mc_f5_home_cover_pct","ml_model_vs_vegas_gap","Replaced — F5 physics baseline"],
         ["mc_f5_expected_total","(new)","Low totals → high tie rate → higher cover prob"],
         ["team_f5_log_odds","(new)","logit(p_home)−logit(p_away) tie-safe strength"],
         ["rolling_f5_tie_rate","(new)","30-day observed tie rate: env. base-rate signal"],
         ["bp_era_diff / bp_whip_diff","(excluded)","Not relevant for F5"]],
        [1.9*inch, 2.0*inch, 2.4*inch]
    ),
    SP(6),
    make_table(
        ["F5 Model Metric","XGB Raw","XGB Platt","Bayesian Stacked"],
        [["AUC-ROC","0.5944","0.5944","0.5986"],
         ["Brier Score","0.2316","0.2317","0.2308"],
         ["Log-Loss","0.6554","0.6558","0.6537"],
         ["Accuracy@50%","0.6100","0.6174","0.6112"],
         ["N (Val 2025)","2,436","2,436","2,436"]],
        [2.0*inch, 1.5*inch, 1.5*inch, 1.5*inch]
    ),
    PB(),
]

# ── SECTION 11: Model Performance ────────────────────────────────────────────
story += [P("11. Model Performance & Fit Statistics", "h1"), HR()]
story += chart("auc_rl", max_w=CW,
               caption="Figure: Full-Game ROC curves — Left: Run Line, Right: Money Line (Val 2025, N≈2,400–2,500)")
story += chart("monthly_auc", max_w=CW,
               caption="Figure: Monthly RL AUC — 2025 validation season. April (0.6133) and August (0.6281) strongest.")
story += chart("calibration", max_w=5.0*inch,
               caption="Figure: Calibration curve — Platt scaling substantially reduces overconfidence in the raw probabilities")
story += [
    SP(4),
    P("<b>Full Metrics Table (Val 2025):</b>", "h3"),
    make_table(
        ["Model","AUC","Brier","Log-Loss","Acc@50%","N"],
        [["RL XGB Raw","0.5793","0.2429","0.6789","0.5771","2,398"],
         ["RL Platt Calibrated","0.5793","0.2255","0.6428","0.6418","2,398"],
         ["RL Bayesian Stacked","0.5760","0.2259","0.6436","0.6393","2,398"],
         ["ML XGB Raw","0.5642","0.2477","0.6886","0.5465","2,514"],
         ["ML Platt Calibrated","0.5642","0.2453","0.6836","0.5549","2,514"],
         ["F5 XGB Raw","0.5944","0.2316","0.6554","0.6100","2,436"],
         ["F5 Platt Calibrated","0.5944","0.2317","0.6558","0.6174","2,436"],
         ["F5 Bayesian Stacked","0.5986","0.2308","0.6537","0.6112","2,436"]],
        [2.1*inch, 0.8*inch, 0.8*inch, 0.85*inch, 0.85*inch, 0.7*inch]
    ),
    SP(8),
    P("<b>Nested Cross-Validation (NCV) Results:</b>", "h3"),
]
story += chart("ncv", max_w=CW,
               caption="Figure: NCV fold AUC (left) and total-runs regression error (right). Fold 1 trains on 2023, Fold 2 on 2023+2024.")
story += [
    make_table(
        ["NCV Fold","Train","Validate","n_train","n_val","RL AUC","ML AUC","Total MAE","Total RMSE"],
        [["Fold 1","2023","2024","2,391","2,395","0.6056","0.6111","3.4247","4.3105"],
         ["Fold 2","2023+2024","2025","4,786","2,398","0.5869","0.5970","3.5778","4.5533"]],
        [0.6*inch, 0.85*inch, 0.85*inch, 0.65*inch, 0.65*inch, 0.65*inch, 0.65*inch, 0.8*inch, 0.8*inch]
    ),
    PB(),
]

# ── SECTION 12: Feature Importance ───────────────────────────────────────────
story += [P("12. Feature Importance", "h1"), HR()]
story += chart("feat_imp", max_w=CW,
               caption="Figure: Top 20 RL features by XGBoost gain. Vegas/market signals dominate; bullpen WHIP and Statcast percentiles follow.")
story += [
    SP(4),
    make_table(
        ["Rank","Feature","Gain","Group"],
        [["1","true_home_prob","0.0400","Vegas/Market — Pinnacle implied probability"],
         ["2","true_away_prob","0.0304","Vegas/Market — Pinnacle implied probability"],
         ["3","bp_whip_diff","0.0222","Bullpen — home vs away WHIP differential"],
         ["4","away_bp_bb9","0.0220","Bullpen — away bullpen walk rate"],
         ["5","away_sp_whiff_pctl","0.0168","SP Statcast — away SP whiff percentile (ABS-era)"],
         ["6","batting_matchup_edge_10d","0.0160","Batting — recent 10-day matchup edge"],
         ["7","away_bp_gb_pct","0.0160","Bullpen — away bullpen groundball rate"],
         ["8","circadian_edge","0.0154","Market — travel timezone advantage"],
         ["9","home_sp_k_minus_bb","0.0154","SP EWMA — K% minus BB% (control quality)"],
         ["10","home_bp_whip","0.0152","Bullpen — home bullpen WHIP"],
         ["11","away_sp_fb_velo_pctl","0.0152","SP Statcast — away SP velocity percentile"],
         ["12","home_sp_xera_pctl","0.0150","SP Statcast — home SP expected ERA percentile"]],
        [0.5*inch, 2.2*inch, 0.7*inch, 3.0*inch]
    ),
    PB(),
]

# ── SECTION 13: Three-Part Lock Gate ─────────────────────────────────────────
story += [
    P("13. Three-Part Lock Gate", "h1"), HR(),
    P("Every candidate bet must pass all three gates sequentially. Failing any single gate "
      "causes immediate rejection — the game is logged to Supabase with the rejection reason "
      "but no stake is placed.", "body"),
    SP(6),
    make_table(
        ["Gate","Condition","Current Threshold","Rationale"],
        [["1 — Sanity","|p_model − p_pinnacle| ≤ X","X = 0.04","Rejects: model too far from sharp market"],
         ["2 — Odds Floor","American odds ≥ X","X = −225","Rejects: juice too high to overcome edge"],
         ["3a — Edge Tier 1","model_edge ≥ 3.0%","EDGE_TIER1 = 0.030","Quarter-Kelly stake"],
         ["3b — Edge Tier 2","model_edge ≥ 1.0%","EDGE_TIER2 = 0.010","Eighth-Kelly stake"]],
        [1.4*inch, 2.1*inch, 1.5*inch, 2.3*inch]
    ),
    SP(8),
    P("NEW  ·  <b>Lock Gate Optimizer</b> (<i>optimize_locks.py</i>):", "h3"),
    P("<b>Phase 1</b> — Model calibration: buckets rl_stacked probability against actual cover rate "
      "on 2025 eval_predictions (4,796 games). Identifies probability ranges where the model is "
      "over/under-confident.", "body"),
    P("<b>Phase 2</b> — Lock simulation: grid-searches all four gate constants (sanity threshold, "
      "edge_tier1, edge_tier2, odds_floor) against 2026 live bets and actuals. Reports ROI, "
      "win rate, bet count, and P&amp;L for each grid cell vs the current constants. "
      "Runs automatically as Step 8c in the daily scheduler (--phase 2 --save).", "body"),
    PB(),
]

# ── SECTION 14: Kelly Staking ─────────────────────────────────────────────────
story += [
    P("14. Kelly Staking, Blend Tracker & Backtest ROI", "h1"), HR(),
    make_table(
        ["Constant","Value","Description"],
        [["SYNTHETIC_BANKROLL","$2,000","Notional reference bankroll"],
         ["MAX_BET","$50","Hard cap per bet regardless of Kelly"],
         ["Tier 1 stake","¼ × Kelly × bankroll","For bets with edge ≥ 3.0% (cap $50)"],
         ["Tier 2 stake","⅛ × Kelly × bankroll","For bets with edge 1.0–2.9% (cap $50)"],
         ["Kelly formula","f = (b·p − q) / b","b = decimal odds−1, p = model prob, q = 1−p"]],
        [2.0*inch, 1.3*inch, 3.0*inch]
    ),
    SP(8),
    P("NEW  ·  <b>Blend Tracker</b> (<i>blend_tracker.py</i>):", "h3"),
    P("Joins all 2026 daily card files (<i>daily_cards/daily_card_2026-*.csv</i>) — which contain "
      "both <i>mc_rl</i> (pure Monte Carlo probability) and <i>xgb_rl</i> (pure XGBoost probability) "
      "— with <i>actuals_2026.parquet</i>. Grid-searches blend weights [0.0, 0.1, … 1.0] and "
      "reports the optimal weight by Brier score, log-loss, and ECE. Runs as Step 8c. "
      "Current blend: XGB_BLEND_WEIGHT=0.40 (40% MC, 60% XGB).", "body"),
    SP(8),
    P("<b>Backtest ROI Summary:</b>", "h3"),
]
story += chart("roi", max_w=CW,
               caption="Figure: Backtest ROI by year and tier (−110 juice). 2026 YTD sample size still small (166 bets) but early signal is strong.")
story += [
    make_table(
        ["Year","All Bets","n (All)","★ Tier 1","n (★)","★★ Tier 2","n (★★)","Dominant Signal"],
        [["2023","−1.0%","1,213","−3.6%","632","+1.9%","581","AWAY +1.5 ★★"],
         ["2024","−0.2%","1,159","−0.6%","632","+0.3%","527","AWAY +1.5 ★★"],
         ["2025","+5.6%","1,119","+2.6%","612","+9.2%","507","AWAY +1.5 ★★"],
         ["2026 YTD","+31.1%","166","+29.8%","75","+32.2%","91","AWAY +1.5 ★★ (91 bets)"]],
        [0.75*inch, 0.75*inch, 0.65*inch, 0.75*inch, 0.65*inch, 0.85*inch, 0.65*inch, 1.8*inch]
    ),
    SP(6),
    P("The ★★ Tier 2 signal (lower edge, smaller stakes) has shown the most consistent positive ROI "
      "across all four seasons. 2026 YTD win rate of 69.2% on 91 ★★ bets is well above the "
      "break-even ~52.4% required at −110.", "body"),
    PB(),
]

# ── SECTION 15: K-Prop & PrizePicks ──────────────────────────────────────────
story += [
    P("15. K-Prop Tracker & PrizePicks Integration  ★ NEW", "h1"), HR(),
    P("The pipeline now tracks starting pitcher strikeout predictions against live PrizePicks K lines. "
      "This serves two purposes: (1) direct K-prop betting signal, and (2) model health monitoring "
      "— sustained K-prop accuracy validates that SP Statcast features are capturing true "
      "pitcher form.", "body"),
    SP(6),
    make_table(
        ["Script","Output","Description"],
        [["prizepicks_pull.py","prizepicks_mlb_YYYY-MM-DD.parquet","Daily PrizePicks MLB prop lines including K lines"],
         ["kprop_tracker.py","kprop_tracker_2026.csv","Logs predicted K vs actual K vs PrizePicks line"],
         ["K-prop model","monte_carlo_runline.py","Gamma-Poisson mixture: λ~Γ(r_k, μ/r_k), K~Poisson(λ)"]],
        [1.6*inch, 2.2*inch, 2.5*inch]
    ),
    SP(6),
    P("<b>K-prop prediction columns tracked:</b> sp, line (PrizePicks K line), pred_mean (model expected Ks), "
      "model_over (P(K > line)), implied_over (market implied prob), ump_k_above_avg, "
      "actual_k, hit_over, model_pick_over, model_correct, edge", "body"),
    SP(6),
]
story += chart("kprop", max_w=CW,
               caption="Figure: K-prop tracker — 2026 YTD accuracy and edge distribution (N=53 tracked starts)")
story.append(PB())

# ── SECTION 16: Daily Scheduler ───────────────────────────────────────────────
story += [
    P("16. Daily Scheduler  ★ UPDATED — 10:00 AM ET, 20 Steps", "h1"), HR(),
    P("The scheduler fires at <b>10:00 AM ET</b> (moved from 11 AM) for the full run, then refreshes "
      "at <b>2:00 PM ET</b> and <b>5:00 PM ET</b> to capture late-confirming lineups and closing odds. "
      "The full run now executes 20 ordered steps (was 9 in the prior version).", "body"),
    SP(6),
    make_table(
        ["Step","Script","Purpose","New?"],
        [["1","lineup_pull.py --recent + --date tomorrow","Today's and tomorrow's lineups + probable SPs",""],
         ["2","statcast_pull_2026.py + extract_actuals_2026.py","Append yesterday's Statcast data; update actuals for F5 rolling features","NEW"],
         ["2b","ump_pull.py + build_ump_stats.py","Umpire assignments and K/BB tendencies",""],
         ["2c","build_bullpen_avail.py + build_batter_splits.py","Bullpen availability and per-batter LHP/RHP splits","NEW"],
         ["3","refresh_raw_data.py","Re-pull Baseball Savant + MLB Stats API raw files","NEW"],
         ["3b","build_pitcher_profile.py + build_team_stats_2026.py","Pitcher EWMA profiles and team stats refresh",""],
         ["3c","enrich_feature_matrix_v2.py","Rebuild enriched v2 feature matrix (8 new groups)","NEW"],
         ["4","build_lineup_quality.py","Per-game lineup wRC+ quality scores",""],
         ["4b","weather_pull.py","Game-time temperature and wind for K props + totals","NEW"],
         ["5","odds_current_pull.py","Dual-region (US + EU) Pinnacle moneylines + run lines",""],
         ["5b","prizepicks_pull.py","PrizePicks player prop lines (K lines)","NEW"],
         ["6","run_today.py --csv --email","Score today's games; send email report",""],
         ["7","supabase_upload.py","Upload picks to wizard_daily_card table",""],
         ["8","clv_audit.py","Close out yesterday's picks with closing-line value",""],
         ["8b","kprop_tracker.py","Log yesterday's K predictions vs actuals","NEW"],
         ["8c","supplemental_pull.py + statcast_framing_pull.py","Supplemental data + catcher framing refresh","NEW"],
         ["8d","build_backtest.py --year 2026","Rebuild 2026 backtest file with resolved picks","NEW"],
         ["8e","blend_tracker.py --update","Rolling MC/XGB blend weight optimization","NEW"],
         ["8f","optimize_locks.py --phase 2 --save","Grid-search optimal gate constants vs 2026 live bets","NEW"],
         ["8g","clv_tracker.py --update","Update closing-line value tracking","NEW"],
         ["9","pipeline_health.py --upload","Pipeline health snapshot to Supabase",""]],
        [0.4*inch, 2.5*inch, 3.0*inch, 0.5*inch]
    ),
    SP(8),
    P("<b>Refresh runs (2 PM + 5 PM ET):</b> lineup_pull → build_lineup_quality → odds_current_pull "
      "→ prizepicks_pull → run_today.py → supabase_upload → pipeline_health. "
      "Skips pitcher profiles and team stats (no intraday change) but re-pulls lineups "
      "(late starters confirm after 10 AM) and odds (lines tighten toward game time).", "body"),
    PB(),
]

# ── SECTION 17: Supabase + Script Inventory ───────────────────────────────────
story += [
    P("17. Supabase Schema & Script Inventory", "h1"), HR(),
    P("<b>Supabase Tables:</b>", "h3"),
    make_table(
        ["Table","Contents"],
        [["wizard_daily_card","Today's qualifying picks with model probs, edge, stake, star tier"],
         ["wizard_backtest","Per-bet resolved P&L log with actual outcome"],
         ["wizard_model_history","Daily AUC / Brier score snapshots per model"],
         ["wizard_backtest_historical","Multi-year ROI archive (2023–2026)"],
         ["wizard_pipeline_health","Step-level timing, row counts, error codes per daily run"]],
        [2.2*inch, 4.1*inch]
    ),
    SP(10),
    P("<b>Script Inventory (alphabetical):</b>", "h3"),
    make_table(
        ["Script","Purpose","GPU","Status"],
        [["_compare_baseline.py","Baseline comparison utility","—","NEW"],
         ["_cmp.py","Quick comparison helper","—","NEW"],
         ["analyze_f5_signals.py","F5 signal analysis and diagnostics","—","NEW"],
         ["backfill_lineup_quality.py","Backfill historical wRC+ lineup scores","—","NEW"],
         ["backfill_mc.py","Backfill Monte Carlo F5 features for historical matrix","—","Updated"],
         ["backtest_engine.py","Multi-year ROI simulation, Kelly re-simulation","—","—"],
         ["backtest_f5_2026.py","F5-specific 2026 backtest","—","NEW"],
         ["backtest_historical.py","Historical backtest across seasons","—","—"],
         ["blend_tracker.py","Rolling MC/XGB blend weight optimizer","—","NEW"],
         ["build_backtest.py","Daily 2026 backtest rebuild","—","NEW"],
         ["build_batter_splits.py","Per-batter LHP/RHP splits for 2026","—","NEW"],
         ["build_bullpen_avail.py","Bullpen availability and workload","—","Updated"],
         ["build_feature_matrix.py","Core feature matrix builder (cuDF GPU)","cuDF","Updated"],
         ["build_historical_profiles.py","Historical pitcher profiles backfill","—","—"],
         ["build_lineup_quality.py","Per-game lineup wRC+ quality scores","—","Updated"],
         ["build_pitcher_profile.py","SP EWMA profiles with Statcast metrics","—","—"],
         ["build_team_stats_2026.py","2026 team stats refresh","—","Updated"],
         ["build_ump_stats.py","Umpire K/BB/RPG tendency stats","—","—"],
         ["calibrate_weights.py","EWMA weight sensitivity calibration","—","—"],
         ["clv_audit.py","Closing-line value audit for resolved picks","—","—"],
         ["clv_tracker.py","CLV time-series tracker","—","Updated"],
         ["data_watchdog.py","Data freshness and quality monitoring","—","NEW"],
         ["enrich_feature_matrix.py","Feature matrix v1 enrichment (legacy)","—","—"],
         ["enrich_feature_matrix_v2.py","Feature matrix v2 (8 new groups)","—","NEW"],
         ["evaluate_model.py","Val set evaluation with Bayesian shrinkage option","—","—"],
         ["extract_actuals_2026.py","Extract 2026 game actuals from Statcast","—","—"],
         ["lineup_pull.py","MLB lineup and probable SP pull","—","—"],
         ["mlb_execution_agent.py","Agent-based execution orchestration","—","—"],
         ["monte_carlo_runline.py","MC simulation (CuPy GPU, Poisson-LogNormal)","CuPy","Updated"],
         ["odds_current_pull.py","Dual-region Pinnacle odds pull","—","Updated"],
         ["optimize_blend.py","Offline MC/XGB blend optimization","—","NEW"],
         ["optimize_locks.py","Gate constant grid-search optimizer","—","NEW"],
         ["pipeline_health.py","Pipeline health snapshot to Supabase","—","Updated"],
         ["prizepicks_pull.py","PrizePicks daily MLB prop lines","—","NEW"],
         ["pull_odds_history_api.py","Historical odds API pull","—","—"],
         ["refresh_raw_data.py","Savant + MLB Stats API raw data refresh","—","NEW"],
         ["run_daily_scheduler.py","APScheduler cron — 10AM/2PM/5PM ET","—","Updated"],
         ["run_today.py","Score today's games; build and email report","—","Updated"],
         ["score_f5_today.py","Score today's F5 model predictions","—","NEW"],
         ["score_models.py","Legacy MF1i/MF3i/MF5i/MFull/M3 scoring","—","—"],
         ["statcast_framing_pull.py","Catcher framing from Baseball Savant","—","NEW"],
         ["statcast_pull_2026.py","Daily Statcast pitch data append","—","—"],
         ["supabase_upload.py","5-table Supabase upsert","—","Updated"],
         ["supplemental_pull.py","Supplemental data pull (park, ump, misc)","—","Updated"],
         ["train_f5_model.py","F5 XGBoost + Bayesian Stacker training","CUDA","NEW"],
         ["train_xgboost.py","Full-game XGBoost + shadow models + Bayesian Stacker","CUDA","Updated"],
         ["tracker_server.py","Local tracker HTTP server","—","—"],
         ["ump_pull.py","Umpire assignment pull","—","Updated"],
         ["weather_pull.py","Game-time weather pull (Open-Meteo)","—","Updated"]],
        [2.2*inch, 3.3*inch, 0.55*inch, 0.75*inch]
    ),
    PB(),
]

# ── build doc ──────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    str(OUTPUT),
    pagesize=letter,
    leftMargin=M_L, rightMargin=M_R,
    topMargin=M_T + 0.35*inch,
    bottomMargin=M_B + 0.25*inch,
)
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)

sz = OUTPUT.stat().st_size // 1024
print(f"\nSaved: {OUTPUT}  ({sz} KB)")
print("Done.")
