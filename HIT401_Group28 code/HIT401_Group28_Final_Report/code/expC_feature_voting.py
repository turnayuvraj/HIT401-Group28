"""
Feature voting experiment (supervisor suggestion, 9 Oct meeting)

Two feature-ranking methods "vote" on which of the 12 features matter most:
  1. Lasso   - LassoCV run once per class (one-vs-rest, 17 runs),
               exactly as in our lasso_feature_selection.py
  2. ReliefF - our from-scratch ReliefF (2,000 random training signals,
               10 nearest matches), exactly as in relieff_comparison.py
Each method ranks the features (1 = best). We add the two ranks; the
lowest total is the most agreed-on feature. Ties are broken by Lasso rank.
Both are fitted on the TRAINING set only. Neither is one of our
classifiers, so the features are not chosen by the model being tested.

Then RF and SVM are tested on the unseen 20% test set using:
all 12, voted top 7, voted top 5, voted top 3.

STANDALONE: upload the 17 class CSVs into Colab, then paste this whole
file into ONE cell and run it.
It loads the data, builds the 12 features and makes the same 80/20 split
as PQD_Code.py. No CNN is trained and nothing extra needs installing.
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LassoCV
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# The 17 class CSVs uploaded straight into Colab (same as PQD_Code.py)
DATA_DIR = "/content"
SAVE_DIR = "/content/voting_outputs"   # download these before closing Colab
os.makedirs(SAVE_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
TOP_K = [7, 5, 3]

COMPOUND_CLASSES = {
    "Flicker_with_Sag", "Flicker_with_Swell", "Harmonics_with_Sag",
    "Harmonics_with_Swell", "Sag_with_Harmonics", "Sag_with_Oscillatory_Transient",
    "Swell_with_Harmonics", "Swell_with_Oscillatory_Transient",
}
FEATURE_NAMES = [
    "mean", "std", "min", "max", "peak_to_peak", "rms",
    "skewness", "kurtosis", "energy", "zero_crossing_count",
    "mean_abs_diff", "median",
]

# ---------------------------------------------------------------------
# 1. Load data (same way as PQD_Code.py)
# ---------------------------------------------------------------------
paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
if len(paths) != 17:
    raise FileNotFoundError(f"Expected 17 class CSVs in {DATA_DIR}, found {len(paths)}. "
                            "Fix DATA_DIR at the top of this cell.")
signals, labels = [], []
for path in paths:
    label = os.path.splitext(os.path.basename(path))[0]
    data = pd.read_csv(path, header=None)
    signals.append(data.values.astype(np.float32))
    labels.extend([label] * len(data))
X_raw = np.vstack(signals)
y = np.array(labels)
classes = sorted(np.unique(y))
print(f"Loaded {X_raw.shape[0]} signals, {len(classes)} classes")

# ---------------------------------------------------------------------
# 2. The 12 features (identical to PQD_Code.py)
# ---------------------------------------------------------------------
def extract_features(signal):
    signal = np.asarray(signal, dtype=float)
    lo, hi = np.min(signal), np.max(signal)
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    return [np.mean(signal), np.std(signal), lo, hi, hi - lo,
            np.sqrt(np.mean(signal ** 2)), skew(signal), kurtosis(signal),
            np.sum(signal ** 2), len(zero_crossings),
            np.mean(np.abs(np.diff(signal))), np.median(signal)]

X_feat = np.array([extract_features(s) for s in X_raw])

# ---------------------------------------------------------------------
# 3. Same 80/20 split as PQD_Code.py
# ---------------------------------------------------------------------
train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=TEST_SIZE,
                                       random_state=RANDOM_STATE, stratify=y)
X_tr, X_te = X_feat[train_idx], X_feat[test_idx]
y_tr, y_te = y[train_idx], y[test_idx]
print(f"Split: {len(train_idx)} train / {len(test_idx)} test")

scaler_rank = StandardScaler().fit(X_tr)      # Lasso and ReliefF need scaled features
X_tr_scaled = scaler_rank.transform(X_tr)
y_tr_num = LabelEncoder().fit(classes).transform(y_tr)


def to_ranks(scores):
    """Highest score gets rank 1."""
    order = np.argsort(-np.asarray(scores))
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks

# ---------------------------------------------------------------------
# 4. Voter 1 - Lasso (same method as our original Lasso run)
# For each class: label signals 1 if they are that class, 0 if not,
# fit LassoCV, add up the size of each feature's coefficient.
# ---------------------------------------------------------------------
print("\nVoter 1: Lasso (17 runs, one per class - 1-2 minutes)")
lasso_score = np.zeros(12)
for class_idx in range(len(classes)):
    y_binary = (y_tr_num == class_idx).astype(float)
    lasso = LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=5000)
    lasso.fit(X_tr_scaled, y_binary)
    lasso_score += np.abs(lasso.coef_)
lasso_score = lasso_score / lasso_score.sum() * 100        # % contribution

# ---------------------------------------------------------------------
# 5. Voter 2 - ReliefF (same from-scratch version as our original run)
# For 2,000 random training signals: features that differ a lot from the
# 10 nearest signals of OTHER classes score up, features that differ from
# the 10 nearest signals of the SAME class score down.
# ---------------------------------------------------------------------
print("Voter 2: ReliefF (1-2 minutes)")
n_samples, n_features = X_tr_scaled.shape
n_classes = len(classes)
relief_score = np.zeros(n_features)
k = 10
class_priors = np.bincount(y_tr_num) / len(y_tr_num)
rng = np.random.RandomState(42)
n_iter = min(2000, n_samples)
sample_idx = rng.choice(n_samples, size=n_iter, replace=False)

for idx in sample_idx:
    xi, ci = X_tr_scaled[idx], y_tr_num[idx]
    diffs = X_tr_scaled - xi
    dists = np.sqrt((diffs ** 2).sum(axis=1))
    dists[idx] = np.inf
    same = (y_tr_num == ci); same[idx] = False
    hit_idx = np.where(same)[0]
    top_hits = hit_idx[np.argsort(dists[hit_idx])[:k]]
    relief_score -= (np.abs(diffs[top_hits]) / n_iter).sum(axis=0)
    for cj in range(n_classes):
        if cj == ci:
            continue
        miss_idx = np.where(y_tr_num == cj)[0]
        top_miss = miss_idx[np.argsort(dists[miss_idx])[:k]]
        weight = class_priors[cj] / (1 - class_priors[ci])
        relief_score += weight * (np.abs(diffs[top_miss]) / n_iter).sum(axis=0)

# ---------------------------------------------------------------------
# 6. Combine the votes
# ---------------------------------------------------------------------
votes = pd.DataFrame({
    "feature": FEATURE_NAMES,
    "lasso_%": np.round(lasso_score, 2),
    "lasso_rank": to_ranks(lasso_score),
    "relieff_rank": to_ranks(relief_score),
})
votes["total"] = votes["lasso_rank"] + votes["relieff_rank"]
rho, _ = spearmanr(votes["lasso_rank"], votes["relieff_rank"])
votes["top5_votes"] = ((votes[["lasso_rank", "relieff_rank"]] <= 5).sum(axis=1))
votes = votes.sort_values(["total", "lasso_rank"]).reset_index(drop=True)
votes.index = votes.index + 1
votes.index.name = "voted_rank"
print("\nVOTING TABLE (lower total = both methods agree it is important)")
print(votes.to_string())
print(f"\nAgreement between Lasso and ReliefF (Spearman): {rho:.2f}")
print("Check: this should be about 0.78, with std, rms, energy on top for both.")
votes.to_csv(f"{SAVE_DIR}/feature_voting_table.csv")

# ---------------------------------------------------------------------
# 7. Test accuracy with the top-voted features
# Columns kept in ORIGINAL order so "All 12" matches the main results.
# ---------------------------------------------------------------------
feature_sets = {"All 12": FEATURE_NAMES}
for k in TOP_K:
    feature_sets[f"Voted top {k}"] = list(votes["feature"].iloc[:k])

is_c = np.array([c in COMPOUND_CLASSES for c in classes])
rows = []
for set_name, feats in feature_sets.items():
    cols = [i for i, n in enumerate(FEATURE_NAMES) if n in feats]
    Xtr_k, Xte_k = X_tr[:, cols], X_te[:, cols]

    rf_k = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf_k.fit(Xtr_k, y_tr)
    sc_k = StandardScaler().fit(Xtr_k)
    svm_k = SVC(kernel="rbf", C=10, gamma="scale", random_state=RANDOM_STATE)
    svm_k.fit(sc_k.transform(Xtr_k), y_tr)

    for model_name, pred in [("Random Forest", rf_k.predict(Xte_k)),
                             ("SVM", svm_k.predict(sc_k.transform(Xte_k)))]:
        f1 = f1_score(y_te, pred, labels=classes, average=None, zero_division=0)
        rows.append({
            "features": set_name, "n_features": len(cols), "model": model_name,
            "test_accuracy": round(accuracy_score(y_te, pred) * 100, 2),
            "non_compound_f1": round(f1[~is_c].mean(), 4),
            "compound_f1": round(f1[is_c].mean(), 4),
            "gap": round(f1[~is_c].mean() - f1[is_c].mean(), 4),
            "feature_list": ", ".join(n for n in FEATURE_NAMES if n in feats),
        })

results = pd.DataFrame(rows)
print("\nTEST ACCURACY WITH VOTED FEATURES (unseen 20% test set)")
print(results.drop(columns="feature_list").to_string(index=False))
print("\nCheck: 'All 12' should be about RF 90.94% and SVM 76.91% (the main results).")
results.to_csv(f"{SAVE_DIR}/feature_voting_results.csv", index=False)

# ---------------------------------------------------------------------
# 8. Chart for the slides
# ---------------------------------------------------------------------
order = list(feature_sets.keys())
x = np.arange(len(order)); w = 0.38
rf_acc = [results[(results.features == s) & (results.model == "Random Forest")]
          ["test_accuracy"].iloc[0] for s in order]
svm_acc = [results[(results.features == s) & (results.model == "SVM")]
           ["test_accuracy"].iloc[0] for s in order]
fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - w / 2, rf_acc, w, label="Random Forest", color="#065A82")
b2 = ax.bar(x + w / 2, svm_acc, w, label="SVM", color="#E8A33D")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                f"{b.get_height():.1f}%", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(order)
ax.set_ylabel("Test accuracy (%)"); ax.set_ylim(0, 105)
ax.set_title("Accuracy using the features voted most important\n(Lasso + ReliefF voting)")
ax.legend(); plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/feature_voting_accuracy.png", dpi=150)
plt.show()

print(f"\nDone. Saved to {SAVE_DIR}: feature_voting_table.csv, "
      "feature_voting_results.csv, feature_voting_accuracy.png")
