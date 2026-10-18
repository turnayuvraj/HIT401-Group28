"""
HIT401 Capstone - Group 28
Power Quality Monitoring and Analysis Based on Data Analytics and AI

Sujwal Adhikari, Abishek Kandel, Yuvraj Singh, Anjana Thapa Magar
Supervisor: Dr. Sami Azam

We compare 4 ways of classifying the 17 IEEE 1159 disturbance classes:
the old threshold method, SVM, Random Forest and a 1D CNN.

The whole point is that all 4 get tested on exactly the same test set,
otherwise you can't really say one is better than another.

Dataset is SEED from Kaggle - 17000 signals, 17 classes, 1000 each,
100 samples per signal (5 kHz, so 20 ms = one 50 Hz cycle).


"""

import os
import glob
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, pearsonr

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

import tensorflow as tf
from tensorflow.keras import layers, models


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

DATA_DIR = "/content"
OUTPUT_DIR = "/content/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Fixing the seeds so we get the same numbers every time. Needed this
# for the report - the CNN was giving a different answer on every run
# before we set these.
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

# The 8 classes with two disturbances happening at once. The other 9
# are "non-compound" - note this includes Pure_Sinusoidal which isn't
# really a disturbance at all, it's the normal waveform. That's why we
# say non-compound vs compound in the report and not single vs compound.
COMPOUND_CLASSES = {
    "Flicker_with_Sag",
    "Flicker_with_Swell",
    "Harmonics_with_Sag",
    "Harmonics_with_Swell",
    "Sag_with_Harmonics",
    "Sag_with_Oscillatory_Transient",
    "Swell_with_Harmonics",
    "Swell_with_Oscillatory_Transient",
}


# ---------------------------------------------------------------------
# 1. Load the data
# Each CSV is one class and the filename is the label, so we can just
# read them all and build the label array from the file names.
# ---------------------------------------------------------------------

print("Loading dataset")

signals, labels = [], []

for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.csv"))):
    label = os.path.splitext(os.path.basename(path))[0]
    data = pd.read_csv(path, header=None)
    signals.append(data.values.astype(np.float32))
    labels.extend([label] * len(data))

X_raw = np.vstack(signals)
y = np.array(labels)
classes = sorted(np.unique(y))

n_compound = sum(c in COMPOUND_CLASSES for c in classes)
print(f"  {X_raw.shape[0]} signals, {len(classes)} classes, "
      f"{X_raw.shape[1]} samples each")
print(f"  {len(classes) - n_compound} non-compound, {n_compound} compound")


# ---------------------------------------------------------------------
# 2. Train/test split
#
# IMPORTANT - we split the INDEX array once and reuse it for all four
# methods. Our first version called train_test_split separately for the
# feature models and the CNN, which meant they weren't actually being
# tested on the same signals. Fixed now. Everything below uses train_idx
# and test_idx, nothing re-splits.
# ---------------------------------------------------------------------

train_idx, test_idx = train_test_split(
    np.arange(len(y)),
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

y_train, y_test = y[train_idx], y[test_idx]
print(f"\nSplit: {len(train_idx)} train / {len(test_idx)} test (80/20, stratified)")


# ---------------------------------------------------------------------
# 3. Threshold baseline - the "old way" we're comparing against
#
# Simplified version of IEEE 1159 magnitude monitoring. Takes the RMS of
# each signal, divides by nominal, and applies fixed bands.
#
# We can't do the duration part of the standard because each signal is
# only one cycle long, so this is magnitude only. Said so in the report.
#
# Big thing to notice: this rule can only ever output 4 labels out of 17.
# So it's capped at about 23.5% no matter what.
# ---------------------------------------------------------------------

print("\nThreshold baseline")

# Only using training-set normal signals for the reference, otherwise
# we'd be leaking test data into the baseline.
normal_train = X_raw[train_idx][y_train == "Pure_Sinusoidal"]
nominal_rms = np.sqrt(np.mean(normal_train ** 2, axis=1)).mean()
print(f"  Nominal RMS reference: {nominal_rms:.4f}")


def threshold_classify(rms_ratio):
    """IEEE 1159 magnitude bands."""
    if rms_ratio < 0.10:
        return "Interruption"
    elif rms_ratio < 0.90:
        return "Sag"
    elif rms_ratio > 1.10:
        return "Swell"
    else:
        return "Pure_Sinusoidal"


rms_test = np.sqrt(np.mean(X_raw[test_idx] ** 2, axis=1))
pred_threshold = np.array([threshold_classify(r / nominal_rms) for r in rms_test])

# Working out the ceiling - what's the best it could possibly get if it
# classified all four of its reachable labels perfectly.
reachable = {"Interruption", "Sag", "Swell", "Pure_Sinusoidal"}
ceiling = np.isin(y_test, list(reachable)).mean() * 100
print(f"  Structural ceiling: {ceiling:.2f}% (4 of 17 labels reachable)")


# ---------------------------------------------------------------------
# 4. Feature extraction (RF and SVM only)
#
# 12 basic statistical features. The CNN skips all of this and reads the
# raw waveform instead - that's the main difference we're testing.
#
# Worth noting these are all computed over the WHOLE window, so none of
# them know where in the signal something happened. This turns out to
# matter a lot for the compound classes (see the discussion in the report).
# ---------------------------------------------------------------------

FEATURE_NAMES = [
    "mean", "std", "min", "max", "peak_to_peak", "rms",
    "skewness", "kurtosis", "energy", "zero_crossing_count",
    "mean_abs_diff", "median",
]


def extract_features(signal):
    """The 12 features for one signal, in the order of FEATURE_NAMES."""
    signal = np.asarray(signal, dtype=float)

    lo, hi = np.min(signal), np.max(signal)
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]

    return [
        np.mean(signal),                      # mean
        np.std(signal),                       # std
        lo,                                   # min
        hi,                                   # max
        hi - lo,                              # peak to peak
        np.sqrt(np.mean(signal ** 2)),        # rms
        skew(signal),                         # skewness
        kurtosis(signal),                     # kurtosis - scipy gives
                                              # excess kurtosis by default
        np.sum(signal ** 2),                  # energy
        len(zero_crossings),                  # count, not a rate
        np.mean(np.abs(np.diff(signal))),     # mean absolute difference
        np.median(signal),                    # median
    ]


print("\nExtracting features")
X_feat = np.array([extract_features(s) for s in X_raw])
print(f"  Feature table: {X_feat.shape}")

X_train_feat = X_feat[train_idx]
X_test_feat = X_feat[test_idx]


# ---------------------------------------------------------------------
# 5. Random Forest
# No scaling here - trees split on thresholds so the scale doesn't matter.
# ---------------------------------------------------------------------

print("\nRandom Forest")

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf.fit(X_train_feat, y_train)
pred_rf = rf.predict(X_test_feat)


# ---------------------------------------------------------------------
# 6. SVM
# This one DOES need scaling because the RBF kernel works on distances,
# and energy is way bigger than skewness so it would dominate everything.
# Scaler is fit on train only.
#
# We didn't tune C or gamma - just used C=10 with the default gamma.
# Probably why our SVM is lower than the papers. Noted in future work.
# ---------------------------------------------------------------------

print("Support Vector Machine")

scaler = StandardScaler().fit(X_train_feat)

svm = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    random_state=RANDOM_STATE,
)
svm.fit(scaler.transform(X_train_feat), y_train)
pred_svm = svm.predict(scaler.transform(X_test_feat))


# ---------------------------------------------------------------------
# 7. CNN
#
# Takes the raw waveform, no features. 3 conv blocks then global average
# pooling. The pooling happens at the END, after the filters have already
# found the patterns - which is the bit we argue matters for compound
# disturbances in the report.
# ---------------------------------------------------------------------

print("Convolutional Neural Network")

encoder = LabelEncoder().fit(y_train)
y_train_enc = encoder.transform(y_train)

X_train_cnn = X_raw[train_idx].reshape(-1, X_raw.shape[1], 1)
X_test_cnn = X_raw[test_idx].reshape(-1, X_raw.shape[1], 1)

cnn = models.Sequential([
    layers.Input(shape=(X_raw.shape[1], 1)),

    layers.Conv1D(32, kernel_size=5, activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),

    layers.Conv1D(64, kernel_size=5, activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),

    layers.Conv1D(128, kernel_size=3, activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling1D(),

    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(classes), activation="softmax"),
])

# This took a while to get working. At the default lr of 0.001 the
# val_accuracy was jumping all over the place (0.4 one epoch, 0.7 the
# next) and never settled. Dropping it to 0.0005 and adding
# ReduceLROnPlateau sorted it out.
cnn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
cnn.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
]

history = cnn.fit(
    X_train_cnn, y_train_enc,
    validation_split=0.15,
    epochs=40,
    batch_size=32,
    callbacks=callbacks,
    verbose=1,
)

pred_cnn = encoder.inverse_transform(
    np.argmax(cnn.predict(X_test_cnn, verbose=0), axis=1)
)

print(f"  Epochs completed: {len(history.history['loss'])} of 40")


# ---------------------------------------------------------------------
# 8. Results
# ---------------------------------------------------------------------

predictions = {
    "Threshold": pred_threshold,
    "SVM": pred_svm,
    "Random Forest": pred_rf,
    "CNN": pred_cnn,
}

accuracies = {name: accuracy_score(y_test, p) * 100
              for name, p in predictions.items()}

print("\n" + "-" * 50)
print("OVERALL ACCURACY")
print("-" * 50)
for name, acc in accuracies.items():
    print(f"  {name:<16}{acc:6.2f}%")
print(f"\n  Threshold ceiling: {ceiling:.2f}%")
print(f"  Random guess:      {100 / len(classes):.2f}%")


# --- Compound vs non-compound ---------------------------------------
# This is the main result of the whole project. Overall accuracy doesn't
# tell you WHERE a model fails, so we split the per-class F1 into the two
# groups and look at the difference.

is_compound = np.array([c in COMPOUND_CLASSES for c in classes])

print("\n" + "-" * 68)
print("MEAN F1: NON-COMPOUND vs COMPOUND")
print("-" * 68)
print(f"  {'Method':<16}{'Non-compound':>14}{'Compound':>12}{'Gap':>10}")

f1_table = pd.DataFrame({"class": classes, "is_compound": is_compound})

for name, pred in predictions.items():
    per_class = f1_score(y_test, pred, labels=classes,
                         average=None, zero_division=0)
    f1_table[name] = per_class

    non_comp = per_class[~is_compound].mean()
    comp = per_class[is_compound].mean()
    print(f"  {name:<16}{non_comp:>14.4f}{comp:>12.4f}{non_comp - comp:>10.4f}")

f1_table.to_csv(f"{OUTPUT_DIR}/per_class_f1.csv", index=False)


# --- Full classification reports -------------------------------------

for name, pred in predictions.items():
    print("\n" + "=" * 68)
    print(name)
    print("=" * 68)
    print(classification_report(y_test, pred, labels=classes,
                                digits=4, zero_division=0))


# --- Why the threshold method fails ---------------------------------
# Printing each class's mean RMS next to what the rule says about it.
# Turned out to be more interesting than expected - most classes aren't
# just slightly off, the rule points them at completely the wrong label.
# This is Table 4-2 in the report.

print("-" * 76)
print("CLASS MEAN RMS vs THRESHOLD BANDS")
print("-" * 76)
print(f"  {'Class':<34}{'RMS/nominal':>12}{'Peak-peak':>11}{'Rule says':>18}")

nominal_all = np.sqrt(np.mean(X_raw[y == "Pure_Sinusoidal"] ** 2, axis=1)).mean()
rms_rows = []

for c in classes:
    subset = X_raw[y == c]
    ratio = np.sqrt(np.mean(subset ** 2, axis=1)).mean() / nominal_all
    peak_to_peak = (subset.max(axis=1) - subset.min(axis=1)).mean()
    assigned = threshold_classify(ratio)
    rms_rows.append([c, ratio, peak_to_peak, assigned])
    print(f"  {c:<34}{ratio:>12.3f}{peak_to_peak:>11.3f}{assigned:>18}")

pd.DataFrame(rms_rows,
             columns=["class", "rms_ratio", "peak_to_peak", "rule_assigns"]
             ).to_csv(f"{OUTPUT_DIR}/class_rms_table.csv", index=False)


# --- Which features actually do anything -----------------------------
# Shuffle a feature and see how much accuracy drops. Near zero means the
# model isn't really using it. Supervisor asked us to actually understand
# the features instead of just throwing them at a classifier, so this and
# the z-scores below are for that.

print("\n" + "-" * 50)
print("RANDOM FOREST PERMUTATION IMPORTANCE")
print("-" * 50)

perm = permutation_importance(rf, X_test_feat, y_test, n_repeats=5,
                              random_state=RANDOM_STATE, n_jobs=-1)

for i in np.argsort(-perm.importances_mean):
    print(f"  {FEATURE_NAMES[i]:<22}{perm.importances_mean[i]:+.4f}")

pd.DataFrame({"feature": FEATURE_NAMES,
              "importance": perm.importances_mean}
             ).to_csv(f"{OUTPUT_DIR}/feature_importance.csv", index=False)


# --- Z-scores: how far each class sits from average on each feature ---

z_scores = pd.DataFrame(
    (X_feat - X_feat.mean(axis=0)) / X_feat.std(axis=0),
    columns=FEATURE_NAMES,
)
z_scores["class"] = y
class_z = z_scores.groupby("class").mean()
class_z.to_csv(f"{OUTPUT_DIR}/class_feature_zscores.csv")
print(f"\n  Class z-scores saved to {OUTPUT_DIR}/class_feature_zscores.csv")


# ---------------------------------------------------------------------
# 9. Figures
# ---------------------------------------------------------------------

print("\nSaving figures")

# Confusion matrix for each method
for name, pred in predictions.items():
    cm = confusion_matrix(y_test, pred, labels=classes)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.title(f"{name} Confusion Matrix ({accuracies[name]:.2f}%)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_{name.replace(' ', '_')}.png", dpi=150)
    plt.close()

# Overall accuracy comparison
plt.figure(figsize=(8, 5))
bars = plt.bar(list(accuracies.keys()), list(accuracies.values()))
for bar, acc in zip(bars, accuracies.values()):
    plt.text(bar.get_x() + bar.get_width() / 2, acc + 1.5,
             f"{acc:.1f}%", ha="center", fontweight="bold")
plt.ylabel("Overall Accuracy (%)")
plt.title("Threshold vs SVM vs Random Forest vs CNN")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/accuracy_comparison.png", dpi=150)
plt.close()

# Compound gap
labels_x = list(predictions.keys())
non_comp_means = [f1_table[n][~is_compound].mean() for n in labels_x]
comp_means = [f1_table[n][is_compound].mean() for n in labels_x]

x_pos = np.arange(len(labels_x))
width = 0.36

fig, ax = plt.subplots(figsize=(8, 4.6))
ax.bar(x_pos - width / 2, non_comp_means, width, label="Non-compound (9 classes)")
ax.bar(x_pos + width / 2, comp_means, width, label="Compound (8 classes)")
for i, (a, b) in enumerate(zip(non_comp_means, comp_means)):
    ax.annotate(f"gap {a - b:.3f}", xy=(i, max(a, b) + 0.08),
                ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(x_pos)
ax.set_xticklabels(labels_x)
ax.set_ylabel("Mean F1 score")
ax.set_ylim(0, 1.15)
ax.set_title("Mean F1: non-compound vs compound disturbance classes")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/compound_gap.png", dpi=150)
plt.close()

# All 17 classes plotted, one example each
fig, axes = plt.subplots(6, 3, figsize=(13, 16), sharex=True)
axes = axes.ravel()

for ax, c in zip(axes, classes):
    example = X_raw[np.where(y == c)[0][0]]
    ax.plot(example, linewidth=0.9,
            color="#C4622D" if c in COMPOUND_CLASSES else "#2F5C9E")
    ax.set_title(c.replace("_", " "), fontsize=9.5)
    ax.grid(alpha=0.2)

for ax in axes[len(classes):]:
    ax.axis("off")

fig.suptitle("The 17 IEEE 1159 disturbance classes in the SEED dataset\n"
             "blue = non-compound, orange = compound", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(f"{OUTPUT_DIR}/all_17_classes.png", dpi=150, bbox_inches="tight")
plt.close()


# Each compound class beside the two single disturbances it combines.
# This shows the classification problem visually: the compound waveform
# carries both signatures at once.
PAIRS = [
    ("Sag", "Harmonics", "Sag_with_Harmonics"),
    ("Swell", "Harmonics", "Swell_with_Harmonics"),
    ("Sag", "Oscillatory_Transient", "Sag_with_Oscillatory_Transient"),
    ("Swell", "Oscillatory_Transient", "Swell_with_Oscillatory_Transient"),
    ("Flicker", "Sag", "Flicker_with_Sag"),
    ("Flicker", "Swell", "Flicker_with_Swell"),
]

example = {c: X_raw[np.where(y == c)[0][0]] for c in classes}

fig, axes = plt.subplots(len(PAIRS), 3, figsize=(13, 2.5 * len(PAIRS)),
                         sharex=True)

for row, (first, second, combined) in enumerate(PAIRS):
    for col, (name, colour) in enumerate([(first, "#2F5C9E"),
                                          (second, "#2F5C9E"),
                                          (combined, "#C4622D")]):
        ax = axes[row, col]
        ax.plot(example[name], linewidth=1.0, color=colour)
        ax.set_title(name.replace("_", " "), fontsize=9)
        ax.grid(alpha=0.2)
    axes[row, 2].set_facecolor("#FCF3EC")

fig.suptitle("Compound disturbances and their single components\n"
             "left and centre: the two single disturbances, "
             "right: the compound class", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f"{OUTPUT_DIR}/compound_decomposition.png", dpi=150,
            bbox_inches="tight")
plt.close()

# Five signals from the same class, to show that the 1,000 signals per
# class are not identical copies - disturbance parameters and starting
# phase both vary within a class.
SHOW = ["Sag", "Harmonics", "Sag_with_Harmonics", "Oscillatory_Transient"]

fig, axes = plt.subplots(1, len(SHOW), figsize=(4.2 * len(SHOW), 3.2))
for ax, c in zip(axes, SHOW):
    for i in np.where(y == c)[0][:5]:
        ax.plot(X_raw[i], linewidth=0.8, alpha=0.75)
    ax.set_title(c.replace("_", " "), fontsize=10)
    ax.set_xlabel("Sample index")
    ax.grid(alpha=0.2)
axes[0].set_ylabel("Amplitude")
fig.suptitle("Five example signals from the same class "
             "(parameters vary within each class)", fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(f"{OUTPUT_DIR}/within_class_variation.png", dpi=150)
plt.close()


# ---------------------------------------------------------------------
# 10. Checking a hunch about the median feature
#
# Median comes out 2nd most important, but when we looked at the class
# z-scores it barely separates anything (max |z| is only 0.16). That
# doesn't add up unless it's picking up something that varies WITHIN a
# class rather than between classes.
#
# We noticed when plotting the waveforms that the starting phase isn't
# consistent, so that was our guess. Testing it here.
# ---------------------------------------------------------------------


n_samples = X_raw.shape[1]
t = np.arange(n_samples)

# One cycle fits the window exactly, so the fundamental is 1 cycle per
# window. Project onto sin/cos and use arctan2 to get the phase back.
cos_component = X_raw @ np.cos(2 * np.pi * t / n_samples) * (2 / n_samples)
sin_component = X_raw @ np.sin(2 * np.pi * t / n_samples) * (2 / n_samples)
phase = np.arctan2(cos_component, sin_component)

median_feature = np.median(X_raw, axis=1)
r, p_value = pearsonr(phase, median_feature)

print("\n" + "-" * 50)
print("MEDIAN vs STARTING PHASE")
print("-" * 50)
print(f"  Pearson r: {r:+.4f}  (p = {p_value:.2g})")
print("  The association is weak, so phase alone does not explain why")
print("  median ranks high in permutation importance. The interpretation")
print("  in the report is therefore left as tentative.")


# ---------------------------------------------------------------------
# 11. Summary
# ---------------------------------------------------------------------

pd.DataFrame({
    "Method": list(accuracies.keys()),
    "Accuracy (%)": [round(a, 2) for a in accuracies.values()],
}).to_csv(f"{OUTPUT_DIR}/accuracy_results.csv", index=False)

print(f"\nDone. All outputs saved to {OUTPUT_DIR}")
