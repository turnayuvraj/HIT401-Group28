# Experiment 4 - Train separate models for non-compound and compound classes
# HIT401 Capstone - Group 28
# This version includes CNN as well as Random Forest and SVM


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# where to save results
OUTPUT_DIR = "/content/outputs"
RANDOM_STATE = 42

# -------------------------------------------------------
# Step 1 - Define the two groups
# 9 non-compound = one thing wrong at a time
# 8 compound = two things wrong at the same time
# -------------------------------------------------------

NON_COMPOUND = [
    'Flicker', 'Harmonics', 'Interruption', 'Notch',
    'Oscillatory_Transient', 'Pure_Sinusoidal',
    'Sag', 'Swell', 'Transient'
]

COMPOUND = [
    'Flicker_with_Sag', 'Flicker_with_Swell',
    'Harmonics_with_Sag', 'Harmonics_with_Swell',
    'Sag_with_Harmonics', 'Sag_with_Oscillatory_Transient',
    'Swell_with_Harmonics', 'Swell_with_Oscillatory_Transient'
]

print("Starting separate model training (RF, SVM and CNN)...")
print(f"Non-compound classes: {len(NON_COMPOUND)}")
print(f"Compound classes: {len(COMPOUND)}")

# -------------------------------------------------------
# Step 2 - Build the CNN
# Same architecture as our original CNN in PQD_Code.py
# n_classes changes depending on which group we train on
# -------------------------------------------------------

def build_cnn(n_samples, n_classes):
    # same three conv block structure as original
    model = models.Sequential([
        layers.Input(shape=(n_samples, 1)),

        # first conv block
        layers.Conv1D(32, 5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),

        # second conv block
        layers.Conv1D(64, 5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),

        # third conv block
        layers.Conv1D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),

        # pool and classify
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation='softmax'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# -------------------------------------------------------
# Step 3 - Function to train and test one group
# Trains RF, SVM and CNN on the same group
# -------------------------------------------------------

def train_and_test(group_name, group_classes):

    print(f"\n{'='*55}")
    print(f"Now training on: {group_name}")
    print(f"{'='*55}")

    # filter data to only keep signals from this group
    mask_train = np.isin(y[train_idx], group_classes)
    mask_test  = np.isin(y[test_idx],  group_classes)

    # feature data for RF and SVM
    X_tr_feat = X_feat[train_idx][mask_train]
    X_te_feat = X_feat[test_idx][mask_test]

    # raw waveform data for CNN
    X_tr_raw  = X_raw[train_idx][mask_train]
    X_te_raw  = X_raw[test_idx][mask_test]

    y_tr = y[train_idx][mask_train]
    y_te = y[test_idx][mask_test]

    print(f"Training samples: {len(y_tr)}")
    print(f"Test samples:     {len(y_te)}")

    # -------------------------------------------------------
    # Train Random Forest
    # 200 trees, same as original
    # -------------------------------------------------------
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_tr_feat, y_tr)
    pred_rf = rf.predict(X_te_feat)
    acc_rf  = accuracy_score(y_te, pred_rf) * 100
    f1_rf   = f1_score(y_te, pred_rf, average='macro', zero_division=0)
    print(f"Random Forest accuracy: {acc_rf:.2f}%  macro F1: {f1_rf:.4f}")

    # -------------------------------------------------------
    # Train SVM
    # scale features first, same C and gamma as original
    # -------------------------------------------------------
    print("\nTraining SVM...")
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_feat)
    X_te_scaled = scaler.transform(X_te_feat)

    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=RANDOM_STATE)
    svm.fit(X_tr_scaled, y_tr)
    pred_svm = svm.predict(X_te_scaled)
    acc_svm  = accuracy_score(y_te, pred_svm) * 100
    f1_svm   = f1_score(y_te, pred_svm, average='macro', zero_division=0)
    print(f"SVM accuracy: {acc_svm:.2f}%  macro F1: {f1_svm:.4f}")

    # -------------------------------------------------------
    # Train CNN
    # encode labels as numbers because CNN needs that
    # same training settings as original CNN
    # -------------------------------------------------------
    print("\nTraining CNN (this takes a few minutes)...")
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    # CNN needs numbers not strings for labels
    encoder = LabelEncoder()
    y_tr_enc = encoder.fit_transform(y_tr)

    # reshape for CNN - needs (samples, timesteps, 1)
    X_tr_cnn = X_tr_raw.reshape(-1, X_raw.shape[1], 1)
    X_te_cnn = X_te_raw.reshape(-1, X_raw.shape[1], 1)

    # build and train
    cnn = build_cnn(X_raw.shape[1], len(group_classes))
    cnn.fit(
        X_tr_cnn, y_tr_enc,
        validation_split=0.15,
        epochs=40,
        batch_size=32,
        callbacks=[
            # stop early if not improving
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=8,
                restore_best_weights=True
            ),
            # reduce learning rate if stuck
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            ),
        ],
        verbose=0  # hide training output to keep it clean
    )

    # predict and convert back to class names
    pred_cnn_enc = np.argmax(cnn.predict(X_te_cnn, verbose=0), axis=1)
    pred_cnn     = encoder.inverse_transform(pred_cnn_enc)
    acc_cnn      = accuracy_score(y_te, pred_cnn) * 100
    f1_cnn       = f1_score(y_te, pred_cnn, average='macro', zero_division=0)
    print(f"CNN accuracy: {acc_cnn:.2f}%  macro F1: {f1_cnn:.4f}")

    # -------------------------------------------------------
    # Per class F1 for all three models
    # so we can see which classes are easy and which are hard
    # -------------------------------------------------------
    pc_rf  = f1_score(y_te, pred_rf,  labels=group_classes,
                      average=None, zero_division=0)
    pc_svm = f1_score(y_te, pred_svm, labels=group_classes,
                      average=None, zero_division=0)
    pc_cnn = f1_score(y_te, pred_cnn, labels=group_classes,
                      average=None, zero_division=0)

    class_results = pd.DataFrame({
        'class':  group_classes,
        'RF F1':  np.round(pc_rf,  4),
        'SVM F1': np.round(pc_svm, 4),
        'CNN F1': np.round(pc_cnn, 4),
    })
    print(f"\nPer class results:")
    print(class_results.to_string(index=False))

    summary = {
        'group':    group_name,
        'rf_acc':   round(acc_rf,  2),
        'rf_f1':    round(f1_rf,   4),
        'svm_acc':  round(acc_svm, 2),
        'svm_f1':   round(f1_svm,  4),
        'cnn_acc':  round(acc_cnn, 2),
        'cnn_f1':   round(f1_cnn,  4),
    }

    return summary, class_results


# -------------------------------------------------------
# Step 4 - Run for non-compound first then compound
# -------------------------------------------------------

summary_nc, details_nc = train_and_test(
    "Non-compound (9 classes)", NON_COMPOUND
)

summary_c, details_c = train_and_test(
    "Compound (8 classes)", COMPOUND
)


# -------------------------------------------------------
# Step 5 - Print final summary
# -------------------------------------------------------

print("\n" + "=" * 65)
print("FINAL SUMMARY - Both groups, all three models")
print("=" * 65)
summary_df = pd.DataFrame([summary_nc, summary_c])
print(summary_df.to_string(index=False))


# -------------------------------------------------------
# Step 6 - Save results
# -------------------------------------------------------

summary_df.to_csv(
    f"{OUTPUT_DIR}/separate_model_summary_with_cnn.csv", index=False
)
details_nc.to_csv(
    f"{OUTPUT_DIR}/noncompound_per_class_f1.csv", index=False
)
details_c.to_csv(
    f"{OUTPUT_DIR}/compound_per_class_f1.csv", index=False
)
print("\nResults saved to outputs folder")


# -------------------------------------------------------
# Step 7 - Chart comparing all three models on both groups
# -------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5))

groups   = ['Non-compound\n(9 classes)', 'Compound\n(8 classes)']
rf_acc   = [summary_nc['rf_acc'],  summary_c['rf_acc']]
svm_acc  = [summary_nc['svm_acc'], summary_c['svm_acc']]
cnn_acc  = [summary_nc['cnn_acc'], summary_c['cnn_acc']]

x = np.arange(2)
w = 0.25

bars_rf  = ax.bar(x - w,   rf_acc,  w, label='Random Forest', color='#065A82')
bars_svm = ax.bar(x,       svm_acc, w, label='SVM',           color='#E8A33D')
bars_cnn = ax.bar(x + w,   cnn_acc, w, label='CNN',           color='#2E8B57')

# add numbers on top of each bar
for bar in [bars_rf, bars_svm, bars_cnn]:
    for b in bar:
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.4,
            f"{b.get_height():.1f}%",
            ha='center', fontsize=9, fontweight='bold'
        )

ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_ylabel('Accuracy (%)')
ax.set_ylim(0, 115)
ax.set_title('All three models trained separately on each group')
ax.legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/separate_model_all_three.png", dpi=150)
plt.close()
print("Chart saved to outputs folder")

print("\nDone - Experiment 4 complete (with CNN)")
