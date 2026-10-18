"""Experiment 2: compare the four methods at different noise levels.

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from tensorflow.keras import callbacks as cb
from tensorflow.keras import layers, models


OUTPUT_DIR = "/content/outputs"
RANDOM_STATE = 42

print("=" * 60)
print("EXPERIMENT 2: NOISE ROBUSTNESS")
print("=" * 60)


def add_noise(signals, snr_db):
    """Add Gaussian noise to each signal at the chosen SNR level."""

    if snr_db is None:
        return signals.copy()

    signal_power = np.mean(signals**2, axis=1, keepdims=True)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    noise = np.random.randn(*signals.shape).astype(np.float32)
    noise *= np.sqrt(noise_power)

    return signals + noise


def extract_features(signal):
    """Calculate the 12 statistical features used by SVM and Random Forest."""

    signal = np.asarray(signal, dtype=float)
    minimum = np.min(signal)
    maximum = np.max(signal)
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]

    return [
        np.mean(signal),
        np.std(signal),
        minimum,
        maximum,
        maximum - minimum,
        np.sqrt(np.mean(signal**2)),
        skew(signal),
        kurtosis(signal),
        np.sum(signal**2),
        len(zero_crossings),
        np.mean(np.abs(np.diff(signal))),
        np.median(signal),
    ]


def threshold_classify(rms_ratio):
    """Classify a signal with the simplified RMS threshold method."""

    if rms_ratio < 0.10:
        return "Interruption"
    if rms_ratio < 0.90:
        return "Sag"
    if rms_ratio > 1.10:
        return "Swell"
    return "Pure_Sinusoidal"


def calculate_compound_gap(predictions, true_labels, class_names, compound_classes):
    """Return the non-compound F1, compound F1 and their difference."""

    compound_mask = np.array(
        [class_name in compound_classes for class_name in class_names]
    )
    per_class_f1 = f1_score(
        true_labels,
        predictions,
        labels=class_names,
        average=None,
        zero_division=0,
    )

    non_compound_f1 = per_class_f1[~compound_mask].mean()
    compound_f1 = per_class_f1[compound_mask].mean()
    gap = non_compound_f1 - compound_f1

    return non_compound_f1, compound_f1, gap


def build_cnn(number_of_samples, number_of_classes):
    """Create the 1D CNN used in the main experiment."""

    model = models.Sequential(
        [
            layers.Input(shape=(number_of_samples, 1)),
            layers.Conv1D(32, 5, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 5, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Conv1D(128, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(number_of_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# None represents the original clean signals.
SNR_LEVELS = [None, 40, 30, 20]
SNR_LABELS = ["Clean", "40 dB", "30 dB", "20 dB"]

y_train = y[train_idx]
y_test = y[test_idx]

label_encoder = LabelEncoder().fit(y_train)
results = []


for snr_level, snr_label in zip(SNR_LEVELS, SNR_LABELS):
    print(f"\n{'=' * 40}")
    print(f"SNR: {snr_label}")
    print(f"{'=' * 40}")

    np.random.seed(RANDOM_STATE)

    X_train_noisy = add_noise(X_raw[train_idx], snr_level)
    X_test_noisy = add_noise(X_raw[test_idx], snr_level)

    # Threshold method
    normal_training_signals = X_train_noisy[y_train == "Pure_Sinusoidal"]
    nominal_rms = np.sqrt(
        np.mean(normal_training_signals**2, axis=1)
    ).mean()
    test_rms = np.sqrt(np.mean(X_test_noisy**2, axis=1))

    threshold_predictions = np.array(
        [threshold_classify(rms / nominal_rms) for rms in test_rms]
    )
    threshold_accuracy = accuracy_score(y_test, threshold_predictions) * 100
    _, _, threshold_gap = calculate_compound_gap(
        threshold_predictions, y_test, classes, COMPOUND_CLASSES
    )

    print(
        f"  Threshold:     {threshold_accuracy:.2f}%  "
        f"gap {threshold_gap:.4f}"
    )

    # Features used by SVM and Random Forest
    X_train_features = np.array(
        [extract_features(signal) for signal in X_train_noisy]
    )
    X_test_features = np.array(
        [extract_features(signal) for signal in X_test_noisy]
    )

    noise_scaler = StandardScaler().fit(X_train_features)
    X_train_scaled = noise_scaler.transform(X_train_features)
    X_test_scaled = noise_scaler.transform(X_test_features)

    # SVM
    noise_svm = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        random_state=RANDOM_STATE,
    )
    noise_svm.fit(X_train_scaled, y_train)
    svm_predictions = noise_svm.predict(X_test_scaled)
    svm_accuracy = accuracy_score(y_test, svm_predictions) * 100
    _, _, svm_gap = calculate_compound_gap(
        svm_predictions, y_test, classes, COMPOUND_CLASSES
    )

    print(f"  SVM:           {svm_accuracy:.2f}%  gap {svm_gap:.4f}")

    # Random Forest
    noise_rf = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    noise_rf.fit(X_train_features, y_train)
    rf_predictions = noise_rf.predict(X_test_features)
    rf_accuracy = accuracy_score(y_test, rf_predictions) * 100
    _, _, rf_gap = calculate_compound_gap(
        rf_predictions, y_test, classes, COMPOUND_CLASSES
    )

    print(f"  Random Forest: {rf_accuracy:.2f}%  gap {rf_gap:.4f}")

    # CNN
    tf.random.set_seed(RANDOM_STATE)

    X_train_cnn = X_train_noisy.reshape(-1, X_raw.shape[1], 1)
    X_test_cnn = X_test_noisy.reshape(-1, X_raw.shape[1], 1)
    y_train_encoded = label_encoder.transform(y_train)

    noise_cnn = build_cnn(X_raw.shape[1], len(classes))
    noise_cnn.fit(
        X_train_cnn,
        y_train_encoded,
        validation_split=0.15,
        epochs=40,
        batch_size=32,
        callbacks=[
            cb.EarlyStopping(
                monitor="val_loss",
                patience=8,
                restore_best_weights=True,
            ),
            cb.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
            ),
        ],
        verbose=0,
    )

    cnn_probabilities = noise_cnn.predict(X_test_cnn, verbose=0)
    cnn_predictions = label_encoder.inverse_transform(
        np.argmax(cnn_probabilities, axis=1)
    )
    cnn_accuracy = accuracy_score(y_test, cnn_predictions) * 100
    _, _, cnn_gap = calculate_compound_gap(
        cnn_predictions, y_test, classes, COMPOUND_CLASSES
    )

    print(f"  CNN:           {cnn_accuracy:.2f}%  gap {cnn_gap:.4f}")

    results.append(
        {
            "snr": snr_label,
            "threshold_acc": round(threshold_accuracy, 2),
            "svm_acc": round(svm_accuracy, 2),
            "rf_acc": round(rf_accuracy, 2),
            "cnn_acc": round(cnn_accuracy, 2),
            "threshold_gap": round(threshold_gap, 4),
            "svm_gap": round(svm_gap, 4),
            "rf_gap": round(rf_gap, 4),
            "cnn_gap": round(cnn_gap, 4),
        }
    )


results_df = pd.DataFrame(results)
results_path = f"{OUTPUT_DIR}/noise_robustness_results.csv"
results_df.to_csv(results_path, index=False)
print(f"\nResults saved to {results_path}")


fig, (accuracy_axis, gap_axis) = plt.subplots(1, 2, figsize=(12, 5))

accuracy_lines = [
    ("Threshold", "threshold_acc", "--"),
    ("SVM", "svm_acc", "-."),
    ("Random Forest", "rf_acc", ":"),
    ("CNN", "cnn_acc", "-"),
]

for method_name, column_name, line_style in accuracy_lines:
    accuracy_axis.plot(
        SNR_LABELS,
        results_df[column_name],
        marker="o",
        linestyle=line_style,
        label=method_name,
    )

accuracy_axis.set_xlabel("Noise level")
accuracy_axis.set_ylabel("Test accuracy (%)")
accuracy_axis.set_title("Accuracy under noise")
accuracy_axis.legend()
accuracy_axis.grid(alpha=0.3)

gap_lines = [
    ("Threshold", "threshold_gap", "--"),
    ("SVM", "svm_gap", "-."),
    ("Random Forest", "rf_gap", ":"),
    ("CNN", "cnn_gap", "-"),
]

for method_name, column_name, line_style in gap_lines:
    gap_axis.plot(
        SNR_LABELS,
        results_df[column_name],
        marker="o",
        linestyle=line_style,
        label=method_name,
    )

gap_axis.set_xlabel("Noise level")
gap_axis.set_ylabel("Compound gap (non-compound F1 - compound F1)")
gap_axis.set_title("Compound gap under noise")
gap_axis.legend()
gap_axis.grid(alpha=0.3)

plt.tight_layout()

figure_path = f"{OUTPUT_DIR}/noise_robustness.png"
plt.savefig(figure_path, dpi=150)
plt.close()

print(f"Figure saved to {figure_path}")
print("\nDone - Experiment 2 complete.")
