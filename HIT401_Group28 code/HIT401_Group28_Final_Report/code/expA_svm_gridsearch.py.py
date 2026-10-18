"""Experiment 1: tune the RBF SVM and compare it with the original model.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


OUTPUT_DIR = "/content/outputs"

print("=" * 60)
print("EXPERIMENT 1: SVM GRID SEARCH")
print("=" * 60)


# Use the training and test sets prepared in the main code.
X_train_features = X_feat[train_idx]
X_test_features = X_feat[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]

# Reuse the scaler that was fitted on the training data.
X_train_scaled = scaler.transform(X_train_features)
X_test_scaled = scaler.transform(X_test_features)


parameter_grid = {
    "C": [0.1, 1, 10, 50, 100, 500],
    "gamma": ["scale", "auto"],
}

print("\nRunning grid search. This may take a few minutes...")

grid_search = GridSearchCV(
    estimator=SVC(kernel="rbf", random_state=42),
    param_grid=parameter_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1,
)

grid_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV score:   {grid_search.best_score_ * 100:.2f}%")


# Compare the tuned model with the original SVM.
best_svm = grid_search.best_estimator_
tuned_predictions = best_svm.predict(X_test_scaled)
tuned_accuracy = accuracy_score(y_test, tuned_predictions) * 100

original_predictions = svm.predict(X_test_scaled)
original_accuracy = accuracy_score(y_test, original_predictions) * 100

print(f"\nUntuned SVM accuracy: {original_accuracy:.2f}%")
print(f"Tuned SVM accuracy:   {tuned_accuracy:.2f}%")
print(f"Improvement:          +{tuned_accuracy - original_accuracy:.2f} pp")


def calculate_compound_gap(predictions, true_labels, class_names):
    """Return the non-compound F1, compound F1 and the gap between them."""

    per_class_f1 = f1_score(
        true_labels,
        predictions,
        labels=class_names,
        average=None,
        zero_division=0,
    )

    compound_mask = np.array(
        [class_name in COMPOUND_CLASSES for class_name in class_names]
    )

    non_compound_f1 = per_class_f1[~compound_mask].mean()
    compound_f1 = per_class_f1[compound_mask].mean()
    gap = non_compound_f1 - compound_f1

    return non_compound_f1, compound_f1, gap


original_nc, original_c, original_gap = calculate_compound_gap(
    original_predictions, y_test, classes
)
tuned_nc, tuned_c, tuned_gap = calculate_compound_gap(
    tuned_predictions, y_test, classes
)

print("\n" + "-" * 55)
print(f"{'':20} {'Non-compound':>14} {'Compound':>10} {'Gap':>8}")
print("-" * 55)
print(
    f"{'Untuned SVM (C=10)':<20} "
    f"{original_nc:>14.4f} {original_c:>10.4f} {original_gap:>8.4f}"
)
print(
    f"{'Tuned SVM':<20} "
    f"{tuned_nc:>14.4f} {tuned_c:>10.4f} {tuned_gap:>8.4f}"
)
print("-" * 55)


results = pd.DataFrame(
    [
        {
            "variant": "Untuned SVM (C=10, gamma=scale)",
            "best_params": "C=10, gamma=scale",
            "accuracy": round(original_accuracy, 4),
            "non_compound_f1": round(original_nc, 4),
            "compound_f1": round(original_c, 4),
            "gap": round(original_gap, 4),
        },
        {
            "variant": "Tuned SVM",
            "best_params": str(grid_search.best_params_),
            "accuracy": round(tuned_accuracy, 4),
            "non_compound_f1": round(tuned_nc, 4),
            "compound_f1": round(tuned_c, 4),
            "gap": round(tuned_gap, 4),
        },
    ]
)

results_path = f"{OUTPUT_DIR}/svm_gridsearch_results.csv"
results.to_csv(results_path, index=False)
print(f"\nResults saved to {results_path}")


fig, (accuracy_axis, gap_axis) = plt.subplots(1, 2, figsize=(10, 4.5))

accuracy_values = [original_accuracy, tuned_accuracy]
accuracy_labels = ["Untuned SVM\n(C=10)", "Tuned SVM"]

accuracy_axis.bar(
    accuracy_labels,
    accuracy_values,
    color=["#5A6672", "#065A82"],
)

for position, value in enumerate(accuracy_values):
    accuracy_axis.text(
        position,
        value + 0.4,
        f"{value:.2f}%",
        ha="center",
        fontweight="bold",
    )

accuracy_axis.set_ylabel("Test accuracy (%)")
accuracy_axis.set_ylim(0, 105)
accuracy_axis.set_title("SVM accuracy: before vs after tuning")

x_positions = np.arange(2)
bar_width = 0.35

gap_axis.bar(
    x_positions - bar_width / 2,
    [original_nc, tuned_nc],
    bar_width,
    label="Non-compound",
    color="#065A82",
)
gap_axis.bar(
    x_positions + bar_width / 2,
    [original_c, tuned_c],
    bar_width,
    label="Compound",
    color="#E8A33D",
)

gap_axis.set_xticks(x_positions)
gap_axis.set_xticklabels(["Untuned SVM", "Tuned SVM"])
gap_axis.set_ylabel("Mean F1 score")
gap_axis.set_ylim(0, 1.15)
gap_axis.set_title("Compound vs non-compound F1:\nbefore vs after tuning")
gap_axis.legend()

plt.tight_layout()

figure_path = f"{OUTPUT_DIR}/svm_gridsearch_comparison.png"
plt.savefig(figure_path, dpi=150)
plt.close()

print(f"Figure saved to {figure_path}")
print("\nDone - Experiment 1 complete.")
