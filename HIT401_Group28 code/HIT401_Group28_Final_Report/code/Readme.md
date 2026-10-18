# Power Quality Disturbance Classification – HIT401 Group 28

Code for the Group 28 final report (Charles Darwin University). It compares an IEEE 1159 threshold baseline, SVM, Random Forest and a 1-D CNN on the 17-class SEED dataset, and reports the compound gap (mean F1 on non-compound minus compound classes).

## Files

| File | Report |
|---|---|
| `main_comparison.py` | Main comparison (Tables 4.1–4.3) |
| `expA_svm_gridsearch.py` | Experiment A – SVM tuning |
| `expB_groupwise_training.py` | Experiment B – group-wise training |
| `expC_feature_voting.py` | Experiment C – feature selection |
| `expD_noise_robustness.py` | Experiment D – noise |
| `expE1`–`expE5` notebooks | Experiment E – CNN ablation, kernel, filter and depth search, final model |
| `expF_second_dataset.ipynb` | Experiment F – second dataset |

## Data (not included)

- **SEED:** https://www.kaggle.com/datasets/sumairaziz/seed-power-quality-disturbance-dataset (17 CSV files, one per class, no header; keep the original file names)
- **Second dataset:** https://doi.org/10.5281/zenodo.22816322 (`PDQ_10c_Clasiffy_data.csv`)

## Environment

Google Colab with a GPU. Packages: numpy, pandas, scipy, scikit-learn, tensorflow, matplotlib, seaborn (`pip install -r requirements.txt`).

## How to run

1. Upload the datasets to Google Drive and set `DATA_DIR` / `OUTPUT_DIR` (and `CSV_PATH` for Experiment F) at the top of each file.
2. Run `main_comparison.py` first.
3. In the same Colab session, run `%run -i expA_svm_gridsearch.py`, `%run -i expB_groupwise_training.py` and `%run -i expD_noise_robustness.py` (they reuse variables from step 2).
4. Run `expC_feature_voting.py`, then the notebooks `expE1` → `expE5`, then `expF_second_dataset.ipynb`.

## Main settings

- Stratified 80/20 split, `random_state = 42` (13,600 train / 3,400 test)
- SVM: RBF, C = 10, gamma = scale · RF: 200 trees
- Final CNN: 3 Conv1D blocks (32-64-128 filters, kernel 3), batch norm, global average pooling, dense 64, dropout 0.3; Adam (lr 0.0005), batch 32, up to 40 epochs with early stopping
- Small run-to-run differences can occur because GPU training is not fully deterministic.
