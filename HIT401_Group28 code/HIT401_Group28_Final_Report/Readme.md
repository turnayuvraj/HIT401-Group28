# HIT401 Group 28 – Power Quality Disturbance Classification

This is the code for our HIT401 final report at CDU. We compared four methods for
classifying power quality disturbances on the SEED dataset (17 classes): a simple
IEEE 1159 threshold rule, SVM, Random Forest and a 1D CNN. We also looked at the
"compound gap", which is the mean F1 on non-compound classes minus the mean F1 on
compound classes.

## What each file does
| File | What it is for |
|---|---|
| PQD_Code.py | Main code. Runs all four methods and the feature analysis (Tables 3.2, 4.1–4.3, Figures 3.1–3.3, 4.1–4.5) |
| expA_svm_gridsearch.py | Experiment A – tuning the SVM (Table 4.4) |
| expB_groupwise_training.py | Experiment B – training on non-compound and compound classes separately (Table 4.5, Figure 4.6) |
| expC_feature_voting.py | Experiment C – feature selection with Lasso and ReliefF (Table 4.6, Figure 4.7) |
| expD_noise_robustness.py | Experiment D – adding noise (Table 4.7, Figure 4.8) |
| cnn_ablation.ipynb, kernel_search.ipynb, filter_search.ipynb, depth_search.ipynb, final_model.ipynb | Experiment E – testing CNN parts and picking the final CNN (Figure 4.9, Tables 4.8–4.9, Figures 4.10–4.11) |
| second_dataset.ipynb | Experiment F – testing the final CNN on a second dataset (Table 4.10) |

Note: inside the scripts we called them "Experiment 1, 2 and 4". In the report these
are Experiments A, D and B.

## Data
The data is not in this repo, you need to download it:
- SEED dataset: https://www.kaggle.com/datasets/sumairaziz/seed-power-quality-disturbance-dataset
  (17 CSV files, one for each class, no header row). Don't rename the files because the
  code uses the file names as the class labels.
- Second dataset: https://doi.org/10.5281/zenodo.22816322 (PDQ_10c_Clasiffy_data.csv)

## What you need
We ran everything on Google Colab with a GPU. It uses numpy, pandas, scipy,
scikit-learn, tensorflow, matplotlib and seaborn (all already on Colab).

## How to run it
1. For PQD_Code.py and expC_feature_voting.py, upload the 17 SEED CSV files straight
   into Colab (/content). The notebooks read from Google Drive, so change DATA_DIR and
   OUTPUT_DIR at the top of each notebook to your own folder (and CSV_PATH in
   second_dataset.ipynb).
2. Run PQD_Code.py first.
3. In the same Colab session, run:
   - `%run -i expA_svm_gridsearch.py`
   - `%run -i expB_groupwise_training.py`
   - `%run -i expD_noise_robustness.py`

   These need the variables from PQD_Code.py, so they won't work on their own.
4. expC_feature_voting.py can be run on its own.
5. Run the notebooks in this order: cnn_ablation, kernel_search, filter_search,
   depth_search, final_model, then second_dataset.

## Settings we used
- Same 80/20 stratified split for everything, random_state = 42
  (13,600 training and 3,400 test signals)
- SVM: RBF kernel, C = 10, gamma = scale
- Random Forest: 200 trees
- Final CNN: 3 Conv1D blocks with 32, 64 and 128 filters (kernel size 3), batch norm,
  global average pooling, dense layer of 64, dropout 0.3. Adam with learning rate
  0.0005, batch size 32, up to 40 epochs with early stopping (patience 8) and
  ReduceLROnPlateau (factor 0.5, patience 3).
