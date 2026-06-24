# Breast Cancer Detection — ML Model Comparison

A machine learning project that trains and compares three classifiers — **Random Forest**, **Gradient Boosting**, and **XGBoost** — to predict whether a breast tumor is benign or malignant, using the [Breast Cancer Wisconsin (Diagnostic) Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data).

> **Scope note:** This is an educational/portfolio project demonstrating an end-to-end ML workflow (EDA → training → evaluation → model comparison). It is **not** a validated diagnostic tool and should not be used for actual medical decision-making.

## Overview

The dataset contains 30 numeric features computed from digitized images of fine needle aspirate (FNA) biopsies — things like cell radius, texture, perimeter, and concavity — for 569 tumor samples labeled as malignant (M) or benign (B).

This project:
1. Cleans and explores the data
2. Trains three classification models
3. Evaluates each using accuracy, precision/recall, ROC-AUC, and 5-fold cross-validation
4. Compares model performance side-by-side
5. Saves the best-performing model for reuse

## Why these metrics matter here

In a cancer-screening context, **accuracy alone is misleading**. A model that predicts "benign" 95% of the time on an imbalanced dataset can still post a high accuracy score while missing real malignant cases. That's why this project also reports:

- **Recall on the malignant class** — how many actual malignant cases the model catches (missing one is the costly error)
- **ROC-AUC** — how well the model separates the two classes across all thresholds, not just the default 0.5 cutoff
- **Cross-validation** — to check that performance isn't just a lucky train/test split

## Results

| Model | Accuracy |
|---|---|
| XGBoost | ~0.97 |
| Random Forest | ~0.96 |
| Gradient Boosting | ~0.96 |

*(Exact numbers will vary slightly by run/seed — see your console output after running `train.py`.)*

Top predictive features (by Random Forest importance) tend to include `concave_points_worst`, `perimeter_worst`, and `radius_worst` — consistent with the clinical intuition that larger, more irregular cell shapes are associated with malignancy.

## Project Structure

```
breast-cancer-detection/
├── data/                   # Place data.csv here (not committed to git)
├── models/                 # Saved trained model (.pkl)
├── figures/                # Saved plots (confusion matrices, ROC curves, etc.)
├── train.py                # Main training & evaluation script
├── predict.py               # Run predictions with the saved model
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/breast-cancer-detection.git
cd breast-cancer-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get the dataset
Download `data.csv` from [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) and place it in the `data/` folder:
```
data/data.csv
```

### 4. Train the models
```bash
python train.py
```
This prints accuracy/classification reports for all three models, saves plots to `figures/`, and saves the best model to `models/breast_cancer_model.pkl`.

### 5. Run predictions on new data
```bash
python predict.py --input data/data.csv --has-labels
```

## Tech Stack

- **Python 3**
- **pandas / numpy** — data handling
- **matplotlib / seaborn** — visualization
- **scikit-learn** — Random Forest, Gradient Boosting, train/test split, metrics
- **XGBoost** — gradient-boosted trees
- **joblib** — model persistence

## Future Improvements

- Hyperparameter tuning (GridSearchCV / Optuna)
- SHAP values for model interpretability
- A simple web demo (Streamlit/Flask) for interactive predictions
- Class-imbalance handling (SMOTE or class weighting) if extended to other datasets

## Dataset Citation

Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1995). Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B

## License

MIT License — see `LICENSE` for details.
