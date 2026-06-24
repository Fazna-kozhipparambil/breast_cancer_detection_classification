# 🩺 Breast Cancer Detection & Classification

A machine learning project that compares **Random Forest** and **Gradient Boosting** classifiers to predict whether a breast tumor is Malignant (M) or Benign (B), achieving up to **99% accuracy**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fazna-kozhipparambil/breast_cancer_detection_classification/blob/main/Breast_Cancer_Detection.ipynb)

---

## 🎯 Problem Statement

Breast cancer is one of the most common cancers worldwide. Early and accurate detection significantly improves survival rates. This project applies supervised machine learning to classify tumors based on 30 cell nucleus measurements — enabling faster and more reliable initial screening support.

---

## 📊 Dataset

- **Source:** Wisconsin Breast Cancer Dataset (UCI Machine Learning Repository)
- **Size:** 569 samples, 30 input features
- **Target:** Diagnosis — Malignant (M) or Benign (B)
- **Class split:** 357 Benign · 212 Malignant
- **Missing values:** 1 empty column dropped (`Unnamed: 32`), no other missing data

**Features include:** radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension — each measured as mean, standard error, and worst value.

---

## 🤖 Models Compared

### Model 1 — Random Forest Classifier

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Malignant (M) | 0.87 | 0.82 | 0.84 |
| Benign (B) | 0.83 | 0.88 | 0.85 |
| **Overall Accuracy** | | | **85%** |

### Model 2 — Gradient Boosting Classifier ✅ Best Model

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Malignant (M) | 0.99 | 0.99 | 0.99 |
| Benign (B) | 0.98 | 0.98 | 0.98 |
| **Overall Accuracy** | | | **99%** |

> **Winner: Gradient Boosting** outperformed Random Forest by 14 percentage points in accuracy. With 99% recall on malignant tumors, it almost never misses a cancer case — which is the most critical metric in medical classification.

---

## 🔬 Approach

| Step | Details |
|------|---------|
| Data Loading | Wisconsin Breast Cancer CSV dataset |
| Exploratory Analysis | Shape inspection, class distribution, sample review |
| Data Cleaning | Dropped empty column (`Unnamed: 32`) and ID column |
| Feature/Target Split | 30 features → X, diagnosis → y |
| Model 1 | Random Forest (max_depth=2) |
| Model 2 | Gradient Boosting (n_estimators=10, learning_rate=1.0) |
| Evaluation | Classification report + Confusion Matrix for both models |

---

## 🛠️ Tech Stack

`Python` · `Pandas` · `Scikit-learn` · `Matplotlib` · `Jupyter Notebook`

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/Fazna-kozhipparambil/breast_cancer_detection_classification.git
cd breast_cancer_detection_classification

# Install dependencies
pip install pandas scikit-learn matplotlib

# Launch notebook
jupyter notebook Breast_Cancer_Detection.ipynb
```

Or open directly in Google Colab using the badge above.

---

## 📁 Project Structure

```
breast_cancer_detection_classification/
│
├── Breast_Cancer_Detection.ipynb   # Main analysis and model comparison
├── data.csv                         # Dataset (add this file — see note below)
└── README.md
```

---

## 💡 What I Learned

- **Gradient Boosting significantly outperforms Random Forest** on this dataset (99% vs 85%) — demonstrating how boosting iteratively corrects errors for complex medical data
- **Recall is the key metric** in cancer detection — a false negative (missing a malignant tumor) is far more dangerous than a false positive
- **Data cleaning matters** — identifying and dropping the empty `Unnamed: 32` column and irrelevant ID column was essential before modelling
- Model comparison is valuable — always evaluate multiple algorithms rather than relying on one

---

## 👩‍💻 Author

**Fazna Kozhipparambil** — Data Science & Machine Learning  
[GitHub](https://github.com/Fazna-kozhipparambil) ·
