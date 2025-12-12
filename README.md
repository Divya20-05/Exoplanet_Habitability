# Exoplanet Habitability Classification - Imbalanced Dataset Workflow

A comprehensive, educational machine learning project for handling severely imbalanced exoplanet habitability classification using various sampling techniques and advanced algorithms.

## 📊 Project Overview

**Dataset**: PHL Exoplanet Catalog 2019
- **Total Exoplanets**: 4,048
- **Features**: 112 → 92 (after cleaning) → 6,509 (after encoding)
- **Target**: Planet Habitability (P_HABITABLE)
- **Class Distribution**:
  - Class 0 (Non-Habitable): 3,993 (98.6%)
  - Class 1 (Habitable): 21 (0.5%)
  - Class 2 (Optimistic Habitable): 34 (0.8%)

**Imbalance Ratio**: 190:1 (extreme!)

---

## 🎯 Project Goals

1. **Educational**: Learn and implement imbalanced classification techniques step-by-step
2. **Comprehensive**: Cover entire ML pipeline from data quality to deployment
3. **Production-Ready**: Build pipelines that prevent data leakage and ensure reproducibility

---

## 🚀 Phases Completed (4/16)

### ✅ Phase 1: Project Setup
- Configured project structure
- Installed dependencies
- Identified extreme class imbalance

### ✅ Phase 2: Data Quality Assessment
- Analyzed missing values in 112 columns
- Detected outliers using IQR method
- Generated distribution visualizations
- **Key Finding**: 20 columns with >80% missing data

### ✅ Phase 3: Data Cleaning
- Removed 20 columns with excessive missing data
- Imputed 68 numerical columns using **median** (robust to outliers)
- Imputed 4 categorical columns using **mode**
- Capped outliers in 76 columns using IQR method
- **Result**: 4,048 rows × 92 columns with zero missing values

### ✅ Phase 4: Encoding & Scaling
- **One-hot encoded** 8 categorical features → 6,432 binary columns
- **Scaled** 77 numerical features using StandardScaler (mean=0, std=1)
- Built production-ready sklearn Pipeline
- **Result**: 4,048 × 6,509 features, ready for ML

---

## 📂 Project Structure

```
Infosys_Exoplanet/
├── config.py                          # Central configuration
├── requirements.txt                   # Python dependencies
├── phl_exoplanet_catalog_2019.csv    # Original dataset
├── 01_data_quality_assessment.py     # Phase 2: Quality analysis
├── 02_data_cleaning.py               # Phase 3: Cleaning & imputation
├── 03_encoding_scaling.py            # Phase 4: Encoding & scaling
└── outputs/
    ├── reports/                      # CSV reports
    │   ├── missing_values_report.csv
    │   ├── outlier_analysis.csv
    │   ├── imputation_report_numerical.csv
    │   └── cleaning_summary.csv
    ├── plots/                        # Visualizations
    │   ├── missing_values_analysis.png
    │   ├── feature_boxplots.png
    │   └── feature_scaling_comparison.png
    ├── processed_data/               # Clean datasets
    │   ├── exoplanet_data_cleaned.csv
    │   └── features_encoded_scaled.csv
    └── models/                       # Saved pipelines
        └── preprocessing_pipeline.pkl
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Required Libraries:**
- pandas >= 2.2.0
- numpy >= 1.26.0
- scikit-learn >= 1.4.0
- xgboost >= 2.0.0
- imbalanced-learn >= 0.12.0
- matplotlib >= 3.8.0
- seaborn >= 0.13.0

---

## 🏃 Running the Project

### Phase 2: Data Quality Assessment
```bash
python3 01_data_quality_assessment.py
```
**Output**: Missing value analysis, outlier detection, distribution plots

### Phase 3: Data Cleaning
```bash
python3 02_data_cleaning.py
```
**Output**: Clean dataset with zero missing values

### Phase 4: Encoding & Scaling
```bash
python3 03_encoding_scaling.py
```
**Output**: Scaled features ready for machine learning

---

## 🎓 Key Concepts & Learnings

### Why Median Over Mean?
```
Dataset: [1, 2, 3, 100]
Mean:    26.5  ← Skewed by outlier!
Median:  2.5   ← Robust, better!
```

### One-Hot Encoding
Converts categorical variables to binary columns without creating artificial ordering:
```
Planet Type     → Is_Terrestrial  Is_Jovian  Is_Neptunian
'Terrestrial'   →       1             0           0
'Jovian'        →       0             1           0
```

### StandardScaler
Normalizes features to mean=0, std=1:
- **Essential for**: SVM, KNN, Logistic Regression, Neural Networks
- **Not needed for**: Random Forest, XGBoost (tree-based models)

### IQR Outlier Capping
Instead of removing outliers (losing precious minority class samples), we cap them:
```
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
Outliers replaced with median
```

---

## 📊 Data Quality Summary

| Metric | Before | After |
|--------|--------|-------|
| Rows | 4,048 | 4,048 (100% retained) |
| Columns | 112 | 92 → 6,509 (after encoding) |
| Missing Values | 92 columns affected | 0 |
| Outliers | Detected in 93 columns | Capped in 76 columns |

---

## 🔜 Next Phases (5-16)

- **Phase 5**: Exploratory Data Analysis (EDA)
- **Phase 6**: PCA & t-SNE Visualization
- **Phase 7**: Baseline Modeling
- **Phase 8**: SMOTE & Sampling Techniques
- **Phase 9**: Advanced Models (XGBoost, SVM)
- **Phase 10**: Pipeline Development
- **Phase 11**: Model Evaluation
- **Phase 12**: Hyperparameter Tuning
- **Phase 13**: Feature Importance & SHAP
- **Phase 14**: Bi-Variate Analysis
- **Phase 15**: Final Model Selection
- **Phase 16**: Deployment Preparation

---

## 📈 Results So Far

✅ **100% Complete Dataset** - Zero missing values  
✅ **6,509 Features** - Properly encoded and scaled  
✅ **Production Pipeline** - Saved for deployment  
✅ **Comprehensive Documentation** - Every step explained

---

## 👨‍💻 Author

Aditya Jatling

---

## 📝 License

This project is for educational purposes.

---

## 🙏 Acknowledgments

- **PHL (Planetary Habitability Laboratory)** for the exoplanet catalog
- **imbalanced-learn** library for sampling techniques
- Educational focus on step-by-step learning with detailed explanations
