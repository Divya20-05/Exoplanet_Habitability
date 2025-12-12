"""
Configuration file for Exoplanet Habitability Classification Project
Central location for all project settings, paths, and hyperparameters
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path("/Users/adityajatling/Documents/Infosys_Exoplanet")
DATA_PATH = PROJECT_ROOT / "phl_exoplanet_catalog_2019.csv"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = OUTPUT_DIR / "reports"
MODELS_DIR = OUTPUT_DIR / "models"
PLOTS_DIR = OUTPUT_DIR / "plots"
PROCESSED_DATA_DIR = OUTPUT_DIR / "processed_data"

# Create directories if they don't exist
for directory in [OUTPUT_DIR, REPORTS_DIR, MODELS_DIR, PLOTS_DIR, PROCESSED_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
TARGET_VARIABLE = "P_HABITABLE"

# Dataset metadata (from inspection)
TOTAL_ROWS = 4048
TOTAL_COLUMNS = 112
CLASS_DISTRIBUTION = {
    0: 3993,  # Non-Habitable (98.6%)
    1: 21,    # Habitable (0.5%)
    2: 34     # Optimistic Habitable (0.8%)
}

# For binary classification, we can merge classes 1 and 2
# Set to True to treat classes 1 and 2 as "Habitable"
BINARY_CLASSIFICATION = True

# ============================================================================
# DATA PREPROCESSING
# ============================================================================
# Missing value thresholds
MAX_MISSING_PERCENTAGE = 80  # Drop columns with >80% missing values

# Outlier detection
IQR_MULTIPLIER = 1.5  # Standard IQR multiplier for outlier detection

# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================
TEST_SIZE = 0.20
RANDOM_STATE = 42  # For reproducibility
STRATIFY = True  # Stratified split to maintain class distribution

# ============================================================================
# CROSS-VALIDATION
# ============================================================================
CV_FOLDS = 5  # Number of folds for stratified K-Fold CV
CV_RANDOM_STATE = 42

# ============================================================================
# SAMPLING STRATEGIES
# ============================================================================
SAMPLING_STRATEGIES = {
    'smote': {'sampling_strategy': 'auto', 'random_state': 42},
    'borderline_smote': {'sampling_strategy': 'auto', 'random_state': 42, 'kind': 'borderline-1'},
    'smote_tomek': {'sampling_strategy': 'auto', 'random_state': 42},
    'adasyn': {'sampling_strategy': 'auto', 'random_state': 42},
}

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================

# Baseline Models
LOGISTIC_REGRESSION_PARAMS = {
    'max_iter': 1000,
    'random_state': 42,
    'solver': 'lbfgs'
}

SVM_PARAMS = {
    'kernel': 'linear',
    'class_weight': 'balanced',
    'random_state': 42,
    'max_iter': 1000
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1
}

# Advanced Models
XGBOOST_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 5,
    'random_state': 42,
    'eval_metric': 'logloss'
}

# ============================================================================
# EVALUATION METRICS
# ============================================================================
EVALUATION_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1',
    'roc_auc',
    'pr_auc',
    'matthews_corrcoef'
]

# Primary metric for model selection
PRIMARY_METRIC = 'f1'

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================
FIGURE_SIZE = (12, 8)
FIGURE_DPI = 100
PLOT_STYLE = 'seaborn-v0_8-darkgrid'

# Color scheme for habitability classes
CLASS_COLORS = {
    0: 'red',      # Non-Habitable
    1: 'green',    # Habitable
    2: 'blue'      # Optimistic Habitable
}

# Binary classification colors
BINARY_COLORS = {
    0: 'red',      # Non-Habitable
    1: 'green'     # Habitable (merged 1 and 2)
}

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================
TOP_N_FEATURES = 15  # Number of top features to display in importance plots

# ============================================================================
# LOGGING
# ============================================================================
LOG_LEVEL = "INFO"
LOG_FILE = OUTPUT_DIR / "project.log"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def print_config():
    """Print current configuration for verification"""
    print("=" * 80)
    print("PROJECT CONFIGURATION")
    print("=" * 80)
    print(f"Data Path: {DATA_PATH}")
    print(f"Target Variable: {TARGET_VARIABLE}")
    print(f"Binary Classification: {BINARY_CLASSIFICATION}")
    print(f"Test Size: {TEST_SIZE}")
    print(f"Random State: {RANDOM_STATE}")
    print(f"CV Folds: {CV_FOLDS}")
    print(f"Primary Metric: {PRIMARY_METRIC}")
    print("=" * 80)

if __name__ == "__main__":
    print_config()
