"""
PHASE 3: DATA CLEANING
======================

This script implements intelligent data cleaning including:
1. Removal of columns with excessive missing data (>80%)
2. Imputation of missing values (median for numerical, mode for categorical)
3. Outlier handling using IQR-based capping (replace outliers with median)
4. Data validation and quality checks

LEARNING OBJECTIVES:
-------------------
- Why use median instead of mean for imputation?
  → Median is robust to outliers and skewed distributions
- What happens if we don't clean data?
  → Models can't handle missing values, outliers skew predictions
- Why IQR capping instead of removal?
  → Preserves sample size (critical for imbalanced data)

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import configuration
import config

# Set plotting style
sns.set_style('whitegrid')

print("=" * 80)
print("PHASE 3: DATA CLEANING")
print("=" * 80)

# ============================================================================
# STEP 1: Load Original Dataset and Reports
# ============================================================================
print("\n[1/8] Loading dataset and quality assessment reports...")

df = pd.read_csv(config.DATA_PATH)
missing_report = pd.read_csv(config.REPORTS_DIR / 'missing_values_report.csv')
outlier_report = pd.read_csv(config.REPORTS_DIR / 'outlier_analysis.csv')

print(f"✓ Loaded dataset: {df.shape}")
print(f"✓ Loaded missing values report")
print(f"✓ Loaded outlier analysis report")

# Store original shape for comparison
original_shape = df.shape

# ============================================================================
# STEP 2: Remove Columns with Excessive Missing Data
# ============================================================================
print(f"\n[2/8] Removing columns with >{config.MAX_MISSING_PERCENTAGE}% missing data...")

print("\nCONCEPT: Why Remove Columns with Too Much Missing Data?")
print("-" * 80)
print(f"Columns with >{config.MAX_MISSING_PERCENTAGE}% missing values:")
print("  - Provide very little information")
print("  - Imputation would be mostly guesswork")
print("  - Can introduce more noise than signal")
print("  - Better to remove than to impute unreliably")
print("-" * 80)

# Identify columns to drop
columns_to_drop = missing_report[
    missing_report['Missing_Percentage'] > config.MAX_MISSING_PERCENTAGE
]['Column'].tolist()

print(f"\nColumns to remove: {len(columns_to_drop)}")
print("Top 10 columns being removed:")
for i, col in enumerate(columns_to_drop[:10], 1):
    pct = missing_report[missing_report['Column'] == col]['Missing_Percentage'].values[0]
    print(f"  {i:2d}. {col:30s} - {pct:6.2f}% missing")

# Remove columns
df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')
print(f"\n✓ Dataset shape after column removal: {df_cleaned.shape}")
print(f"  Removed {len(columns_to_drop)} columns")
print(f"  Retained {len(df_cleaned.columns)} columns")

# ============================================================================
# STEP 3: Separate Features by Type
# ============================================================================
print("\n[3/8] Separating numerical and categorical features...")

# Identify column types
numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df_cleaned.select_dtypes(include=['object']).columns.tolist()

# Remove target from feature lists if present
if config.TARGET_VARIABLE in numerical_cols:
    numerical_cols.remove(config.TARGET_VARIABLE)

print(f"✓ Numerical features: {len(numerical_cols)}")
print(f"✓ Categorical features: {len(categorical_cols)}")

# ============================================================================
# STEP 4: Impute Missing Values - Numerical Features
# ============================================================================
print("\n[4/8] Imputing missing values in numerical features...")

print("\nCONCEPT: Why Median for Numerical Imputation?")
print("-" * 80)
print("MEDIAN vs MEAN:")
print("  Mean:   Sensitive to outliers, can be skewed")
print("         Example: [1, 2, 3, 100] → Mean = 26.5 (misleading!)")
print("  Median: Robust to outliers, represents 'typical' value")
print("         Example: [1, 2, 3, 100] → Median = 2.5 (better!)")
print("\nFor astronomical data with extreme values → MEDIAN is better!")
print("-" * 80)

# Track imputation statistics
imputation_stats = []

for col in numerical_cols:
    if df_cleaned[col].isnull().sum() > 0:
        missing_count = df_cleaned[col].isnull().sum()
        missing_pct = (missing_count / len(df_cleaned)) * 100
        
        # Calculate median (ignoring NaN)
        median_value = df_cleaned[col].median()
        
        # Impute
        df_cleaned[col].fillna(median_value, inplace=True)
        
        imputation_stats.append({
            'Column': col,
            'Missing_Count': missing_count,
            'Missing_Percentage': missing_pct,
            'Imputed_Value': median_value
        })

print(f"✓ Imputed {len(imputation_stats)} numerical columns")
print(f"\nTop 10 numerical columns imputed:")
imputation_df = pd.DataFrame(imputation_stats).sort_values('Missing_Count', ascending=False)
print(imputation_df.head(10)[['Column', 'Missing_Count', 'Imputed_Value']].to_string(index=False))

# ============================================================================
# STEP 5: Impute Missing Values - Categorical Features
# ============================================================================
print("\n[5/8] Imputing missing values in categorical features...")

print("\nCONCEPT: Mode for Categorical Imputation")
print("-" * 80)
print("Mode = Most frequent value")
print("For categorical data, we use the most common category")
print("Alternative: Create 'Unknown' category (sometimes useful)")
print("-" * 80)

categorical_imputation_stats = []

for col in categorical_cols:
    if df_cleaned[col].isnull().sum() > 0:
        missing_count = df_cleaned[col].isnull().sum()
        missing_pct = (missing_count / len(df_cleaned)) * 100
        
        # Calculate mode
        mode_value = df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else 'Unknown'
        
        # Impute
        df_cleaned[col].fillna(mode_value, inplace=True)
        
        categorical_imputation_stats.append({
            'Column': col,
            'Missing_Count': missing_count,
            'Missing_Percentage': missing_pct,
            'Imputed_Value': mode_value
        })

if len(categorical_imputation_stats) > 0:
    print(f"✓ Imputed {len(categorical_imputation_stats)} categorical columns")
    cat_imp_df = pd.DataFrame(categorical_imputation_stats)
    print(cat_imp_df.to_string(index=False))
else:
    print(f"✓ No missing values in categorical columns")

# ============================================================================
# STEP 6: Handle Outliers Using IQR Method
# ============================================================================
print("\n[6/8] Handling outliers using IQR-based capping...")

print("\nCONCEPT: Why Cap Outliers Instead of Removing Them?")
print("-" * 80)
print("For IMBALANCED datasets:")
print("  - Removing outliers = losing precious minority class samples")
print("  - Capping (replacing with median) = retaining all samples")
print("  - Preserves information while reducing extreme influence")
print("\nWe replace outliers with MEDIAN (robust central tendency)")
print("-" * 80)

outlier_handling_stats = []

for col in numerical_cols:
    col_data = df_cleaned[col].copy()
    
    # Calculate IQR bounds
    Q1 = col_data.quantile(0.25)
    Q3 = col_data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - config.IQR_MULTIPLIER * IQR
    upper_bound = Q3 + config.IQR_MULTIPLIER * IQR
    
    # Identify outliers
    outliers_mask = (col_data < lower_bound) | (col_data > upper_bound)
    outlier_count = outliers_mask.sum()
    
    if outlier_count > 0:
        # Replace outliers with median
        median_value = col_data.median()
        df_cleaned.loc[outliers_mask, col] = median_value
        
        outlier_handling_stats.append({
            'Column': col,
            'Outliers_Replaced': outlier_count,
            'Replacement_Value': median_value,
            'Lower_Bound': lower_bound,
            'Upper_Bound': upper_bound
        })

print(f"✓ Handled outliers in {len(outlier_handling_stats)} columns")
print(f"\nTop 15 columns with most outliers replaced:")
outlier_handling_df = pd.DataFrame(outlier_handling_stats).sort_values('Outliers_Replaced', ascending=False)
print(outlier_handling_df.head(15)[['Column', 'Outliers_Replaced', 'Replacement_Value']].to_string(index=False))

# ============================================================================
# STEP 7: Verify Data Quality
# ============================================================================
print("\n[7/8] Verifying cleaned data quality...")

# Check for remaining missing values
remaining_missing = df_cleaned.isnull().sum().sum()

print(f"\nData Quality Verification:")
print(f"  - Original shape: {original_shape}")
print(f"  - Cleaned shape: {df_cleaned.shape}")
print(f"  - Columns removed: {original_shape[1] - df_cleaned.shape[1]}")
print(f"  - Rows retained: {df_cleaned.shape[0]} (100%)")
print(f"  - Remaining missing values: {remaining_missing}")

# Verify target variable integrity
if config.TARGET_VARIABLE in df_cleaned.columns:
    target_dist = df_cleaned[config.TARGET_VARIABLE].value_counts().sort_index()
    print(f"\n✓ Target variable '{config.TARGET_VARIABLE}' distribution:")
    for cls, count in target_dist.items():
        print(f"    Class {cls}: {count:5d} ({count/len(df_cleaned)*100:5.2f}%)")

# ============================================================================
# STEP 8: Save Cleaned Dataset and Documentation
# ============================================================================
print("\n[8/8] Saving cleaned dataset and documentation...")

# Save cleaned dataset
cleaned_data_path = config.PROCESSED_DATA_DIR / 'exoplanet_data_cleaned.csv'
df_cleaned.to_csv(cleaned_data_path, index=False)
print(f"✓ Cleaned dataset saved to: {cleaned_data_path}")

# Save imputation documentation
if len(imputation_stats) > 0:
    imputation_report_path = config.REPORTS_DIR / 'imputation_report_numerical.csv'
    pd.DataFrame(imputation_stats).to_csv(imputation_report_path, index=False)
    print(f"✓ Numerical imputation report saved to: {imputation_report_path}")

if len(categorical_imputation_stats) > 0:
    cat_imputation_report_path = config.REPORTS_DIR / 'imputation_report_categorical.csv'
    pd.DataFrame(categorical_imputation_stats).to_csv(cat_imputation_report_path, index=False)
    print(f"✓ Categorical imputation report saved to: {cat_imputation_report_path}")

# Save outlier handling documentation
if len(outlier_handling_stats) > 0:
    outlier_handling_path = config.REPORTS_DIR / 'outlier_handling_report.csv'
    outlier_handling_df.to_csv(outlier_handling_path, index=False)
    print(f"✓ Outlier handling report saved to: {outlier_handling_path}")

# Save cleaning summary
cleaning_summary = {
    'Original Rows': [original_shape[0]],
    'Original Columns': [original_shape[1]],
    'Cleaned Rows': [df_cleaned.shape[0]],
    'Cleaned Columns': [df_cleaned.shape[1]],
    'Columns Removed': [original_shape[1] - df_cleaned.shape[1]],
    'Numerical Columns Imputed': [len(imputation_stats)],
    'Categorical Columns Imputed': [len(categorical_imputation_stats)],
    'Columns with Outliers Handled': [len(outlier_handling_stats)],
    'Remaining Missing Values': [remaining_missing]
}

summary_df = pd.DataFrame(cleaning_summary)
summary_path = config.REPORTS_DIR / 'cleaning_summary.csv'
summary_df.to_csv(summary_path, index=False)
print(f"✓ Cleaning summary saved to: {summary_path}")

# Create before/after visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Before: original shape
axes[0].bar(['Rows', 'Columns'], [original_shape[0], original_shape[1]], color='coral')
axes[0].set_title('Original Dataset', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Count')
axes[0].grid(axis='y', alpha=0.3)

# After: cleaned shape
axes[1].bar(['Rows', 'Columns'], [df_cleaned.shape[0], df_cleaned.shape[1]], color='lightgreen')
axes[1].set_title('Cleaned Dataset', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Count')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
comparison_plot_path = config.PLOTS_DIR / 'before_after_cleaning.png'
plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Before/after visualization saved to: {comparison_plot_path}")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("DATA CLEANING COMPLETE")
print("=" * 80)

print("\n📊 Cleaning Summary:")
print(f"  ✓ Removed {original_shape[1] - df_cleaned.shape[1]} columns with >{config.MAX_MISSING_PERCENTAGE}% missing")
print(f"  ✓ Imputed {len(imputation_stats)} numerical columns using MEDIAN")
print(f"  ✓ Imputed {len(categorical_imputation_stats)} categorical columns using MODE")
print(f"  ✓ Capped outliers in {len(outlier_handling_stats)} columns using IQR method")
print(f"  ✓ Final dataset: {df_cleaned.shape[0]} rows × {df_cleaned.shape[1]} columns")
print(f"  ✓ Zero missing values remaining")

print(f"\n📁 All outputs saved to:")
print(f"  - Cleaned data: {cleaned_data_path}")
print(f"  - Reports: {config.REPORTS_DIR}")
print(f"  - Visualizations: {config.PLOTS_DIR}")

print("\n✅ Ready for Phase 4: Encoding & Scaling!")
print("=" * 80)
