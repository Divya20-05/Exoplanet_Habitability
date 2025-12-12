"""
PHASE 2: DATA QUALITY ASSESSMENT
=================================

This script performs comprehensive data quality analysis including:
1. Missing value analysis for every feature
2. Outlier detection using boxplots and IQR method
3. Distribution visualization for all features
4. Descriptive statistics generation

LEARNING OBJECTIVES:
-------------------
- Understand the importance of data quality before modeling
- Learn about missing data patterns (MCAR, MAR, MNAR)
- Identify outliers and their impact on model performance
- Visualize feature distributions to understand data characteristics

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
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("=" * 80)
print("PHASE 2: DATA QUALITY ASSESSMENT")
print("=" * 80)

# ============================================================================
# STEP 1: Load Dataset
# ============================================================================
print("\n[1/6] Loading dataset...")
df = pd.read_csv(config.DATA_PATH)
print(f"✓ Loaded {len(df)} rows and {len(df.columns)} columns")

# ============================================================================
# STEP 2: Missing Values Analysis
# ============================================================================
print("\n[2/6] Analyzing missing values...")

# Calculate missing value statistics
missing_stats = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum().values,
    'Missing_Percentage': (df.isnull().sum().values / len(df) * 100)
})

# Sort by missing percentage
missing_stats = missing_stats.sort_values('Missing_Percentage', ascending=False)

# Save to CSV
missing_report_path = config.REPORTS_DIR / 'missing_values_report.csv'
missing_stats.to_csv(missing_report_path, index=False)
print(f"✓ Missing values report saved to: {missing_report_path}")

# Print summary
print(f"\nMissing Values Summary:")
print(f"  - Total columns: {len(df.columns)}")
print(f"  - Columns with missing data: {(missing_stats['Missing_Count'] > 0).sum()}")
print(f"  - Columns with >50% missing: {(missing_stats['Missing_Percentage'] > 50).sum()}")
print(f"  - Columns with >80% missing: {(missing_stats['Missing_Percentage'] > 80).sum()}")

print(f"\nTop 15 columns with most missing data:")
print(missing_stats.head(15).to_string(index=False))

# Visualize missing values
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Bar chart of top 20 columns with missing values
top_missing = missing_stats[missing_stats['Missing_Count'] > 0].head(20)
axes[0].barh(top_missing['Column'], top_missing['Missing_Percentage'], color='coral')
axes[0].set_xlabel('Missing Percentage (%)', fontsize=12)
axes[0].set_title('Top 20 Columns with Missing Values', fontsize=14, fontweight='bold')
axes[0].axvline(x=50, color='red', linestyle='--', label='50% threshold')
axes[0].axvline(x=80, color='darkred', linestyle='--', label='80% threshold')
axes[0].legend()
axes[0].invert_yaxis()

# Plot 2: Distribution of missing percentages
axes[1].hist(missing_stats['Missing_Percentage'], bins=50, color='steelblue', edgecolor='black')
axes[1].set_xlabel('Missing Percentage (%)', fontsize=12)
axes[1].set_ylabel('Number of Columns', fontsize=12)
axes[1].set_title('Distribution of Missing Percentages Across All Columns', fontsize=14, fontweight='bold')
axes[1].axvline(x=50, color='red', linestyle='--', label='50% threshold')
axes[1].axvline(x=80, color='darkred', linestyle='--', label='80% threshold')
axes[1].legend()

plt.tight_layout()
missing_plot_path = config.PLOTS_DIR / 'missing_values_analysis.png'
plt.savefig(missing_plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Missing values visualization saved to: {missing_plot_path}")
plt.close()

# ============================================================================
# STEP 3: Missing Data Patterns (MCAR/MAR/MNAR Concept)
# ============================================================================
print("\n[3/6] Analyzing missing data patterns...")

print("\nCONCEPT: Types of Missing Data")
print("-" * 80)
print("1. MCAR (Missing Completely At Random):")
print("   - Missing values are randomly distributed")
print("   - No relationship between missingness and any variable")
print("   - Safe to drop or impute")
print()
print("2. MAR (Missing At Random):")
print("   - Missingness depends on observed data")
print("   - Example: Income missing more often for younger people")
print("   - Can be handled with careful imputation")
print()
print("3. MNAR (Missing Not At Random):")
print("   - Missingness depends on the missing value itself")
print("   - Example: High earners don't report income")
print("   - Most problematic; requires domain knowledge")
print("-" * 80)

# For this dataset, we'll check if missingness correlates with target variable
print("\nChecking if missingness correlates with habitability class...")
for col in missing_stats.head(10)['Column']:
    if missing_stats[missing_stats['Column'] == col]['Missing_Percentage'].values[0] < 100:
        df[f'{col}_is_missing'] = df[col].isnull().astype(int)
        
# ============================================================================
# STEP 4: Descriptive Statistics
# ============================================================================
print("\n[4/6] Generating descriptive statistics...")

# Separate numerical and categorical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"\n✓ Found {len(numerical_cols)} numerical columns")
print(f"✓ Found {len(categorical_cols)} categorical columns")

# Numerical statistics
numerical_stats = df[numerical_cols].describe()
numerical_stats_path = config.REPORTS_DIR / 'numerical_statistics.csv'
numerical_stats.to_csv(numerical_stats_path)
print(f"✓ Numerical statistics saved to: {numerical_stats_path}")

# ============================================================================
# STEP 5: Outlier Detection using IQR Method
# ============================================================================
print("\n[5/6] Detecting outliers using IQR method...")

print("\nCONCEPT: IQR (Interquartile Range) Method")
print("-" * 80)
print("IQR = Q3 - Q1 (difference between 75th and 25th percentiles)")
print(f"Lower Bound = Q1 - {config.IQR_MULTIPLIER} * IQR")
print(f"Upper Bound = Q3 + {config.IQR_MULTIPLIER} * IQR")
print("Values outside these bounds are considered outliers")
print("-" * 80)

# Calculate outliers for each numerical column
outlier_summary = []

for col in numerical_cols:
    if col != config.TARGET_VARIABLE:  # Skip target variable
        # Remove NaN values for calculation
        col_data = df[col].dropna()
        
        if len(col_data) > 0:
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - config.IQR_MULTIPLIER * IQR
            upper_bound = Q3 + config.IQR_MULTIPLIER * IQR
            
            outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
            outlier_pct = (outliers / len(col_data)) * 100
            
            outlier_summary.append({
                'Column': col,
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'Lower_Bound': lower_bound,
                'Upper_Bound': upper_bound,
                'Outlier_Count': outliers,
                'Outlier_Percentage': outlier_pct
            })

outlier_df = pd.DataFrame(outlier_summary)
outlier_df = outlier_df.sort_values('Outlier_Percentage', ascending=False)

outlier_report_path = config.REPORTS_DIR / 'outlier_analysis.csv'
outlier_df.to_csv(outlier_report_path, index=False)
print(f"✓ Outlier analysis saved to: {outlier_report_path}")

print(f"\nTop 15 columns with most outliers:")
print(outlier_df.head(15)[['Column', 'Outlier_Count', 'Outlier_Percentage']].to_string(index=False))

# ============================================================================
# STEP 6: Distribution Visualization
# ============================================================================
print("\n[6/6] Creating distribution visualizations...")
print("This may take a few minutes for 112 features...")

# Create boxplots for numerical features (top 30 with least missing data)
valid_numerical_cols = missing_stats[
    (missing_stats['Column'].isin(numerical_cols)) & 
    (missing_stats['Missing_Percentage'] < 80)
]['Column'].head(30).tolist()

if len(valid_numerical_cols) > 0:
    # Create boxplots
    n_cols = 5
    n_rows = (len(valid_numerical_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, col in enumerate(valid_numerical_cols):
        if idx < len(axes):
            df.boxplot(column=col, ax=axes[idx])
            axes[idx].set_title(col, fontsize=10)
            axes[idx].tick_params(labelsize=8)
    
    # Hide empty subplots
    for idx in range(len(valid_numerical_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    boxplot_path = config.PLOTS_DIR / 'feature_boxplots.png'
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Boxplots saved to: {boxplot_path}")
    plt.close()

    # Create histograms
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, col in enumerate(valid_numerical_cols):
        if idx < len(axes):
            df[col].hist(ax=axes[idx], bins=30, color='skyblue', edgecolor='black')
            axes[idx].set_title(col, fontsize=10)
            axes[idx].set_xlabel('')
            axes[idx].tick_params(labelsize=8)
    
    # Hide empty subplots
    for idx in range(len(valid_numerical_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    hist_path = config.PLOTS_DIR / 'feature_histograms.png'
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    print(f"✓ Histograms saved to: {hist_path}")
    plt.close()

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "=" * 80)
print("DATA QUALITY ASSESSMENT COMPLETE")
print("=" * 80)
print("\nKey Findings:")
print(f"1. Missing Values:")
print(f"   - {(missing_stats['Missing_Count'] > 0).sum()} columns have missing data")
print(f"   - {(missing_stats['Missing_Percentage'] > 80).sum()} columns have >80% missing (candidates for removal)")

print(f"\n2. Outliers:")
print(f"   - Analysis completed for {len(outlier_df)} numerical columns")
print(f"   - Outliers detected using IQR method (multiplier = {config.IQR_MULTIPLIER})")

print(f"\n3. Data Types:")
print(f"   - {len(numerical_cols)} numerical features")
print(f"   - {len(categorical_cols)} categorical features")

print(f"\nAll reports saved to: {config.REPORTS_DIR}")
print(f"All plots saved to: {config.PLOTS_DIR}")
print("=" * 80)
