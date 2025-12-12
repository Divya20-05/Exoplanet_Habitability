"""
PHASE 4: ENCODING & SCALING
============================

This script implements feature transformation including:
1. One-Hot Encoding for categorical variables
2. StandardScaler for numerical features
3. Production-ready sklearn Pipeline
4. Prevention of data leakage

LEARNING OBJECTIVES:
-------------------
- What is One-Hot Encoding and why do we need it?
- What is StandardScaler and why is scaling important?
- What happens if we DON'T scale features?
- How to build pipelines to prevent data leakage?

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import configuration
import config

# Set plotting style
sns.set_style('whitegrid')

print("=" * 80)
print("PHASE 4: ENCODING & SCALING")
print("=" * 80)

# ============================================================================
# STEP 1: Load Cleaned Dataset
# ============================================================================
print("\n[1/7] Loading cleaned dataset...")

cleaned_data_path = config.PROCESSED_DATA_DIR / 'exoplanet_data_cleaned.csv'
df = pd.read_csv(cleaned_data_path)

print(f"✓ Loaded cleaned dataset: {df.shape}")
print(f"✓ Zero missing values: {df.isnull().sum().sum() == 0}")

# ============================================================================
# STEP 2: Separate Features and Target
# ============================================================================
print("\n[2/7] Separating features and target variable...")

# Extract target
y = df[config.TARGET_VARIABLE].copy()

# Drop target and identifier columns
identifier_cols = ['P_NAME', 'S_NAME', 'S_ALT_NAMES', 'S_CONSTELLATION', 
                   'S_CONSTELLATION_ABR', 'S_CONSTELLATION_ENG']
cols_to_drop = [config.TARGET_VARIABLE] + [col for col in identifier_cols if col in df.columns]

X = df.drop(columns=cols_to_drop, errors='ignore')

print(f"✓ Features shape: {X.shape}")
print(f"✓ Target shape: {y.shape}")
print(f"\nTarget distribution:")
print(y.value_counts().sort_index())

# ============================================================================
# STEP 3: Identify Numerical and Categorical Features
# ============================================================================
print("\n[3/7] Identifying feature types...")

numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"✓ Numerical features: {len(numerical_features)}")
print(f"✓ Categorical features: {len(categorical_features)}")

if len(categorical_features) > 0:
    print(f"\nCategorical features found:")
    for i, col in enumerate(categorical_features, 1):
        unique_vals = X[col].nunique()
        print(f"  {i}. {col:30s} - {unique_vals} unique values")

# ============================================================================
# STEP 4: ONE-HOT ENCODING Explanation and Implementation
# ============================================================================
print("\n[4/7] Applying One-Hot Encoding to categorical features...")

print("\n" + "=" * 80)
print("CONCEPT: What is One-Hot Encoding?")
print("=" * 80)
print("Machine learning algorithms work with NUMBERS, not text!")
print("\nExample: Planet Type = ['Terrestrial', 'Jovian', 'Neptunian']")
print("\n❌ BAD APPROACH: Label Encoding")
print("   Terrestrial → 0")
print("   Jovian      → 1")
print("   Neptunian   → 2")
print("   Problem: Algorithm thinks Neptunian (2) > Jovian (1)!")
print("   But there's NO ordinal relationship!")
print("\n✅ GOOD APPROACH: One-Hot Encoding")
print("   Create separate binary column for each category:")
print("   Original        → Is_Terrestrial  Is_Jovian  Is_Neptunian")
print("   'Terrestrial'   →       1             0           0")
print("   'Jovian'        →       0             1           0")
print("   'Neptunian'     →       0             0           1")
print("\n   Now no artificial ordering exists!")
print("=" * 80)

# Show example before encoding
if len(categorical_features) > 0:
    sample_cat_col = categorical_features[0]
    print(f"\nExample: '{sample_cat_col}' unique values:")
    print(X[sample_cat_col].value_counts().head())

# ============================================================================
# STEP 5: STANDARDSCALER Explanation
# ============================================================================
print("\n[5/7] Explaining StandardScaler...")

print("\n" + "=" * 80)
print("CONCEPT: Why Do We Need Scaling?")
print("=" * 80)
print("Features have DIFFERENT SCALES:")
print("  - Planet Mass:        1 to 13,000 Jupiter masses")
print("  - Orbital Period:     0.09 to 730,000 days")
print("  - Distance from star: 0.01 to 500 AU")
print("\n❌ WITHOUT SCALING:")
print("   Algorithms like SVM, KNN, Neural Networks get confused!")
print("   Features with larger scales DOMINATE the model")
print("   Example: Planet mass (13,000) >> Eccentricity (0.9)")
print("           Algorithm ignores eccentricity!")
print("\n✅ WITH STANDARDSCALER:")
print("   Formula: z = (x - mean) / std_deviation")
print("   Result: All features have mean=0, std=1")
print("   Now all features contribute EQUALLY")
print("\nAlgorithms that NEED scaling:")
print("  ✓ SVM (Support Vector Machines)")
print("  ✓ KNN (K-Nearest Neighbors)")
print("  ✓ Logistic Regression")
print("  ✓ Neural Networks")
print("\nAlgorithms that DON'T need scaling:")
print("  - Tree-based: Random Forest, XGBoost, Decision Trees")
print("  (They split on individual features, scale doesn't matter)")
print("=" * 80)

# Show example of feature scales
print("\nExample: Feature Scales BEFORE Scaling")
print("-" * 80)
sample_features = numerical_features[:5]
for feat in sample_features:
    print(f"{feat:30s}: min={X[feat].min():12.4f}, max={X[feat].max():12.4f}, "
          f"mean={X[feat].mean():12.4f}")

# ============================================================================
# STEP 6: Build Preprocessing Pipeline
# ============================================================================
print("\n[6/7] Building preprocessing pipeline...")

print("\nCONCEPT: Why Use Pipeline?")
print("-" * 80)
print("Pipeline ensures:")
print("  1. No data leakage (fit only on training data)")
print("  2. Same transformations applied to train and test")
print("  3. Easy to save and deploy")
print("  4. Production-ready code")
print("-" * 80)

# Create preprocessing pipeline
if len(categorical_features) > 0:
    # Pipeline with both numerical and categorical
    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical', StandardScaler(), numerical_features),
            ('categorical', OneHotEncoder(drop='first', sparse_output=False, 
                                         handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    print(f"\n✓ Created pipeline with:")
    print(f"  - StandardScaler for {len(numerical_features)} numerical features")
    print(f"  - OneHotEncoder for {len(categorical_features)} categorical features")
else:
    # Pipeline with only numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical', StandardScaler(), numerical_features)
        ],
        remainder='passthrough'
    )
    print(f"\n✓ Created pipeline with:")
    print(f"  - StandardScaler for {len(numerical_features)} numerical features")
    print(f"  - No categorical features found")

# Fit and transform the data
X_transformed = preprocessor.fit_transform(X)

# Get feature names after transformation
if len(categorical_features) > 0:
    # Get one-hot encoded feature names
    encoder = preprocessor.named_transformers_['categorical']
    encoded_features = encoder.get_feature_names_out(categorical_features)
    all_feature_names = numerical_features + list(encoded_features)
else:
    all_feature_names = numerical_features

print(f"\n✓ Transformation complete!")
print(f"  Original features: {X.shape[1]}")
print(f"  Transformed features: {X_transformed.shape[1]}")
if len(categorical_features) > 0:
    print(f"  (One-hot encoding expanded {len(categorical_features)} categorical → "
          f"{len(all_feature_names) - len(numerical_features)} binary columns)")

# Create DataFrame with transformed data
X_transformed_df = pd.DataFrame(X_transformed, columns=all_feature_names, index=X.index)

# Show scaling effect
print("\nExample: Feature Scales AFTER Scaling")
print("-" * 80)
for feat in sample_features:
    scaled_feat = X_transformed_df[feat]
    print(f"{feat:30s}: min={scaled_feat.min():12.4f}, max={scaled_feat.max():12.4f}, "
          f"mean={scaled_feat.mean():12.4f}, std={scaled_feat.std():12.4f}")

# ============================================================================
# STEP 7: Save Processed Data and Pipeline
# ============================================================================
print("\n[7/7] Saving processed data and preprocessing pipeline...")

# Save transformed features
processed_features_path = config.PROCESSED_DATA_DIR / 'features_encoded_scaled.csv'
X_transformed_df.to_csv(processed_features_path, index=False)
print(f"✓ Scaled features saved to: {processed_features_path}")

# Save target variable
target_path = config.PROCESSED_DATA_DIR / 'target.csv'
y.to_csv(target_path, index=False, header=True)
print(f"✓ Target variable saved to: {target_path}")

# Save feature names
feature_names_path = config.PROCESSED_DATA_DIR / 'feature_names.csv'
pd.DataFrame({'Feature': all_feature_names}).to_csv(feature_names_path, index=False)
print(f"✓ Feature names saved to: {feature_names_path}")

# Save preprocessing pipeline using joblib
import joblib
pipeline_path = config.MODELS_DIR / 'preprocessing_pipeline.pkl'
joblib.dump(preprocessor, pipeline_path)
print(f"✓ Preprocessing pipeline saved to: {pipeline_path}")

# Create visualization: Before vs After Scaling
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Feature Scaling: Before vs After', fontsize=16, fontweight='bold')

# Select 3 sample numerical features
sample_viz_features = numerical_features[:3]

for idx, feat in enumerate(sample_viz_features):
    # Before scaling
    axes[0, idx].hist(X[feat], bins=50, color='coral', edgecolor='black', alpha=0.7)
    axes[0, idx].set_title(f'BEFORE: {feat}', fontsize=11, fontweight='bold')
    axes[0, idx].set_xlabel('Original Scale')
    axes[0, idx].set_ylabel('Frequency')
    axes[0, idx].axvline(X[feat].mean(), color='red', linestyle='--', 
                         linewidth=2, label=f'Mean: {X[feat].mean():.2f}')
    axes[0, idx].legend()
    
    # After scaling
    axes[1, idx].hist(X_transformed_df[feat], bins=50, color='lightgreen', 
                      edgecolor='black', alpha=0.7)
    axes[1, idx].set_title(f'AFTER: {feat}', fontsize=11, fontweight='bold')
    axes[1, idx].set_xlabel('Scaled (mean=0, std=1)')
    axes[1, idx].set_ylabel('Frequency')
    axes[1, idx].axvline(X_transformed_df[feat].mean(), color='blue', linestyle='--', 
                         linewidth=2, label=f'Mean: {X_transformed_df[feat].mean():.2f}')
    axes[1, idx].axvline(X_transformed_df[feat].mean() - X_transformed_df[feat].std(), 
                         color='green', linestyle=':', alpha=0.7, label='-1 std')
    axes[1, idx].axvline(X_transformed_df[feat].mean() + X_transformed_df[feat].std(), 
                         color='green', linestyle=':', alpha=0.7, label='+1 std')
    axes[1, idx].legend()

plt.tight_layout()
scaling_viz_path = config.PLOTS_DIR / 'feature_scaling_comparison.png'
plt.savefig(scaling_viz_path, dpi=300, bbox_inches='tight')
print(f"✓ Scaling visualization saved to: {scaling_viz_path}")
plt.close()

# Create encoding visualization if categorical features exist
if len(categorical_features) > 0:
    print(f"\n✓ One-hot encoding created {len(all_feature_names) - len(numerical_features)} new binary features")
    
    # Show encoded feature names
    encoded_feature_names = [f for f in all_feature_names if f not in numerical_features]
    print(f"\nNew encoded features (first 10):")
    for i, feat in enumerate(encoded_feature_names[:10], 1):
        print(f"  {i:2d}. {feat}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("ENCODING & SCALING COMPLETE")
print("=" * 80)

print("\n📊 Transformation Summary:")
print(f"  ✓ Original dataset: {X.shape[0]:,} samples × {X.shape[1]} features")
print(f"  ✓ Transformed dataset: {X_transformed.shape[0]:,} samples × {X_transformed.shape[1]} features")
print(f"  ✓ Numerical features scaled: {len(numerical_features)}")
if len(categorical_features) > 0:
    print(f"  ✓ Categorical features encoded: {len(categorical_features)}")
    print(f"    → Created {len(all_feature_names) - len(numerical_features)} binary features")
print(f"\n  ✓ All features now have mean ≈ 0, std ≈ 1")
print(f"  ✓ Ready for machine learning algorithms!")

print(f"\n📁 Files Saved:")
print(f"  1. Scaled features: {processed_features_path}")
print(f"  2. Target variable: {target_path}")
print(f"  3. Feature names: {feature_names_path}")
print(f"  4. Preprocessing pipeline: {pipeline_path}")
print(f"  5. Visualization: {scaling_viz_path}")

print("\n🎓 Key Takeaways:")
print("  1. One-Hot Encoding: Converts categorical → binary (no artificial order)")
print("  2. StandardScaler: Makes all features comparable (mean=0, std=1)")
print("  3. Pipeline: Prevents data leakage, ensures reproducibility")
print("  4. Scaling matters for: SVM, KNN, Logistic Regression, Neural Nets")
print("  5. Scaling optional for: Random Forest, XGBoost (tree-based)")

print("\n✅ Ready for Phase 5: Exploratory Data Analysis (EDA)!")
print("=" * 80)
