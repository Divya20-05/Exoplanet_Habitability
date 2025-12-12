"""
Quick dataset inspection script to understand structure and target variable
"""
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('phl_exoplanet_catalog_2019.csv')

print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"\nDataset shape: {df.shape}")

print("\n" + "=" * 80)
print("TARGET VARIABLE: P_HABITABLE")
print("=" * 80)
print(df['P_HABITABLE'].value_counts())
print(f"\nMissing values in target: {df['P_HABITABLE'].isna().sum()}")

# Calculate imbalance ratio
if df['P_HABITABLE'].nunique() == 2:
    value_counts = df['P_HABITABLE'].value_counts()
    majority_class = value_counts.max()
    minority_class = value_counts.min()
    imbalance_ratio = majority_class / minority_class
    print(f"\nImbalance Ratio: {imbalance_ratio:.2f}:1")
    print(f"Minority class percentage: {(minority_class/len(df))*100:.2f}%")

print("\n" + "=" * 80)
print("COLUMN NAMES AND DATA TYPES")
print("=" * 80)
print(df.dtypes)

print("\n" + "=" * 80)
print("FIRST FEW ROWS")
print("=" * 80)
print(df.head())

print("\n" + "=" * 80)
print("MISSING VALUES SUMMARY")
print("=" * 80)
missing_summary = df.isnull().sum()
missing_pct = (missing_summary / len(df)) * 100
missing_df = pd.DataFrame({
    'Column': missing_summary.index,
    'Missing_Count': missing_summary.values,
    'Missing_Percentage': missing_pct.values
})
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
print(missing_df.head(20))
print(f"\nTotal columns with missing values: {len(missing_df)}")
