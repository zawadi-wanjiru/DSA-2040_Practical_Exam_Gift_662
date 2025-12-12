"""
DSA 2040 Practical Exam - Section 2, Task 1
Data Preprocessing and Exploration
Student: Gift Wanjiru Gachunga (672662)

This script performs:
- Data loading from scikit-learn Iris dataset
- Preprocessing (missing values, normalization, encoding)
- Exploratory data analysis (statistics, visualizations)
- Train/test split
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(672662)

# Configure plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_data():
    """
    Load the Iris dataset from scikit-learn

    Returns:
        pandas DataFrame with features and target
    """
    print("=" * 70)
    print("LOADING IRIS DATASET")
    print("=" * 70)

    # Load iris dataset
    iris = load_iris()

    # Create DataFrame
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target

    # Map target numbers to species names
    species_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    df['species_name'] = df['species'].map(species_names)

    print(f"✓ Loaded Iris dataset")
    print(f"  - Samples: {len(df)}")
    print(f"  - Features: {len(iris.feature_names)}")
    print(f"  - Classes: {len(iris.target_names)}")
    print(f"  - Feature names: {iris.feature_names}")
    print(f"  - Class names: {list(iris.target_names)}")

    return df


def check_missing_values(df):
    """
    Check for missing values in the dataset

    Args:
        df: pandas DataFrame

    Returns:
        DataFrame with missing value information
    """
    print("\n" + "=" * 70)
    print("CHECKING FOR MISSING VALUES")
    print("=" * 70)

    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100

    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing_Count': missing.values,
        'Missing_Percentage': missing_pct.values
    })

    print(missing_df.to_string(index=False))

    total_missing = missing.sum()
    if total_missing == 0:
        print("\n✓ No missing values found in the dataset")
    else:
        print(f"\n⚠ Found {total_missing} missing values")

    return missing_df


def normalize_features(df):
    """
    Normalize features using Min-Max scaling

    Args:
        df: pandas DataFrame with features

    Returns:
        DataFrame with normalized features and scaler object
    """
    print("\n" + "=" * 70)
    print("NORMALIZING FEATURES (MIN-MAX SCALING)")
    print("=" * 70)

    # Select only numeric feature columns
    feature_cols = [col for col in df.columns if col not in ['species', 'species_name']]

    # Create a copy for normalized data
    df_normalized = df.copy()

    # Apply Min-Max scaling
    scaler = MinMaxScaler()
    df_normalized[feature_cols] = scaler.fit_transform(df[feature_cols])

    print("✓ Features normalized to range [0, 1]")
    print("\nOriginal vs Normalized ranges:")
    for col in feature_cols:
        orig_min, orig_max = df[col].min(), df[col].max()
        norm_min, norm_max = df_normalized[col].min(), df_normalized[col].max()
        print(f"  {col:30s}: [{orig_min:6.2f}, {orig_max:6.2f}] → [{norm_min:.2f}, {norm_max:.2f}]")

    return df_normalized, scaler


def compute_statistics(df):
    """
    Compute and display summary statistics

    Args:
        df: pandas DataFrame
    """
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'species']

    print("\nDescriptive Statistics (Original Data):")
    print(df[numeric_cols].describe().to_string())

    # Statistics by species
    print("\n\nStatistics by Species:")
    for species in df['species_name'].unique():
        species_data = df[df['species_name'] == species][numeric_cols]
        print(f"\n{species.capitalize()}:")
        print(species_data.describe().loc[['mean', 'std']].to_string())


def create_visualizations(df, df_normalized):
    """
    Create exploratory visualizations

    Args:
        df: Original DataFrame
        df_normalized: Normalized DataFrame
    """
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    feature_cols = [col for col in df.columns if col not in ['species', 'species_name']]

    # 1. Pairplot
    print("  Creating pairplot...")
    plt.figure(figsize=(14, 10))
    pairplot = sns.pairplot(df, hue='species_name',
                           vars=feature_cols,
                           diag_kind='kde',
                           plot_kws={'alpha': 0.6},
                           palette='Set2')
    pairplot.fig.suptitle('Iris Dataset - Pairplot by Species', y=1.01, fontsize=16, fontweight='bold')
    plt.savefig('../images/pairplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: ../images/pairplot.png")

    # 2. Correlation Heatmap
    print("  Creating correlation heatmap...")
    plt.figure(figsize=(10, 8))
    correlation = df[feature_cols].corr()
    sns.heatmap(correlation, annot=True, fmt='.3f', cmap='coolwarm',
                square=True, linewidths=1, cbar_kws={'label': 'Correlation'})
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('../images/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: ../images/correlation_heatmap.png")

    # 3. Boxplots for outlier detection
    print("  Creating boxplots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for idx, col in enumerate(feature_cols):
        sns.boxplot(data=df, x='species_name', y=col, ax=axes[idx], palette='Set2')
        axes[idx].set_title(f'{col}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Species')
        axes[idx].set_ylabel('Value (cm)')

    plt.suptitle('Boxplots for Outlier Detection by Species',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('../images/boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: ../images/boxplots.png")

    # 4. Distribution plots
    print("  Creating distribution plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for idx, col in enumerate(feature_cols):
        for species in df['species_name'].unique():
            species_data = df[df['species_name'] == species][col]
            axes[idx].hist(species_data, alpha=0.5, bins=15, label=species)

        axes[idx].set_title(f'Distribution: {col}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Value (cm)')
        axes[idx].set_ylabel('Frequency')
        axes[idx].legend()

    plt.suptitle('Feature Distributions by Species',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('../images/distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: ../images/distributions.png")

    print("\n✓ All visualizations created successfully")


def detect_outliers(df):
    """
    Identify potential outliers using IQR method

    Args:
        df: pandas DataFrame
    """
    print("\n" + "=" * 70)
    print("OUTLIER DETECTION (IQR METHOD)")
    print("=" * 70)

    feature_cols = [col for col in df.columns if col not in ['species', 'species_name']]

    outliers_summary = []

    for col in feature_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

        outliers_summary.append({
            'Feature': col,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'Lower_Bound': lower_bound,
            'Upper_Bound': upper_bound,
            'Outlier_Count': len(outliers)
        })

    outliers_df = pd.DataFrame(outliers_summary)
    print(outliers_df.to_string(index=False))

    total_outliers = outliers_df['Outlier_Count'].sum()
    if total_outliers == 0:
        print("\n✓ No significant outliers detected")
    else:
        print(f"\n⚠ Found {total_outliers} potential outliers across features")


def split_train_test(df, test_size=0.2):
    """
    Split data into training and testing sets

    Args:
        df: pandas DataFrame
        test_size: proportion of data for testing (default 0.2 = 20%)

    Returns:
        X_train, X_test, y_train, y_test
    """
    print("\n" + "=" * 70)
    print("TRAIN/TEST SPLIT")
    print("=" * 70)

    # Features and target
    feature_cols = [col for col in df.columns if col not in ['species', 'species_name']]
    X = df[feature_cols]
    y = df['species']

    # Split the data (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=672662, stratify=y
    )

    print(f"✓ Data split completed")
    print(f"  - Training set: {len(X_train)} samples ({(1-test_size)*100:.0f}%)")
    print(f"  - Testing set:  {len(X_test)} samples ({test_size*100:.0f}%)")
    print(f"  - Features: {X_train.shape[1]}")
    print(f"\n  Class distribution in training set:")
    for class_label, count in y_train.value_counts().sort_index().items():
        species_name = ['setosa', 'versicolor', 'virginica'][class_label]
        print(f"    {species_name:12s}: {count} samples")

    return X_train, X_test, y_train, y_test


def main():
    """
    Main function to run preprocessing and exploration pipeline
    """
    print("\n" + "=" * 70)
    print("DSA 2040 - DATA PREPROCESSING AND EXPLORATION")
    print("Student: Gift Wanjiru Gachunga (672662)")
    print("Dataset: Iris (scikit-learn built-in)")
    print("=" * 70)

    # 1. Load data
    df = load_data()

    # 2. Check for missing values
    check_missing_values(df)

    # 3. Compute statistics
    compute_statistics(df)

    # 4. Detect outliers
    detect_outliers(df)

    # 5. Normalize features
    df_normalized, scaler = normalize_features(df)

    # 6. Create visualizations
    create_visualizations(df, df_normalized)

    # 7. Train/test split
    X_train, X_test, y_train, y_test = split_train_test(df_normalized)

    # Save preprocessed data
    print("\n" + "=" * 70)
    print("SAVING PREPROCESSED DATA")
    print("=" * 70)

    df_normalized.to_csv('../datasets/iris_normalized.csv', index=False)
    print("✓ Saved normalized data: ../datasets/iris_normalized.csv")

    # Save train/test splits
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    train_data.to_csv('../datasets/iris_train.csv', index=False)
    test_data.to_csv('../datasets/iris_test.csv', index=False)
    print("✓ Saved training data: ../datasets/iris_train.csv")
    print("✓ Saved testing data: ../datasets/iris_test.csv")

    print("\n" + "=" * 70)
    print("PREPROCESSING AND EXPLORATION COMPLETE!")
    print("=" * 70)
    print("\nGenerated Files:")
    print("  - ../datasets/iris_normalized.csv (normalized dataset)")
    print("  - ../datasets/iris_train.csv (training set)")
    print("  - ../datasets/iris_test.csv (testing set)")
    print("  - ../images/pairplot.png")
    print("  - ../images/correlation_heatmap.png")
    print("  - ../images/boxplots.png")
    print("  - ../images/distributions.png")

    return df, df_normalized, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df, df_normalized, X_train, X_test, y_train, y_test = main()
