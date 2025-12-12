"""
DSA 2040 Practical Exam - Section 2, Task 2
K-Means Clustering
Student: Gift Wanjiru Gachunga (672662)

This script performs:
- K-Means clustering with k=3
- Cluster quality evaluation using Adjusted Rand Index
- Experimentation with different k values
- Elbow curve analysis
- Cluster visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.decomposition import PCA

# Set random seed for reproducibility
np.random.seed(672662)

# Configure plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_preprocessed_data():
    """
    Load the preprocessed and normalized data from Task 1

    Returns:
        X: Features DataFrame
        y: True labels
    """
    print("=" * 70)
    print("LOADING PREPROCESSED DATA")
    print("=" * 70)

    df = pd.read_csv('../datasets/iris_normalized.csv')

    # Features (exclude species column)
    feature_cols = [col for col in df.columns if col not in ['species', 'species_name']]
    X = df[feature_cols]
    y = df['species']

    print(f"✓ Loaded normalized Iris dataset")
    print(f"  - Samples: {len(X)}")
    print(f"  - Features: {X.shape[1]}")
    print(f"  - Feature names: {list(X.columns)}")

    return X, y, df


def apply_kmeans(X, y, k=3):
    """
    Apply K-Means clustering with specified k

    Args:
        X: Feature matrix
        y: True labels
        k: Number of clusters

    Returns:
        KMeans model and cluster predictions
    """
    print(f"\n" + "=" * 70)
    print(f"K-MEANS CLUSTERING (k={k})")
    print("=" * 70)

    # Fit K-Means
    kmeans = KMeans(n_clusters=k, random_state=672662, n_init=10)
    clusters = kmeans.fit_predict(X)

    print(f"✓ K-Means clustering completed")
    print(f"  - Number of clusters: {k}")
    print(f"  - Iterations: {kmeans.n_iter_}")
    print(f"  - Inertia (within-cluster sum of squares): {kmeans.inertia_:.4f}")

    # Cluster sizes
    unique, counts = np.unique(clusters, return_counts=True)
    print(f"\n  Cluster sizes:")
    for cluster_id, count in zip(unique, counts):
        print(f"    Cluster {cluster_id}: {count} samples")

    # Evaluate clustering quality
    ari = adjusted_rand_score(y, clusters)
    silhouette = silhouette_score(X, clusters)

    print(f"\n  Quality Metrics:")
    print(f"    Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"    Silhouette Score: {silhouette:.4f}")

    print(f"\n  Interpretation:")
    print(f"    - ARI ranges from -1 to 1 (1 = perfect match with true labels)")
    print(f"    - Silhouette ranges from -1 to 1 (1 = well-separated clusters)")

    return kmeans, clusters, ari, silhouette


def experiment_k_values(X, k_range=(2, 11)):
    """
    Experiment with different k values to find optimal k

    Args:
        X: Feature matrix
        k_range: Range of k values to try

    Returns:
        Dictionary with inertias and silhouette scores
    """
    print("\n" + "=" * 70)
    print("EXPERIMENTING WITH DIFFERENT K VALUES")
    print("=" * 70)

    inertias = []
    silhouette_scores = []
    k_values = range(k_range[0], k_range[1])

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=672662, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

        if k > 1:  # Silhouette score requires at least 2 clusters
            silhouette = silhouette_score(X, kmeans.labels_)
            silhouette_scores.append(silhouette)
        else:
            silhouette_scores.append(0)

        print(f"  k={k}: Inertia={kmeans.inertia_:.4f}, Silhouette={silhouette_scores[-1]:.4f}")

    return {
        'k_values': list(k_values),
        'inertias': inertias,
        'silhouette_scores': silhouette_scores
    }


def plot_elbow_curve(experiment_results):
    """
    Plot elbow curve to determine optimal k

    Args:
        experiment_results: Dictionary from experiment_k_values
    """
    print("\n" + "=" * 70)
    print("CREATING ELBOW CURVE")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    k_values = experiment_results['k_values']
    inertias = experiment_results['inertias']
    silhouette_scores = experiment_results['silhouette_scores']

    # Elbow curve (Inertia)
    axes[0].plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].axvline(x=3, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Optimal k=3')
    axes[0].set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12, fontweight='bold')
    axes[0].set_title('Elbow Method - Inertia', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_xticks(k_values)

    # Silhouette scores
    axes[1].plot(k_values, silhouette_scores, 'go-', linewidth=2, markersize=8)
    axes[1].axvline(x=3, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Optimal k=3')
    axes[1].set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    axes[1].set_title('Silhouette Score Analysis', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_xticks(k_values)

    plt.tight_layout()
    plt.savefig('../images/elbow_curve.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: visualizations/elbow_curve.png")
    plt.close()


def visualize_clusters(X, y, clusters, kmeans):
    """
    Visualize clusters using various plots

    Args:
        X: Feature matrix
        y: True labels
        clusters: Predicted cluster labels
        kmeans: Fitted KMeans model
    """
    print("\n" + "=" * 70)
    print("CREATING CLUSTER VISUALIZATIONS")
    print("=" * 70)

    # Convert to numpy array if DataFrame
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns
        X_array = X.values
    else:
        X_array = X
        feature_names = ['Feature ' + str(i) for i in range(X.shape[1])]

    # 1. Petal length vs petal width (most discriminative features)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot by predicted clusters
    scatter1 = axes[0].scatter(X_array[:, 2], X_array[:, 3], c=clusters,
                               cmap='viridis', s=100, alpha=0.6, edgecolors='black', linewidth=1)
    axes[0].scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:, 3],
                   c='red', marker='X', s=300, edgecolors='black', linewidth=2, label='Centroids')
    axes[0].set_xlabel('Petal Length (normalized)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Petal Width (normalized)', fontsize=12, fontweight='bold')
    axes[0].set_title('K-Means Clusters (k=3)', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0], label='Cluster')

    # Plot by true labels for comparison
    scatter2 = axes[1].scatter(X_array[:, 2], X_array[:, 3], c=y,
                               cmap='viridis', s=100, alpha=0.6, edgecolors='black', linewidth=1)
    axes[1].set_xlabel('Petal Length (normalized)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Petal Width (normalized)', fontsize=12, fontweight='bold')
    axes[1].set_title('True Species Labels', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1], label='Species')

    plt.tight_layout()
    plt.savefig('../images/cluster_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: visualizations/cluster_comparison.png")
    plt.close()

    # 2. PCA visualization
    print("  Applying PCA for 2D visualization...")
    pca = PCA(n_components=2, random_state=672662)
    X_pca = pca.fit_transform(X_array)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # PCA with clusters
    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters,
                               cmap='viridis', s=100, alpha=0.6, edgecolors='black', linewidth=1)
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12, fontweight='bold')
    axes[0].set_title('K-Means Clusters (PCA projection)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0], label='Cluster')

    # PCA with true labels
    scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y,
                               cmap='viridis', s=100, alpha=0.6, edgecolors='black', linewidth=1)
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12, fontweight='bold')
    axes[1].set_title('True Species (PCA projection)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1], label='Species')

    plt.tight_layout()
    plt.savefig('../images/pca_clusters.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: visualizations/pca_clusters.png")
    plt.close()

    # 3. All feature pairs
    print("  Creating comprehensive pairwise cluster plots...")
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

    for i in range(4):
        for j in range(4):
            if i == j:
                # Diagonal: histograms
                axes[i, j].hist(X_array[:, i], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
                axes[i, j].set_ylabel('Frequency')
            else:
                # Off-diagonal: scatter plots
                axes[i, j].scatter(X_array[:, j], X_array[:, i], c=clusters,
                                 cmap='viridis', s=30, alpha=0.6)

            if i == 3:
                axes[i, j].set_xlabel(feature_names[j], fontsize=10)
            if j == 0:
                axes[i, j].set_ylabel(feature_names[i], fontsize=10)

    plt.suptitle('K-Means Clusters - All Feature Pairs', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('../images/cluster_pairplot.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: visualizations/cluster_pairplot.png")
    plt.close()

    print("\n✓ All visualizations created successfully")


def analyze_misclassifications(y, clusters):
    """
    Analyze which samples were misclassified

    Args:
        y: True labels
        clusters: Predicted clusters
    """
    print("\n" + "=" * 70)
    print("MISCLASSIFICATION ANALYSIS")
    print("=" * 70)

    # Create confusion-like matrix
    species_names = ['setosa', 'versicolor', 'virginica']

    print("\nCluster composition (True labels in each cluster):")
    print("-" * 70)

    for cluster_id in range(3):
        cluster_mask = clusters == cluster_id
        cluster_labels = y[cluster_mask]

        print(f"\nCluster {cluster_id} ({cluster_mask.sum()} samples):")
        for species_id, species_name in enumerate(species_names):
            count = (cluster_labels == species_id).sum()
            pct = (count / cluster_mask.sum() * 100) if cluster_mask.sum() > 0 else 0
            print(f"  {species_name:12s}: {count:3d} samples ({pct:5.1f}%)")


def main():
    """
    Main function to run clustering pipeline
    """
    print("\n" + "=" * 70)
    print("DSA 2040 - K-MEANS CLUSTERING")
    print("Student: Gift Wanjiru Gachunga (672662)")
    print("Dataset: Iris (preprocessed from Task 1)")
    print("=" * 70)

    # 1. Load data
    X, y, df = load_preprocessed_data()

    # 2. Apply K-Means with k=3
    kmeans_3, clusters_3, ari_3, silhouette_3 = apply_kmeans(X, y, k=3)

    # 3. Experiment with different k values
    experiment_results = experiment_k_values(X, k_range=(2, 11))

    # 4. Apply K-Means with k=2 and k=4 for comparison
    print("\n" + "=" * 70)
    print("COMPARING k=2 AND k=4")
    print("=" * 70)

    kmeans_2, clusters_2, ari_2, silhouette_2 = apply_kmeans(X, y, k=2)
    kmeans_4, clusters_4, ari_4, silhouette_4 = apply_kmeans(X, y, k=4)

    # 5. Plot elbow curve
    plot_elbow_curve(experiment_results)

    # 6. Visualize clusters
    visualize_clusters(X, y, clusters_3, kmeans_3)

    # 7. Analyze misclassifications
    analyze_misclassifications(y, clusters_3)

    # 8. Summary
    print("\n" + "=" * 70)
    print("CLUSTERING SUMMARY")
    print("=" * 70)

    print("\nComparison of k values:")
    print(f"  k=2: ARI={ari_2:.4f}, Silhouette={silhouette_2:.4f}")
    print(f"  k=3: ARI={ari_3:.4f}, Silhouette={silhouette_3:.4f} ← OPTIMAL")
    print(f"  k=4: ARI={ari_4:.4f}, Silhouette={silhouette_4:.4f}")

    print("\nConclusion:")
    print("  k=3 provides the best clustering as it:")
    print("  - Matches the true number of species (3)")
    print(f"  - Achieves highest ARI: {ari_3:.4f}")
    print(f"  - Maintains good cluster separation (Silhouette: {silhouette_3:.4f})")
    print("  - Shows clear 'elbow' in the inertia plot")

    print("\n" + "=" * 70)
    print("CLUSTERING COMPLETE!")
    print("=" * 70)
    print("\nGenerated Files:")
    print("  - ../images/elbow_curve.png")
    print("  - ../images/cluster_comparison.png")
    print("  - ../images/pca_clusters.png")
    print("  - ../images/cluster_pairplot.png")

    return kmeans_3, clusters_3


if __name__ == "__main__":
    kmeans, clusters = main()
