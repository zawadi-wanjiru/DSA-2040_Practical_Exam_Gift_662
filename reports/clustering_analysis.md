# K-Means Clustering Analysis

**Student:** Gift Wanjiru Gachunga (672662)
**Course:** DSA 2040 - Data Warehousing and Data Mining
**Task:** Section 2, Task 2 - Clustering (15 Marks)

---

## Cluster Quality Assessment

The K-Means clustering with k=3 achieved an **Adjusted Rand Index (ARI) of 0.7163**, indicating substantial agreement between predicted clusters and true species labels. The **Silhouette Score of 0.5048** demonstrates reasonable cluster separation, though not perfect. The elbow curve analysis clearly shows k=3 as optimal, where inertia drops from 12.13 (k=2) to 6.98 (k=3), with diminishing returns beyond this point.

Cluster 1 (Setosa) achieved **perfect separation** with 100% purity, containing all 50 Setosa samples with zero misclassifications. This reflects Setosa's distinct morphological characteristics, particularly its shorter petals. However, Clusters 0 and 2 (Versicolor and Virginica) show **overlap**: Cluster 0 contains 77% Versicolor but 23% Virginica (14 samples), while Cluster 2 contains 92% Virginica but 7.7% Versicolor (3 samples). This 17-sample confusion between Versicolor and Virginica stems from their similar petal and sepal dimensions, as evident in the pairplot and PCA visualizations.

The experimentation with k=2 and k=4 validates k=3 as optimal. While k=2 achieves higher silhouette (0.63), it merges Versicolor and Virginica inappropriately (ARI=0.57). Conversely, k=4 unnecessarily splits natural groupings (ARI=0.62). The k=3 configuration balances cluster cohesion, separation, and biological meaningfulness.

## Real-World Applications

**Customer Segmentation:**
K-Means clustering enables businesses to partition customers into distinct groups based on purchasing behavior, demographics, and engagement patterns. For example, an e-commerce platform could identify "budget-conscious buyers," "premium shoppers," and "occasional purchasers," allowing tailored marketing strategies, personalized recommendations, and targeted promotions that maximize conversion rates and customer lifetime value.

**Image Compression:**
In computer vision, K-Means reduces image file sizes by clustering similar pixel colors and replacing each cluster with its centroid color. A photograph with millions of unique colors can be compressed to k representative colors (e.g., k=16 or k=64), significantly reducing storage requirements while maintaining visual quality—critical for web optimization and mobile applications where bandwidth is limited.

**Anomaly Detection:**
Manufacturing quality control systems use clustering to identify defective products. By clustering normal production data, items falling far from cluster centroids or forming isolated small clusters are flagged as anomalies for inspection. This approach detects sensor malfunctions, equipment drift, and product defects earlier than traditional threshold-based methods, reducing waste and improving product consistency.

---

## Impact of Synthetic Data

While the Iris dataset is real botanical data (not synthetic), its limited size (150 samples) and controlled collection conditions may not fully represent the variability in natural populations. Real-world species identification faces challenges like measurement errors, environmental variations, hybrid specimens, and missing features that could reduce clustering accuracy. Production systems would require larger, more diverse datasets and potentially more sophisticated clustering algorithms (e.g., DBSCAN for non-spherical clusters, hierarchical clustering for taxonomy trees) to handle biological complexity. Nevertheless, the methodology demonstrated—elbow curve analysis, silhouette scoring, and ARI evaluation—remains directly applicable to operational clustering tasks across domains.

---

**Analysis Date:** December 12, 2025
**Clustering Algorithm:** K-Means (scikit-learn)
**Optimal k:** 3 clusters
**Key Metrics:** ARI=0.7163, Silhouette=0.5048, Inertia=6.9822
