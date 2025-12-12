"""
DSA 2040 Practical Exam - Section 2, Task 3
Classification and Association Rule Mining
Student: Gift Wanjiru Gachunga (672662)

Part A: Classification using Decision Tree and KNN
Part B: Association Rule Mining using Apriori algorithm
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, classification_report, confusion_matrix)
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import random

# Set random seed for reproducibility
np.random.seed(672662)
random.seed(672662)

# Configure plotting
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


# ============================================================================
# PART A: CLASSIFICATION
# ============================================================================

def load_train_test_data():
    """
    Load preprocessed train and test data from Task 1

    Returns:
        X_train, X_test, y_train, y_test
    """
    print("=" * 70)
    print("PART A: CLASSIFICATION")
    print("=" * 70)
    print("\n" + "=" * 70)
    print("LOADING TRAIN/TEST DATA")
    print("=" * 70)

    train_df = pd.read_csv('datasets/iris_train.csv')
    test_df = pd.read_csv('datasets/iris_test.csv')

    # Features and labels
    feature_cols = [col for col in train_df.columns if col not in ['species', 'species_name']]

    X_train = train_df[feature_cols]
    y_train = train_df['species']
    X_test = test_df[feature_cols]
    y_test = test_df['species']

    print(f"✓ Loaded preprocessed data")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Testing samples: {len(X_test)}")
    print(f"  - Features: {list(feature_cols)}")

    return X_train, X_test, y_train, y_test


def train_decision_tree(X_train, X_test, y_train, y_test):
    """
    Train Decision Tree classifier and evaluate

    Args:
        X_train, X_test: Feature matrices
        y_train, y_test: Labels

    Returns:
        Trained model and predictions
    """
    print("\n" + "=" * 70)
    print("DECISION TREE CLASSIFIER")
    print("=" * 70)

    # Train Decision Tree
    dt_model = DecisionTreeClassifier(random_state=672662, max_depth=4)
    dt_model.fit(X_train, y_train)

    # Predictions
    y_pred = dt_model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("✓ Decision Tree trained")
    print(f"  - Max depth: {dt_model.max_depth}")
    print(f"  - Number of leaves: {dt_model.get_n_leaves()}")
    print(f"  - Tree depth: {dt_model.get_depth()}")

    print("\n  Performance Metrics:")
    print(f"    Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1-Score:  {f1:.4f}")

    print("\n  Classification Report:")
    species_names = ['setosa', 'versicolor', 'virginica']
    print(classification_report(y_test, y_pred, target_names=species_names))

    return dt_model, y_pred, {'accuracy': accuracy, 'precision': precision,
                               'recall': recall, 'f1': f1}


def visualize_decision_tree(dt_model, feature_names):
    """
    Visualize the trained decision tree

    Args:
        dt_model: Trained DecisionTreeClassifier
        feature_names: List of feature names
    """
    print("\n" + "=" * 70)
    print("VISUALIZING DECISION TREE")
    print("=" * 70)

    species_names = ['setosa', 'versicolor', 'virginica']

    plt.figure(figsize=(20, 12))
    plot_tree(dt_model,
             feature_names=feature_names,
             class_names=species_names,
             filled=True,
             rounded=True,
             fontsize=10)
    plt.title('Decision Tree Classifier - Iris Species', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('images/decision_tree.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: images/decision_tree.png")
    plt.close()


def train_knn(X_train, X_test, y_train, y_test, k=5):
    """
    Train K-Nearest Neighbors classifier

    Args:
        X_train, X_test: Feature matrices
        y_train, y_test: Labels
        k: Number of neighbors

    Returns:
        Trained model, predictions, and metrics
    """
    print("\n" + "=" * 70)
    print(f"K-NEAREST NEIGHBORS CLASSIFIER (k={k})")
    print("=" * 70)

    # Train KNN
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)

    # Predictions
    y_pred = knn_model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"✓ KNN trained with k={k}")
    print("\n  Performance Metrics:")
    print(f"    Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1-Score:  {f1:.4f}")

    print("\n  Classification Report:")
    species_names = ['setosa', 'versicolor', 'virginica']
    print(classification_report(y_test, y_pred, target_names=species_names))

    return knn_model, y_pred, {'accuracy': accuracy, 'precision': precision,
                                'recall': recall, 'f1': f1}


def compare_classifiers(dt_metrics, knn_metrics):
    """
    Compare Decision Tree and KNN performance

    Args:
        dt_metrics: Dictionary of Decision Tree metrics
        knn_metrics: Dictionary of KNN metrics
    """
    print("\n" + "=" * 70)
    print("CLASSIFIER COMPARISON")
    print("=" * 70)

    comparison_df = pd.DataFrame({
        'Decision Tree': [dt_metrics['accuracy'], dt_metrics['precision'],
                         dt_metrics['recall'], dt_metrics['f1']],
        'KNN (k=5)': [knn_metrics['accuracy'], knn_metrics['precision'],
                     knn_metrics['recall'], knn_metrics['f1']]
    }, index=['Accuracy', 'Precision', 'Recall', 'F1-Score'])

    print("\n" + comparison_df.to_string())

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    comparison_df.plot(kind='bar', ax=ax, color=['steelblue', 'coral'], width=0.7)
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Classifier Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    ax.legend(title='Classifier')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)

    plt.tight_layout()
    plt.savefig('images/classifier_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: images/classifier_comparison.png")
    plt.close()

    # Determine better classifier
    dt_avg = np.mean(list(dt_metrics.values()))
    knn_avg = np.mean(list(knn_metrics.values()))

    print("\n  Average Performance:")
    print(f"    Decision Tree: {dt_avg:.4f}")
    print(f"    KNN (k=5):     {knn_avg:.4f}")

    if dt_avg > knn_avg:
        print("\n  ✓ Decision Tree performs better overall")
        print("    Reason: Higher average across all metrics, particularly in")
        print("    precision and recall, indicating better class discrimination.")
    else:
        print("\n  ✓ KNN performs better overall")
        print("    Reason: Higher average across all metrics, benefiting from")
        print("    the normalized feature space and local decision boundaries.")


# ============================================================================
# PART B: ASSOCIATION RULE MINING
# ============================================================================

def generate_transaction_data(n_transactions=50):
    """
    Generate synthetic transactional data for market basket analysis

    Args:
        n_transactions: Number of transactions to generate

    Returns:
        List of transactions
    """
    print("\n\n" + "=" * 70)
    print("PART B: ASSOCIATION RULE MINING")
    print("=" * 70)
    print("\n" + "=" * 70)
    print("GENERATING SYNTHETIC TRANSACTION DATA")
    print("=" * 70)

    # Define items (grocery store items)
    items = [
        'milk', 'bread', 'eggs', 'butter', 'cheese',
        'yogurt', 'coffee', 'tea', 'sugar', 'flour',
        'rice', 'pasta', 'chicken', 'beef', 'fish',
        'apple', 'banana', 'orange', 'tomato', 'onion'
    ]

    # Define common item associations for realistic patterns
    frequent_pairs = [
        ('milk', 'bread'),
        ('milk', 'eggs'),
        ('bread', 'butter'),
        ('coffee', 'sugar'),
        ('tea', 'sugar'),
        ('pasta', 'tomato'),
        ('chicken', 'rice'),
        ('eggs', 'bacon'),
        ('cheese', 'bread'),
        ('apple', 'banana')
    ]

    transactions = []

    for i in range(n_transactions):
        # Random transaction size (3-8 items)
        trans_size = random.randint(3, 8)

        # Start with some random items
        transaction = set(random.sample(items, min(trans_size//2, len(items))))

        # Add items from frequent pairs with high probability
        if random.random() > 0.4:  # 60% chance to include frequent pair
            pair = random.choice(frequent_pairs)
            transaction.add(pair[0])
            transaction.add(pair[1])

        # Fill up to desired size
        while len(transaction) < trans_size:
            transaction.add(random.choice(items))

        transactions.append(list(transaction))

    print(f"✓ Generated {n_transactions} transactions")
    print(f"  - Item pool: {len(items)} unique items")
    print(f"  - Transaction sizes: 3-8 items")
    print(f"  - Patterns: {len(frequent_pairs)} frequent item pairs")

    # Display sample transactions
    print("\n  Sample Transactions:")
    for i, trans in enumerate(transactions[:5], 1):
        print(f"    {i}. {trans}")

    # Save to CSV
    trans_df = pd.DataFrame({
        'TransactionID': range(1, len(transactions)+1),
        'Items': [', '.join(sorted(t)) for t in transactions]
    })
    trans_df.to_csv('datasets/transactions.csv', index=False)
    print("\n✓ Saved: datasets/transactions.csv")

    return transactions


def apply_apriori(transactions, min_support=0.2, min_confidence=0.5):
    """
    Apply Apriori algorithm to find association rules

    Args:
        transactions: List of transactions
        min_support: Minimum support threshold
        min_confidence: Minimum confidence threshold

    Returns:
        DataFrame of association rules
    """
    print("\n" + "=" * 70)
    print("APPLYING APRIORI ALGORITHM")
    print("=" * 70)

    # Encode transactions
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_array, columns=te.columns_)

    print(f"  Parameters:")
    print(f"    Min Support:    {min_support}")
    print(f"    Min Confidence: {min_confidence}")

    # Find frequent itemsets
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    print(f"\n✓ Found {len(frequent_itemsets)} frequent itemsets")

    # Generate association rules
    if len(frequent_itemsets) == 0:
        print("⚠ No frequent itemsets found. Try lowering min_support.")
        return None

    rules = association_rules(frequent_itemsets, metric="confidence",
                              min_threshold=min_confidence, num_itemsets=len(frequent_itemsets))

    if len(rules) == 0:
        print("⚠ No rules found. Try lowering thresholds.")
        return None

    # Add lift metric
    print(f"✓ Generated {len(rules)} association rules")

    # Sort by lift
    rules = rules.sort_values('lift', ascending=False)

    return rules


def display_top_rules(rules, top_n=5):
    """
    Display top association rules

    Args:
        rules: DataFrame of association rules
        top_n: Number of top rules to display
    """
    print("\n" + "=" * 70)
    print(f"TOP {top_n} ASSOCIATION RULES (sorted by lift)")
    print("=" * 70)

    top_rules = rules.head(top_n)

    for idx, row in top_rules.iterrows():
        antecedent = ', '.join(list(row['antecedents']))
        consequent = ', '.join(list(row['consequents']))

        print(f"\nRule {idx+1}:")
        print(f"  {antecedent} → {consequent}")
        print(f"  Support:    {row['support']:.4f} ({row['support']*100:.1f}%)")
        print(f"  Confidence: {row['confidence']:.4f} ({row['confidence']*100:.1f}%)")
        print(f"  Lift:       {row['lift']:.4f}")

    # Create visualization
    plt.figure(figsize=(12, 8))

    # Prepare data for visualization
    rules_viz = top_rules.copy()
    rules_viz['rule'] = rules_viz.apply(
        lambda x: f"{', '.join(list(x['antecedents']))} → {', '.join(list(x['consequents']))}",
        axis=1
    )

    # Create scatter plot
    scatter = plt.scatter(rules_viz['support'], rules_viz['confidence'],
                         s=rules_viz['lift']*100, c=rules_viz['lift'],
                         cmap='viridis', alpha=0.6, edgecolors='black', linewidth=1.5)

    # Add labels for top rules
    for idx, row in rules_viz.iterrows():
        plt.annotate(f"Rule {idx+1}",
                    xy=(row['support'], row['confidence']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold')

    plt.xlabel('Support', fontsize=12, fontweight='bold')
    plt.ylabel('Confidence', fontsize=12, fontweight='bold')
    plt.title('Association Rules - Support vs Confidence\n(Size = Lift)',
             fontsize=14, fontweight='bold', pad=20)
    plt.colorbar(scatter, label='Lift')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/association_rules.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: images/association_rules.png")
    plt.close()

    return top_rules


def analyze_rule(rule):
    """
    Analyze a specific association rule

    Args:
        rule: Series containing rule information
    """
    print("\n" + "=" * 70)
    print("RULE ANALYSIS")
    print("=" * 70)

    antecedent = ', '.join(list(rule['antecedents']))
    consequent = ', '.join(list(rule['consequents']))

    print(f"\nAnalyzing Rule: {antecedent} → {consequent}")
    print("\n" + "-" * 70)

    print("\nMetrics Interpretation:")
    print(f"  Support ({rule['support']:.4f}):")
    print(f"    This rule appears in {rule['support']*100:.1f}% of all transactions.")
    print(f"    Indicates the rule's frequency in the dataset.")

    print(f"\n  Confidence ({rule['confidence']:.4f}):")
    print(f"    When customers buy {antecedent},")
    print(f"    they also buy {consequent} {rule['confidence']*100:.1f}% of the time.")
    print(f"    Measures the rule's reliability.")

    print(f"\n  Lift ({rule['lift']:.4f}):")
    if rule['lift'] > 1:
        print(f"    Items are {rule['lift']:.2f}x more likely to be bought together")
        print(f"    than if they were independent. Strong positive association!")
    elif rule['lift'] < 1:
        print(f"    Items are less likely to be bought together than independently.")
    else:
        print(f"    Items are independent (no association).")

    print("\n" + "-" * 70)
    print("\nBusiness Implications:")
    print(f"  1. Product Placement:")
    print(f"     Place {consequent} near {antecedent} to increase")
    print(f"     cross-selling opportunities.")

    print(f"\n  2. Recommendation Systems:")
    print(f"     When a customer adds {antecedent} to cart,")
    print(f"     suggest {consequent} as a related product.")

    print(f"\n  3. Promotional Strategies:")
    print(f"     Bundle {antecedent} and {consequent} together")
    print(f"     for special offers to increase basket size.")

    print(f"\n  4. Inventory Management:")
    print(f"     Ensure adequate stock of {consequent} when")
    print(f"     {antecedent} is in high demand.")


def main():
    """
    Main function to run classification and association rule mining
    """
    print("\n" + "=" * 70)
    print("DSA 2040 - CLASSIFICATION & ASSOCIATION RULE MINING")
    print("Student: Gift Wanjiru Gachunga (672662)")
    print("=" * 70)

    # ========== PART A: CLASSIFICATION ==========

    # 1. Load data
    X_train, X_test, y_train, y_test = load_train_test_data()

    # 2. Train Decision Tree
    dt_model, dt_pred, dt_metrics = train_decision_tree(X_train, X_test, y_train, y_test)

    # 3. Visualize Decision Tree
    visualize_decision_tree(dt_model, X_train.columns)

    # 4. Train KNN
    knn_model, knn_pred, knn_metrics = train_knn(X_train, X_test, y_train, y_test, k=5)

    # 5. Compare classifiers
    compare_classifiers(dt_metrics, knn_metrics)

    # ========== PART B: ASSOCIATION RULE MINING ==========

    # 6. Generate transaction data
    transactions = generate_transaction_data(n_transactions=50)

    # 7. Apply Apriori
    rules = apply_apriori(transactions, min_support=0.2, min_confidence=0.5)

    if rules is not None and len(rules) > 0:
        # 8. Display top rules
        top_rules = display_top_rules(rules, top_n=5)

        # 9. Analyze one rule
        analyze_rule(top_rules.iloc[0])
    else:
        print("\n⚠ Not enough rules found for analysis.")

    # Summary
    print("\n" + "=" * 70)
    print("TASK COMPLETE!")
    print("=" * 70)
    print("\nGenerated Files:")
    print("  Part A - Classification:")
    print("    - images/decision_tree.png")
    print("    - images/classifier_comparison.png")
    print("  Part B - Association Rules:")
    print("    - datasets/transactions.csv")
    print("    - images/association_rules.png")


if __name__ == "__main__":
    main()
