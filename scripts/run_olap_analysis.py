"""
DSA 2040 Practical Exam - Section 1, Task 3
OLAP Queries and Visualization
Student: Gift Wanjiru Gachunga (672662)

This script executes OLAP queries and creates visualizations
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Database path
DB_PATH = '../datasets/retail_dw.db'


def execute_query(query, description):
    """Execute a SQL query and display results"""
    print(f"\n{'=' * 70}")
    print(f"{description}")
    print('=' * 70)

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(query, conn)
    conn.close()

    print(df.to_string(index=False))
    print(f"\nTotal rows: {len(df)}")

    return df


def main():
    """Run all OLAP queries and create visualizations"""

    print("DSA 2040 - OLAP Analysis")
    print("Student: Gift Wanjiru Gachunga (672662)")
    print("=" * 70)

    # Query 1: ROLL-UP - Total sales by country and quarter
    rollup_query = """
    SELECT
        c.country,
        t.quarter,
        t.year,
        COUNT(DISTINCT s.rowid) as total_transactions,
        SUM(s.quantity) as total_units_sold,
        ROUND(SUM(s.total_sales), 2) as total_sales_amount,
        ROUND(AVG(s.total_sales), 2) as avg_transaction_value
    FROM SalesFact s
    JOIN CustomerDim c ON s.customer_key = c.rowid
    JOIN TimeDim t ON s.time_key = t.rowid
    GROUP BY c.country, t.quarter, t.year
    ORDER BY t.year DESC, t.quarter DESC, total_sales_amount DESC
    LIMIT 15;
    """

    df_rollup = execute_query(rollup_query, "QUERY 1: ROLL-UP - Sales by Country and Quarter")

    # Query 2: DRILL-DOWN - Sales for UK by month
    drilldown_query = """
    SELECT
        c.country,
        t.year,
        t.month,
        t.month_name,
        COUNT(DISTINCT s.rowid) as total_transactions,
        SUM(s.quantity) as total_units_sold,
        ROUND(SUM(s.total_sales), 2) as total_sales_amount
    FROM SalesFact s
    JOIN CustomerDim c ON s.customer_key = c.rowid
    JOIN TimeDim t ON s.time_key = t.rowid
    WHERE c.country = 'UK'
    GROUP BY c.country, t.year, t.month, t.month_name
    ORDER BY t.year DESC, t.month DESC;
    """

    df_drilldown = execute_query(drilldown_query, "QUERY 2: DRILL-DOWN - UK Sales by Month")

    # Query 3: SLICE - Electronics category sales
    slice_query = """
    SELECT
        p.product_category,
        p.product_subcategory,
        COUNT(DISTINCT s.rowid) as total_transactions,
        SUM(s.quantity) as total_units_sold,
        ROUND(SUM(s.total_sales), 2) as total_sales_amount,
        ROUND(SUM(s.profit), 2) as total_profit
    FROM SalesFact s
    JOIN ProductDim p ON s.product_key = p.rowid
    WHERE p.product_category = 'Electronics'
    GROUP BY p.product_category, p.product_subcategory
    ORDER BY total_sales_amount DESC;
    """

    df_slice = execute_query(slice_query, "QUERY 3: SLICE - Electronics Category Sales")

    # Create visualizations
    print(f"\n{'=' * 70}")
    print("Creating Visualizations...")
    print('=' * 70)

    # Visualization 1: Sales by Country (from roll-up query)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top countries by sales
    country_sales = df_rollup.groupby('country')['total_sales_amount'].sum().sort_values(ascending=False).head(10)
    axes[0, 0].barh(country_sales.index, country_sales.values, color='steelblue')
    axes[0, 0].set_xlabel('Total Sales Amount')
    axes[0, 0].set_title('Top 10 Countries by Total Sales', fontsize=14, fontweight='bold')
    axes[0, 0].invert_yaxis()

    # Sales by Quarter
    quarter_sales = df_rollup.groupby('quarter')['total_sales_amount'].sum()
    axes[0, 1].bar(quarter_sales.index, quarter_sales.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
    axes[0, 1].set_xlabel('Quarter')
    axes[0, 1].set_ylabel('Total Sales Amount')
    axes[0, 1].set_title('Sales by Quarter', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks([1, 2, 3, 4])

    # UK Sales Trend by Month
    if len(df_drilldown) > 0:
        df_drilldown['period'] = df_drilldown['year'].astype(str) + '-' + df_drilldown['month'].astype(str).str.zfill(2)
        axes[1, 0].plot(range(len(df_drilldown)), df_drilldown['total_sales_amount'],
                        marker='o', linewidth=2, markersize=6, color='#FF6B6B')
        axes[1, 0].set_xlabel('Time Period')
        axes[1, 0].set_ylabel('Total Sales Amount')
        axes[1, 0].set_title('UK Sales Trend by Month', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

    # Electronics subcategory sales
    if len(df_slice) > 0:
        axes[1, 1].bar(range(len(df_slice)), df_slice['total_sales_amount'], color='#4ECDC4')
        axes[1, 1].set_xticks(range(len(df_slice)))
        axes[1, 1].set_xticklabels(df_slice['product_subcategory'], rotation=45, ha='right')
        axes[1, 1].set_ylabel('Total Sales Amount')
        axes[1, 1].set_title('Electronics Sales by Subcategory', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('../images/sales_visualization.png', dpi=300, bbox_inches='tight')
    print("✓ Saved visualization: ../images/sales_visualization.png")

    # Additional simple bar chart for country sales
    plt.figure(figsize=(12, 6))
    top_10_countries = df_rollup.groupby('country')['total_sales_amount'].sum().sort_values(ascending=False).head(10)
    plt.bar(range(len(top_10_countries)), top_10_countries.values, color='steelblue', edgecolor='navy', linewidth=1.5)
    plt.xlabel('Country', fontsize=12, fontweight='bold')
    plt.ylabel('Total Sales Amount', fontsize=12, fontweight='bold')
    plt.title('Top 10 Countries by Sales Revenue', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(range(len(top_10_countries)), top_10_countries.index, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('../images/sales_by_country.png', dpi=300, bbox_inches='tight')
    print("✓ Saved visualization: ../images/sales_by_country.png")

    print("\n" + "=" * 70)
    print("OLAP Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
