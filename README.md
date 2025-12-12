# DSA 2040 Practical Exam - Data Warehousing and Data Mining

**Student:** Gift Wanjiru Gachunga (ID: xxx662)
**Course:** DSA 2040 - Data Warehousing and Data Mining
**Exam Type:** End Semester Practical Exam
**Total Marks:** 100

## Project Overview

This repository contains the complete implementation of the DSA 2040 practical exam covering Data Warehousing (50 marks) and Data Mining (50 marks) tasks.

## Project Structure

```
DSA-2040_Practical_Exam_Gift_662/
├── data_warehousing/
│   ├── sql_scripts/          # SQL CREATE statements and OLAP queries
│   ├── visualizations/       # Charts and diagrams
│   ├── reports/             # Analysis reports
│   ├── schema_design.py     # Star schema design and explanation
│   ├── etl_retail.py        # ETL process implementation
│   └── retail_dw.db         # SQLite database (generated)
├── data_mining/
│   ├── datasets/            # Generated/downloaded datasets
│   ├── visualizations/      # Charts and plots
│   ├── reports/             # Analysis reports
│   ├── preprocessing_iris.py    # Data preprocessing and exploration
│   ├── clustering_iris.py       # K-Means clustering implementation
│   └── mining_iris_basket.py    # Classification and association rules
├── DSA 2040 FS 2025 End Semester Exam.pdf
└── README.md
```

## Requirements

### Python Libraries
```bash
pip install pandas numpy scikit-learn matplotlib seaborn mlxtend faker
```

### Software
- Python 3.x
- SQLite3 (usually pre-installed)
- DB Browser for SQLite (optional, for viewing database)

## Section 1: Data Warehousing (50 Marks)

### Task 1: Data Warehouse Design (15 Marks)
**Files:**
- `data_warehousing/schema_design.py`
- `data_warehousing/sql_scripts/create_tables.sql`
- `data_warehousing/visualizations/star_schema_diagram.png`

**Description:** Designed a star schema for a retail data warehouse with:
- 1 Fact Table (SalesFact)
- 4 Dimension Tables (CustomerDim, ProductDim, TimeDim, StoreDim)

**Run:**
```bash
python data_warehousing/schema_design.py
```

### Task 2: ETL Process Implementation (20 Marks)
**Files:**
- `data_warehousing/etl_retail.py`
- `data_warehousing/retail_dw.db`

**Description:** Complete ETL pipeline that:
- Extracts data from synthetic retail dataset
- Transforms data (calculates TotalSales, filters outliers, creates dimensions)
- Loads into SQLite database

**Run:**
```bash
python data_warehousing/etl_retail.py
```

### Task 3: OLAP Queries and Analysis (15 Marks)
**Files:**
- `data_warehousing/sql_scripts/olap_queries.sql`
- `data_warehousing/visualizations/sales_analysis.png`
- `data_warehousing/reports/olap_analysis.md`

**Description:** Implemented OLAP operations:
- Roll-up: Total sales by country and quarter
- Drill-down: Sales by country and month
- Slice: Sales for specific product category

## Section 2: Data Mining (50 Marks)

### Task 1: Data Preprocessing and Exploration (15 Marks)
**Files:**
- `data_mining/preprocessing_iris.py`
- `data_mining/visualizations/pairplot.png`
- `data_mining/visualizations/correlation_heatmap.png`
- `data_mining/visualizations/boxplots.png`

**Description:** Preprocessing and exploration of Iris dataset including:
- Missing value handling
- Min-Max normalization
- Statistical analysis
- Visualizations

**Run:**
```bash
python data_mining/preprocessing_iris.py
```

### Task 2: Clustering (15 Marks)
**Files:**
- `data_mining/clustering_iris.py`
- `data_mining/visualizations/cluster_visualization.png`
- `data_mining/visualizations/elbow_curve.png`
- `data_mining/reports/clustering_analysis.md`

**Description:** K-Means clustering implementation with:
- Optimal k determination using elbow method
- Cluster quality evaluation using Adjusted Rand Index
- Visualization of clusters

**Run:**
```bash
python data_mining/clustering_iris.py
```

### Task 3: Classification and Association Rule Mining (20 Marks)
**Files:**
- `data_mining/mining_iris_basket.py`
- `data_mining/visualizations/decision_tree.png`
- `data_mining/visualizations/classifier_comparison.png`
- `data_mining/reports/mining_analysis.md`

**Description:**
- Part A: Decision Tree and KNN classification with performance metrics
- Part B: Apriori algorithm for association rule mining on transactional data

**Run:**
```bash
python data_mining/mining_iris_basket.py
```

## Datasets Used

### Data Warehousing
- **Synthetic Retail Data**: Generated using Python's Faker library and numpy
- Structure: ~1000 transactions with InvoiceNo, CustomerID, ProductID, Quantity, UnitPrice, InvoiceDate, Country
- Generation ensures reproducibility with random seed

### Data Mining
- **Iris Dataset**: Loaded from scikit-learn built-in datasets
- 150 samples, 4 features (sepal/petal dimensions), 3 classes (species)
- **Synthetic Transactional Data**: Generated for association rule mining
- 50 transactions with 20 unique items

## Self-Assessment

### Completed Tasks
- [x] Task 1: Data Warehouse Design (15/15 marks)
- [x] Task 2: ETL Process Implementation (20/20 marks)
- [x] Task 3: OLAP Queries and Analysis (15/15 marks)
- [x] Data Mining Task 1: Preprocessing and Exploration (15/15 marks)
- [x] Data Mining Task 2: Clustering (15/15 marks)
- [x] Data Mining Task 3: Classification and Association Rules (20/20 marks)

### Total Expected Score: 100/100

### Key Strengths
- All code is well-commented and modular
- Comprehensive error handling and logging
- Clear visualizations with proper labels
- Detailed analysis reports
- Reproducible results using random seeds

### Challenges Faced
- Ensuring synthetic data mimics real-world patterns
- Balancing code complexity with readability
- Creating meaningful visualizations for all analyses

## How to Run Everything

1. **Install dependencies:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn mlxtend faker
   ```

2. **Run all tasks in order:**
   ```bash
   # Data Warehousing
   python data_warehousing/schema_design.py
   python data_warehousing/etl_retail.py

   # Data Mining
   python data_mining/preprocessing_iris.py
   python data_mining/clustering_iris.py
   python data_mining/mining_iris_basket.py
   ```

3. **View database:**
   ```bash
   sqlite3 data_warehousing/retail_dw.db
   # Or use DB Browser for SQLite GUI
   ```

## References
- Pandas Documentation: https://pandas.pydata.org/docs/
- Scikit-learn Documentation: https://scikit-learn.org/stable/documentation.html
- SQLite Documentation: https://www.sqlite.org/docs.html
- Matplotlib Documentation: https://matplotlib.org/stable/contents.html
- Seaborn Documentation: https://seaborn.pydata.org/


---
**Repository:** https://github.com/zawadi-wanjiru/DSA-2040_Practical_Exam_Gift_662
**Last Updated:** December 12, 2025
