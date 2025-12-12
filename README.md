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
├── section_1_data_warehousing/
│   ├── task_1_warehouse_design/
│   │   ├── star_schema_diagram.png          # Star schema visualization
│   │   ├── create_tables.sql                # SQL CREATE statements
│   │   └── design_explanation.md            # Schema design rationale
│   ├── task_2_etl_process/
│   │   ├── etl_retail.py                    # ETL implementation
│   │   ├── retail_data.csv                  # Generated synthetic data
│   │   └── retail_dw.db                     # SQLite database
│   └── task_3_olap_queries/
│       ├── olap_queries.sql                 # Roll-up, Drill-down, Slice queries
│       ├── sales_visualization.png          # Query result visualization
│       └── analysis_report.md               # OLAP analysis report
├── section_2_data_mining/
│   ├── task_1_preprocessing/
│   │   ├── preprocessing_iris.py            # Data preprocessing & exploration
│   │   └── visualizations/                  # Pairplot, heatmap, boxplots
│   ├── task_2_clustering/
│   │   ├── clustering_iris.py               # K-Means clustering
│   │   └── visualizations/                  # Cluster plots, elbow curve
│   └── task_3_classification_association/
│       ├── mining_iris_basket.py            # Classification & Apriori
│       ├── datasets/                        # Synthetic transactional data
│       └── visualizations/                  # Decision tree, comparisons
├── DSA 2040 FS 2025 End Semester Exam.pdf
├── README.md
└── .gitignore
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
- `section_1_data_warehousing/task_1_warehouse_design/create_tables.sql`
- `section_1_data_warehousing/task_1_warehouse_design/star_schema_diagram.png`
- `section_1_data_warehousing/task_1_warehouse_design/design_explanation.md`

**Description:** Designed a star schema for a retail data warehouse with:
- 1 Fact Table (SalesFact)
- 4 Dimension Tables (CustomerDim, ProductDim, TimeDim, StoreDim)

### Task 2: ETL Process Implementation (20 Marks)
**Files:**
- `section_1_data_warehousing/task_2_etl_process/etl_retail.py`
- `section_1_data_warehousing/task_2_etl_process/retail_dw.db`
- `section_1_data_warehousing/task_2_etl_process/retail_data.csv`

**Description:** Complete ETL pipeline that:
- Extracts data from synthetic retail dataset
- Transforms data (calculates TotalSales, filters outliers, creates dimensions)
- Loads into SQLite database

**Run:**
```bash
python section_1_data_warehousing/task_2_etl_process/etl_retail.py
```

### Task 3: OLAP Queries and Analysis (15 Marks)
**Files:**
- `section_1_data_warehousing/task_3_olap_queries/olap_queries.sql`
- `section_1_data_warehousing/task_3_olap_queries/sales_visualization.png`
- `section_1_data_warehousing/task_3_olap_queries/analysis_report.md`

**Description:** Implemented OLAP operations:
- Roll-up: Total sales by country and quarter
- Drill-down: Sales by country and month
- Slice: Sales for specific product category

## Section 2: Data Mining (50 Marks)

### Task 1: Data Preprocessing and Exploration (15 Marks)
**Files:**
- `section_2_data_mining/task_1_preprocessing/preprocessing_iris.py`
- `section_2_data_mining/task_1_preprocessing/visualizations/`

**Description:** Preprocessing and exploration of Iris dataset including:
- Missing value handling
- Min-Max normalization
- Statistical analysis
- Visualizations (pairplot, correlation heatmap, boxplots)

**Run:**
```bash
python section_2_data_mining/task_1_preprocessing/preprocessing_iris.py
```

### Task 2: Clustering (15 Marks)
**Files:**
- `section_2_data_mining/task_2_clustering/clustering_iris.py`
- `section_2_data_mining/task_2_clustering/visualizations/`

**Description:** K-Means clustering implementation with:
- Optimal k determination using elbow method
- Cluster quality evaluation using Adjusted Rand Index
- Visualization of clusters

**Run:**
```bash
python section_2_data_mining/task_2_clustering/clustering_iris.py
```

### Task 3: Classification and Association Rule Mining (20 Marks)
**Files:**
- `section_2_data_mining/task_3_classification_association/mining_iris_basket.py`
- `section_2_data_mining/task_3_classification_association/datasets/`
- `section_2_data_mining/task_3_classification_association/visualizations/`

**Description:**
- Part A: Decision Tree and KNN classification with performance metrics
- Part B: Apriori algorithm for association rule mining on transactional data

**Run:**
```bash
python section_2_data_mining/task_3_classification_association/mining_iris_basket.py
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
   # Section 1: Data Warehousing
   python section_1_data_warehousing/task_2_etl_process/etl_retail.py

   # Section 2: Data Mining
   python section_2_data_mining/task_1_preprocessing/preprocessing_iris.py
   python section_2_data_mining/task_2_clustering/clustering_iris.py
   python section_2_data_mining/task_3_classification_association/mining_iris_basket.py
   ```

3. **View database:**
   ```bash
   sqlite3 section_1_data_warehousing/task_2_etl_process/retail_dw.db
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
