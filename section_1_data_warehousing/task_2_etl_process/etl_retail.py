"""
DSA 2040 Practical Exam - Section 1, Task 2
ETL Process Implementation for Retail Data Warehouse
Student: Gift Wanjiru Gachunga (662)

This script implements a complete ETL pipeline:
- Extract: Generate synthetic retail data
- Transform: Calculate metrics, create dimensions, filter outliers
- Load: Load into SQLite database matching the star schema from Task 1
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import random
import logging

# Set random seed for reproducibility
np.random.seed(662)
random.seed(662)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_data(num_rows=1000):
    """
    Generate synthetic retail transaction data

    Args:
        num_rows: Number of transaction rows to generate

    Returns:
        pandas DataFrame with synthetic retail data
    """
    logger.info(f"Generating {num_rows} synthetic transactions...")

    # Define data parameters
    countries = ['UK', 'Germany', 'France', 'Spain', 'Netherlands',
                 'Belgium', 'Switzerland', 'Italy', 'Portugal', 'Austria']
    products = [
        ('ELEC-001', 'Laptop Pro 15"', 'Electronics', 'Computers', 899.99),
        ('ELEC-002', 'Wireless Mouse', 'Electronics', 'Accessories', 29.99),
        ('ELEC-003', 'USB-C Cable', 'Electronics', 'Accessories', 12.99),
        ('ELEC-004', 'Bluetooth Speaker', 'Electronics', 'Audio', 79.99),
        ('CLOTH-001', 'Cotton T-Shirt', 'Clothing', 'Tops', 19.99),
        ('CLOTH-002', 'Jeans Classic', 'Clothing', 'Bottoms', 49.99),
        ('CLOTH-003', 'Winter Jacket', 'Clothing', 'Outerwear', 129.99),
        ('HOME-001', 'Coffee Maker', 'Home & Garden', 'Kitchen', 59.99),
        ('HOME-002', 'Desk Lamp', 'Home & Garden', 'Lighting', 34.99),
        ('SPORT-001', 'Yoga Mat', 'Sports', 'Fitness', 24.99),
        ('SPORT-002', 'Running Shoes', 'Sports', 'Footwear', 89.99),
        ('BOOK-001', 'Python Programming', 'Books', 'Technology', 39.99),
    ]

    # Generate date range (last 2 years)
    end_date = datetime(2025, 8, 12)
    start_date = end_date - timedelta(days=730)

    # Generate transactions
    data = []
    for i in range(num_rows):
        invoice_no = f'INV-{100000 + i}'
        customer_id = f'CUST-{random.randint(1, 100):04d}'

        # Random date within range
        days_diff = (end_date - start_date).days
        random_days = random.randint(0, days_diff)
        invoice_date = start_date + timedelta(days=random_days)

        # Random product
        product = random.choice(products)
        stock_code, description, category, subcategory, base_price = product

        # Add some price variation (+/- 10%)
        unit_price = base_price * random.uniform(0.9, 1.1)

        # Quantity between 1 and 50
        quantity = random.randint(1, 50)

        # Random country
        country = random.choice(countries)

        data.append({
            'InvoiceNo': invoice_no,
            'StockCode': stock_code,
            'Description': description,
            'Category': category,
            'Subcategory': subcategory,
            'Quantity': quantity,
            'InvoiceDate': invoice_date.strftime('%Y-%m-%d %H:%M:%S'),
            'UnitPrice': round(unit_price, 2),
            'CustomerID': customer_id,
            'Country': country
        })

    # Add some outliers that should be filtered out
    # Negative quantities
    for i in range(5):
        idx = random.randint(0, len(data) - 1)
        data[idx]['Quantity'] = -random.randint(1, 10)

    # Zero or negative prices
    for i in range(5):
        idx = random.randint(0, len(data) - 1)
        data[idx]['UnitPrice'] = 0

    df = pd.DataFrame(data)
    logger.info(f"Generated {len(df)} rows of synthetic data")

    return df


def extract_data():
    """
    EXTRACT phase: Generate or load retail data

    Returns:
        pandas DataFrame with raw retail data
    """
    logger.info("=== EXTRACT PHASE ===")

    # Generate synthetic data
    df = generate_synthetic_data(num_rows=1000)

    # Save to CSV for reference
    csv_path = 'retail_data.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved raw data to {csv_path}")
    logger.info(f"Extracted {len(df)} rows")

    return df


def transform_data(df):
    """
    TRANSFORM phase: Clean, calculate, and prepare data for loading

    Args:
        df: Raw DataFrame from extract phase

    Returns:
        Tuple of transformed DataFrames (sales_fact, dimensions)
    """
    logger.info("=== TRANSFORM PHASE ===")

    initial_rows = len(df)
    logger.info(f"Initial rows: {initial_rows}")

    # Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    logger.info("Converted InvoiceDate to datetime format")

    # Handle missing values (check for any)
    missing_before = df.isnull().sum().sum()
    df = df.dropna()
    if missing_before > 0:
        logger.info(f"Removed {missing_before} missing values")

    # Remove outliers: Quantity < 0 or UnitPrice <= 0
    df_clean = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    removed_outliers = len(df) - len(df_clean)
    logger.info(f"Removed {removed_outliers} outlier rows (negative quantity or invalid price)")

    # Filter data for last year (from August 12, 2024 to August 12, 2025)
    cutoff_date = datetime(2024, 8, 12)
    df_clean = df_clean[df_clean['InvoiceDate'] >= cutoff_date]
    logger.info(f"Filtered to last year: {len(df_clean)} rows remaining")

    # Calculate TotalSales
    df_clean['TotalSales'] = df_clean['Quantity'] * df_clean['UnitPrice']
    logger.info("Calculated TotalSales = Quantity * UnitPrice")

    # Extract time dimensions
    df_clean['Year'] = df_clean['InvoiceDate'].dt.year
    df_clean['Month'] = df_clean['InvoiceDate'].dt.month
    df_clean['Quarter'] = df_clean['InvoiceDate'].dt.quarter
    df_clean['DayOfWeek'] = df_clean['InvoiceDate'].dt.day_name()
    df_clean['Date'] = df_clean['InvoiceDate'].dt.date

    logger.info(f"Final transformed rows: {len(df_clean)}")

    return df_clean


def create_dimensions(df):
    """
    Create dimension tables from transformed data

    Args:
        df: Transformed DataFrame

    Returns:
        Dictionary of dimension DataFrames
    """
    logger.info("=== CREATING DIMENSION TABLES ===")

    # Customer Dimension
    customer_dim = df.groupby('CustomerID').agg({
        'Country': 'first',
        'TotalSales': 'sum',
        'InvoiceNo': 'count'
    }).reset_index()
    customer_dim.columns = ['customer_id', 'country', 'total_purchases', 'transaction_count']

    # Assign regions and age groups randomly for demo
    regions = {'UK': 'Northern Europe', 'Germany': 'Central Europe',
               'France': 'Western Europe', 'Spain': 'Southern Europe',
               'Netherlands': 'Northern Europe', 'Belgium': 'Western Europe',
               'Switzerland': 'Central Europe', 'Italy': 'Southern Europe',
               'Portugal': 'Southern Europe', 'Austria': 'Central Europe'}

    customer_dim['region'] = customer_dim['country'].map(regions)
    customer_dim['customer_name'] = customer_dim['customer_id'].apply(
        lambda x: f"Customer {x.split('-')[1]}")
    customer_dim['age_group'] = np.random.choice(
        ['18-25', '26-35', '36-45', '46-55', '56+'],
        size=len(customer_dim))
    customer_dim['customer_segment'] = np.random.choice(
        ['Regular', 'Premium', 'VIP'],
        size=len(customer_dim),
        p=[0.6, 0.3, 0.1])

    logger.info(f"Created CustomerDim with {len(customer_dim)} unique customers")

    # Product Dimension
    product_dim = df[['StockCode', 'Description', 'Category',
                      'Subcategory', 'UnitPrice']].drop_duplicates()
    product_dim.columns = ['product_id', 'product_name', 'product_category',
                          'product_subcategory', 'unit_cost']
    product_dim['brand'] = product_dim['product_category'].apply(
        lambda x: f"{x[:4].upper()} Brand")

    logger.info(f"Created ProductDim with {len(product_dim)} unique products")

    # Time Dimension
    time_dim = df[['Date', 'DayOfWeek', 'Month', 'Quarter', 'Year']].drop_duplicates()
    time_dim['month_name'] = pd.to_datetime(time_dim['Date']).dt.month_name()
    time_dim['day_of_month'] = pd.to_datetime(time_dim['Date']).dt.day
    time_dim['is_weekend'] = time_dim['DayOfWeek'].isin(['Saturday', 'Sunday'])
    time_dim.columns = ['date', 'day_of_week', 'month', 'quarter', 'year',
                        'month_name', 'day_of_month', 'is_weekend']

    logger.info(f"Created TimeDim with {len(time_dim)} unique dates")

    # Store Dimension (simplified - assign stores randomly)
    store_data = [
        ('STORE-001', 'London Flagship', 'UK', 'London', 'Flagship', 'Large'),
        ('STORE-002', 'Berlin Central', 'Germany', 'Berlin', 'Standard', 'Medium'),
        ('STORE-003', 'Paris Outlet', 'France', 'Paris', 'Outlet', 'Small'),
    ]
    store_dim = pd.DataFrame(store_data, columns=[
        'store_id', 'store_name', 'store_country', 'store_city', 'store_type', 'store_size'])

    logger.info(f"Created StoreDim with {len(store_dim)} stores")

    return {
        'customer': customer_dim,
        'product': product_dim,
        'time': time_dim,
        'store': store_dim
    }


def load_to_database(df, dimensions, db_path='retail_dw.db'):
    """
    LOAD phase: Load data into SQLite database

    Args:
        df: Transformed fact data
        dimensions: Dictionary of dimension DataFrames
        db_path: Path to SQLite database file
    """
    logger.info("=== LOAD PHASE ===")

    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables using schema from Task 1
    logger.info("Creating database tables...")

    with open('../task_1_warehouse_design/create_tables.sql', 'r') as f:
        schema_sql = f.read()
        # Execute each statement separately
        for statement in schema_sql.split(';'):
            if statement.strip():
                try:
                    cursor.execute(statement)
                except sqlite3.OperationalError as e:
                    if 'already exists' not in str(e):
                        logger.warning(f"SQL execution warning: {e}")

    conn.commit()
    logger.info("Database schema created successfully")

    # Load dimension tables
    logger.info("Loading dimension tables...")

    # Load CustomerDim
    dimensions['customer'].to_sql('CustomerDim', conn, if_exists='replace', index=False)
    logger.info(f"Loaded {len(dimensions['customer'])} rows into CustomerDim")

    # Load ProductDim
    dimensions['product'].to_sql('ProductDim', conn, if_exists='replace', index=False)
    logger.info(f"Loaded {len(dimensions['product'])} rows into ProductDim")

    # Load TimeDim
    dimensions['time'].to_sql('TimeDim', conn, if_exists='replace', index=False)
    logger.info(f"Loaded {len(dimensions['time'])} rows into TimeDim")

    # Load StoreDim
    dimensions['store'].to_sql('StoreDim', conn, if_exists='replace', index=False)
    logger.info(f"Loaded {len(dimensions['store'])} rows into StoreDim")

    # Prepare fact table data
    logger.info("Preparing fact table...")

    # Get dimension keys
    customer_keys = pd.read_sql("SELECT rowid as customer_key, customer_id FROM CustomerDim", conn)
    product_keys = pd.read_sql("SELECT rowid as product_key, product_id FROM ProductDim", conn)
    time_keys = pd.read_sql("SELECT rowid as time_key, date FROM TimeDim", conn)
    store_keys = pd.read_sql("SELECT rowid as store_key, store_id FROM StoreDim", conn)

    # Merge to get foreign keys
    df['Date'] = df['Date'].astype(str)
    time_keys['date'] = time_keys['date'].astype(str)

    fact_table = df.merge(customer_keys.rename(columns={'customer_id': 'CustomerID'}), on='CustomerID', how='left')
    fact_table = fact_table.merge(product_keys.rename(columns={'product_id': 'StockCode'}), on='StockCode', how='left')
    fact_table = fact_table.merge(time_keys.rename(columns={'date': 'Date'}), on='Date', how='left')

    # Assign random stores
    fact_table['store_key'] = np.random.choice(store_keys['store_key'].values, size=len(fact_table))

    # Calculate profit (simplified: 30% of sales)
    fact_table['profit'] = fact_table['TotalSales'] * 0.3
    fact_table['discount_amount'] = 0

    # Select fact table columns
    sales_fact = fact_table[[
        'customer_key', 'product_key', 'time_key', 'store_key',
        'Quantity', 'UnitPrice', 'TotalSales', 'discount_amount', 'profit'
    ]].copy()

    sales_fact.columns = [
        'customer_key', 'product_key', 'time_key', 'store_key',
        'quantity', 'unit_price', 'total_sales', 'discount_amount', 'profit'
    ]

    # Load fact table
    sales_fact.to_sql('SalesFact', conn, if_exists='replace', index=False)
    logger.info(f"Loaded {len(sales_fact)} rows into SalesFact")

    # Verify data
    logger.info("\n=== DATABASE SUMMARY ===")
    for table in ['CustomerDim', 'ProductDim', 'TimeDim', 'StoreDim', 'SalesFact']:
        count = pd.read_sql(f"SELECT COUNT(*) as count FROM {table}", conn).iloc[0]['count']
        logger.info(f"{table}: {count} rows")

    conn.close()
    logger.info(f"Database saved to {db_path}")


def run_etl_pipeline():
    """
    Main ETL pipeline function
    Coordinates Extract, Transform, Load phases with logging
    """
    logger.info("=" * 60)
    logger.info("STARTING ETL PIPELINE FOR RETAIL DATA WAREHOUSE")
    logger.info("=" * 60)

    start_time = datetime.now()

    try:
        # Extract
        df_raw = extract_data()

        # Transform
        df_transformed = transform_data(df_raw)

        # Create dimensions
        dimensions = create_dimensions(df_transformed)

        # Load
        load_to_database(df_transformed, dimensions)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info("=" * 60)
        logger.info(f"ETL PIPELINE COMPLETED SUCCESSFULLY in {duration:.2f} seconds")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"ETL Pipeline failed: {str(e)}", exc_info=True)
        return False


if __name__ == "__main__":
    # Run the ETL pipeline
    success = run_etl_pipeline()

    if success:
        print("\n✓ ETL process completed successfully!")
        print("✓ Database file: retail_dw.db")
        print("✓ Source data: retail_data.csv")
        print("\nYou can now query the database using SQLite or DB Browser.")
    else:
        print("\n✗ ETL process failed. Check logs for details.")
