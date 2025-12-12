-- ============================================================================
-- DSA 2040 Practical Exam - Section 1, Task 1
-- Data Warehouse Schema Design for Retail Company
-- Student: Gift Wanjiru Gachunga (672662)
-- ============================================================================

-- STAR SCHEMA DESIGN
-- This schema supports queries for:
-- - Total sales by product category per quarter
-- - Customer demographics analysis
-- - Inventory trends

-- ============================================================================
-- DIMENSION TABLES
-- ============================================================================

-- Customer Dimension Table
-- Stores customer demographic information
CREATE TABLE CustomerDim (
    customer_key INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id VARCHAR(20) NOT NULL UNIQUE,
    customer_name VARCHAR(100),
    country VARCHAR(50),
    region VARCHAR(50),
    city VARCHAR(50),
    age_group VARCHAR(20),
    customer_segment VARCHAR(30),
    CONSTRAINT check_age_group CHECK (age_group IN ('18-25', '26-35', '36-45', '46-55', '56+'))
);

-- Product Dimension Table
-- Stores product information including categories
CREATE TABLE ProductDim (
    product_key INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id VARCHAR(20) NOT NULL UNIQUE,
    product_name VARCHAR(200),
    product_category VARCHAR(50),
    product_subcategory VARCHAR(50),
    brand VARCHAR(50),
    unit_cost DECIMAL(10, 2),
    CONSTRAINT check_category CHECK (product_category IN ('Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books'))
);

-- Time Dimension Table
-- Stores date information with hierarchical time attributes
CREATE TABLE TimeDim (
    time_key INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL UNIQUE,
    day_of_week VARCHAR(10),
    day_of_month INTEGER,
    month INTEGER,
    month_name VARCHAR(10),
    quarter INTEGER,
    year INTEGER,
    is_weekend BOOLEAN,
    CONSTRAINT check_month CHECK (month BETWEEN 1 AND 12),
    CONSTRAINT check_quarter CHECK (quarter BETWEEN 1 AND 4)
);

-- Store Dimension Table
-- Stores information about retail store locations
CREATE TABLE StoreDim (
    store_key INTEGER PRIMARY KEY AUTOINCREMENT,
    store_id VARCHAR(20) NOT NULL UNIQUE,
    store_name VARCHAR(100),
    store_country VARCHAR(50),
    store_city VARCHAR(50),
    store_type VARCHAR(30),
    store_size VARCHAR(20),
    CONSTRAINT check_store_type CHECK (store_type IN ('Flagship', 'Standard', 'Outlet'))
);

-- ============================================================================
-- FACT TABLE
-- ============================================================================

-- Sales Fact Table
-- Central fact table containing sales transactions with measures
CREATE TABLE SalesFact (
    sales_key INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Foreign Keys to Dimension Tables
    customer_key INTEGER NOT NULL,
    product_key INTEGER NOT NULL,
    time_key INTEGER NOT NULL,
    store_key INTEGER NOT NULL,

    -- Measures (Facts)
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10, 2) NOT NULL,
    total_sales DECIMAL(12, 2) NOT NULL,
    discount_amount DECIMAL(10, 2) DEFAULT 0,
    profit DECIMAL(12, 2),

    -- Constraints
    CONSTRAINT fk_customer FOREIGN KEY (customer_key) REFERENCES CustomerDim(customer_key),
    CONSTRAINT fk_product FOREIGN KEY (product_key) REFERENCES ProductDim(product_key),
    CONSTRAINT fk_time FOREIGN KEY (time_key) REFERENCES TimeDim(time_key),
    CONSTRAINT fk_store FOREIGN KEY (store_key) REFERENCES StoreDim(store_key),
    CONSTRAINT check_quantity CHECK (quantity > 0),
    CONSTRAINT check_price CHECK (unit_price > 0)
);

-- ============================================================================
-- INDEXES FOR QUERY PERFORMANCE
-- ============================================================================

-- Indexes on foreign keys in fact table for faster joins
CREATE INDEX idx_sales_customer ON SalesFact(customer_key);
CREATE INDEX idx_sales_product ON SalesFact(product_key);
CREATE INDEX idx_sales_time ON SalesFact(time_key);
CREATE INDEX idx_sales_store ON SalesFact(store_key);

-- Indexes on dimension tables for common queries
CREATE INDEX idx_product_category ON ProductDim(product_category);
CREATE INDEX idx_time_quarter_year ON TimeDim(quarter, year);
CREATE INDEX idx_customer_country ON CustomerDim(country);
CREATE INDEX idx_store_country ON StoreDim(store_country);

-- ============================================================================
-- END OF SCHEMA
-- ============================================================================
