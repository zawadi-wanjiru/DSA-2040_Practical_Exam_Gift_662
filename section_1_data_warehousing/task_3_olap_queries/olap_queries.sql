-- ============================================================================
-- DSA 2040 Practical Exam - Section 1, Task 3
-- OLAP Queries and Analysis
-- Student: Gift Wanjiru Gachunga (672662)
-- ============================================================================

-- These queries demonstrate OLAP operations on the retail data warehouse:
-- 1. Roll-up: Aggregate data to higher level (country and quarter)
-- 2. Drill-down: Show detailed data at lower level (country by month)
-- 3. Slice: Filter data for specific dimension value (product category)

-- ============================================================================
-- QUERY 1: ROLL-UP
-- Total sales by country and quarter
-- This aggregates sales data at the country and quarter level
-- ============================================================================

SELECT
    c.country,
    t.quarter,
    t.year,
    COUNT(DISTINCT s.rowid) as total_transactions,
    SUM(s.quantity) as total_units_sold,
    ROUND(SUM(s.total_sales), 2) as total_sales_amount,
    ROUND(AVG(s.total_sales), 2) as avg_transaction_value,
    ROUND(SUM(s.profit), 2) as total_profit
FROM SalesFact s
JOIN CustomerDim c ON s.customer_key = c.rowid
JOIN TimeDim t ON s.time_key = t.rowid
GROUP BY c.country, t.quarter, t.year
ORDER BY t.year DESC, t.quarter DESC, total_sales_amount DESC;

-- ============================================================================
-- QUERY 2: DRILL-DOWN
-- Sales details for a specific country (UK) by month
-- This provides more granular detail than the roll-up query
-- ============================================================================

SELECT
    c.country,
    t.year,
    t.month,
    t.month_name,
    COUNT(DISTINCT s.rowid) as total_transactions,
    COUNT(DISTINCT s.customer_key) as unique_customers,
    SUM(s.quantity) as total_units_sold,
    ROUND(SUM(s.total_sales), 2) as total_sales_amount,
    ROUND(AVG(s.total_sales), 2) as avg_transaction_value,
    ROUND(SUM(s.profit), 2) as total_profit,
    ROUND(SUM(s.profit) / SUM(s.total_sales) * 100, 2) as profit_margin_pct
FROM SalesFact s
JOIN CustomerDim c ON s.customer_key = c.rowid
JOIN TimeDim t ON s.time_key = t.rowid
WHERE c.country = 'UK'
GROUP BY c.country, t.year, t.month, t.month_name
ORDER BY t.year DESC, t.month DESC;

-- ============================================================================
-- QUERY 3: SLICE
-- Total sales for Electronics category
-- This filters the data cube for a specific dimension value
-- ============================================================================

SELECT
    p.product_category,
    p.product_subcategory,
    COUNT(DISTINCT s.rowid) as total_transactions,
    SUM(s.quantity) as total_units_sold,
    ROUND(SUM(s.total_sales), 2) as total_sales_amount,
    ROUND(AVG(s.unit_price), 2) as avg_unit_price,
    ROUND(SUM(s.profit), 2) as total_profit,
    ROUND(SUM(s.profit) / SUM(s.total_sales) * 100, 2) as profit_margin_pct
FROM SalesFact s
JOIN ProductDim p ON s.product_key = p.rowid
WHERE p.product_category = 'Electronics'
GROUP BY p.product_category, p.product_subcategory
ORDER BY total_sales_amount DESC;

-- ============================================================================
-- BONUS QUERY: DICE
-- Multi-dimensional slice showing Electronics sales in UK for Q3-Q4 2024
-- ============================================================================

SELECT
    p.product_category,
    p.product_name,
    c.country,
    t.quarter,
    COUNT(s.rowid) as transactions,
    SUM(s.quantity) as units_sold,
    ROUND(SUM(s.total_sales), 2) as total_sales
FROM SalesFact s
JOIN ProductDim p ON s.product_key = p.rowid
JOIN CustomerDim c ON s.customer_key = c.rowid
JOIN TimeDim t ON s.time_key = t.rowid
WHERE p.product_category = 'Electronics'
  AND c.country = 'UK'
  AND t.quarter IN (3, 4)
  AND t.year = 2024
GROUP BY p.product_category, p.product_name, c.country, t.quarter
ORDER BY total_sales DESC
LIMIT 10;
