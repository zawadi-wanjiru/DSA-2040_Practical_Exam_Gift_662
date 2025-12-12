# Data Warehouse Schema Design - Task 1

**Student:** Gift Wanjiru Gachunga (662)
**Course:** DSA 2040 - Data Warehousing and Data Mining
**Task:** Section 1, Task 1 - Data Warehouse Design (15 Marks)

---

## Star Schema Design

### Overview
This data warehouse implements a **star schema** for a retail company that tracks sales across multiple product categories, customer demographics, and store locations.

### Schema Components

#### Fact Table: **SalesFact**
The central fact table contains sales transactions with the following:

**Measures (Quantitative Data):**
- `quantity` - Number of units sold
- `unit_price` - Price per unit
- `total_sales` - Total revenue (quantity × unit_price)
- `discount_amount` - Discount applied to the transaction
- `profit` - Profit margin on the sale

**Foreign Keys:**
- `customer_key` → CustomerDim
- `product_key` → ProductDim
- `time_key` → TimeDim
- `store_key` → StoreDim

#### Dimension Tables

**1. CustomerDim (Customer Dimension)**
- Demographics: name, age group, customer segment
- Geographic: country, region, city
- Supports customer demographics analysis

**2. ProductDim (Product Dimension)**
- Product details: name, brand, unit cost
- Categorization: category, subcategory
- Enables analysis by product category (Electronics, Clothing, etc.)

**3. TimeDim (Time Dimension)**
- Date hierarchy: day, month, quarter, year
- Additional attributes: day of week, month name, is_weekend
- Supports time-based analysis (sales by quarter, trends over time)

**4. StoreDim (Store Dimension)**
- Store identification: store ID, name
- Location: country, city
- Store characteristics: type (Flagship/Standard/Outlet), size
- Enables inventory and sales analysis by location

---

## Why Star Schema Over Snowflake Schema?

**1. Query Performance:**
Star schema provides better query performance because it requires fewer joins. When analyzing sales by product category per quarter, the query only needs to join the fact table with two dimension tables (ProductDim and TimeDim), rather than traversing normalized hierarchy tables as in a snowflake schema.

**2. Simplicity:**
The denormalized dimension tables make the schema easier to understand and query for business users and analysts. This simplicity reduces query complexity and makes it more intuitive to write SQL queries for common business questions like "What were total sales in Q1 by Electronics category?"

**3. ETL Efficiency:**
Star schema simplifies the ETL process since dimension tables are denormalized, reducing the number of lookups required during data loading. This is particularly important for retail data where frequent updates occur, and the redundancy trade-off (storing category names with each product) is minimal compared to the performance benefits for read-heavy analytical queries.

---

## Supported Query Types

This star schema effectively supports the required business queries:

1. **Total sales by product category per quarter**
   Join SalesFact → ProductDim (for category) → TimeDim (for quarter)

2. **Customer demographics analysis**
   Join SalesFact → CustomerDim (for age group, segment, region)

3. **Inventory trends**
   Join SalesFact → ProductDim → TimeDim → StoreDim (for location-based inventory analysis)

---

## Schema Diagram

```
                           STAR SCHEMA DESIGN

                         ┌─────────────────┐
                         │   CustomerDim   │
                         ├─────────────────┤
                         │ customer_key PK │
                         │ customer_id     │
                         │ customer_name   │
                         │ country         │
                         │ region          │
                         │ city            │
                         │ age_group       │
                         │ customer_segment│
                         └────────┬────────┘
                                  │
                                  │
         ┌─────────────────┐      │      ┌─────────────────┐
         │   ProductDim    │      │      │    TimeDim      │
         ├─────────────────┤      │      ├─────────────────┤
         │ product_key PK  │      │      │ time_key PK     │
         │ product_id      │      │      │ date            │
         │ product_name    │      │      │ day_of_week     │
         │ product_category│      │      │ month           │
         │ subcategory     │      │      │ month_name      │
         │ brand           │      │      │ quarter         │
         │ unit_cost       │      │      │ year            │
         └────────┬────────┘      │      │ is_weekend      │
                  │                │      └────────┬────────┘
                  │                │               │
                  │         ┌──────▼──────────┐    │
                  │         │   SalesFact     │    │
                  │         ├─────────────────┤    │
                  └────────►│ sales_key PK    │◄───┘
                            │ customer_key FK │
                            │ product_key FK  │
                            │ time_key FK     │
                            │ store_key FK    │
                            │─────────────────│
                            │ quantity        │
                            │ unit_price      │
                            │ total_sales     │
                            │ discount_amount │
                            │ profit          │
                            └────────▲────────┘
                                     │
                                     │
                         ┌───────────┴────────┐
                         │    StoreDim        │
                         ├────────────────────┤
                         │ store_key PK       │
                         │ store_id           │
                         │ store_name         │
                         │ store_country      │
                         │ store_city         │
                         │ store_type         │
                         │ store_size         │
                         └────────────────────┘
```

---

## Implementation Notes

- **Primary Keys:** All dimension tables use surrogate keys (auto-incrementing integers) for efficient joins
- **Natural Keys:** Business keys (customer_id, product_id, etc.) are maintained for reference
- **Data Types:** Appropriate SQLite data types chosen for each attribute
- **Constraints:** Check constraints ensure data quality (e.g., valid quarters, positive prices)
- **Indexes:** Created on foreign keys and commonly queried attributes for performance

---

## Conclusion

This star schema design provides an efficient, scalable foundation for the retail data warehouse. It balances denormalization for query performance with appropriate structure for maintaining data integrity, making it well-suited for the analytical requirements of tracking sales, customer behavior, and inventory trends.
