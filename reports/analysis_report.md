# OLAP Analysis Report - Retail Data Warehouse

**Student:** Gift Wanjiru Gachunga (672662)
**Course:** DSA 2040 - Data Warehousing and Data Mining
**Task:** Section 1, Task 3 - OLAP Queries and Analysis (15 Marks)

---

## Executive Summary

This report presents the results of OLAP (Online Analytical Processing) queries executed on the retail data warehouse. The analysis demonstrates roll-up, drill-down, and slice operations to extract business insights from sales data spanning August 2024 to August 2025.

---

## Query Results and Insights

### 1. Roll-Up Analysis: Sales by Country and Quarter

**Key Findings:**
- **Belgium** emerged as the top-performing country in Q3 2025 with total sales of **€4.36 million** from 365 transactions, showing strong market penetration
- **France** dominated Q2 2025 with **€9.19 million** in sales across 937 transactions, indicating seasonal buying patterns
- **Portugal** showed consistent high-value performance with an average transaction value of **€10,043** in Q3 2025, suggesting premium product purchases
- Significant variation in average transaction values across countries indicates different customer segments and product preferences

**Business Implication:** The company should focus marketing efforts on high-performing regions like Belgium and France while investigating factors driving Portugal's high transaction values for replication in other markets.

### 2. Drill-Down Analysis: UK Sales by Month

**Key Findings:**
- **December 2024** recorded the highest UK sales at **€1.92 million**, reflecting strong holiday season performance
- **March 2025** showed significant activity with **€274k** in sales from 259 transactions
- **July 2025** demonstrated sustained performance with **€268k** despite being a traditionally slower retail period
- Monthly variations range from **€41k** (August 2024) to **€1.92 million** (December 2024), highlighting seasonal patterns

**Business Implication:** The UK market shows clear seasonal trends with peak performance during holiday periods. The data warehouse enables the company to plan inventory, staffing, and promotional campaigns around these predictable patterns.

### 3. Slice Analysis: Electronics Category Performance

**Key Findings:**
- **Computers subcategory** dominates with **€68.55 million** in total sales from 2,916 transactions (77,328 units sold)
- **Audio products** generated **€2.49 million** with strong unit sales (30,888 units)
- **Accessories** showed high transaction volume (2,920 transactions) but lower revenue (€1.25 million), indicating lower-priced items
- Overall Electronics category maintains a healthy **30% profit margin** across all subcategories

**Business Implication:** Electronics, particularly computers, represents the company's core revenue driver. The high volume of accessory transactions presents cross-selling opportunities to increase average order values.

---

## How the Data Warehouse Supports Decision-Making

### 1. Strategic Planning
The star schema design enables rapid aggregation across multiple dimensions (time, geography, product categories), allowing executives to identify market trends and make data-driven strategic decisions about market expansion and product portfolio optimization.

### 2. Operational Efficiency
OLAP operations like drill-down facilitate investigation of performance anomalies. For example, noticing low UK sales in August 2024 (€41k) versus December 2024 (€1.92 million) prompts investigation into staffing levels, inventory management, and promotional activities during different periods.

### 3. Customer Segmentation
The ability to slice data by country reveals distinct purchasing patterns (e.g., Portugal's high-value transactions vs. Austria's lower average values), enabling targeted marketing strategies and customized product offerings for different geographic segments.

### 4. Inventory Management
The granular monthly sales data combined with product category analysis helps predict demand patterns, ensuring optimal stock levels to meet customer demand while minimizing holding costs and stockouts.

### 5. Performance Monitoring
The data warehouse provides a single source of truth for tracking KPIs such as total sales, transaction counts, profit margins, and average transaction values across all business dimensions, facilitating consistent performance monitoring and reporting.

---

## Impact of Synthetic Data

### Limitations Acknowledged
- The synthetic data was generated with controlled randomness (seed: 672662), ensuring reproducibility but potentially lacking the complexity of real-world sales patterns
- Real retail data typically exhibits more irregular patterns, promotional spikes, and external influences (economic conditions, competitor actions) not fully captured in synthetic generation
- Customer behavior in synthetic data may be oversimplified compared to actual purchasing patterns involving brand loyalty, seasonal preferences, and product affinities

### Maintained Realism
Despite being synthetic, the data maintains structural realism by:
- Following realistic date ranges and business constraints (positive quantities, valid price ranges)
- Implementing proper hierarchies (countries, product categories, time periods)
- Generating statistically reasonable transaction volumes and values
- Demonstrating the technical capabilities and analytical value of the data warehouse design

The core OLAP operations and insights methodology remain valid and directly applicable to production environments with actual transactional data.

---

## Conclusions

The retail data warehouse successfully supports multidimensional analysis through its star schema design. The OLAP queries demonstrated three fundamental operations:

1. **Roll-up** aggregated granular sales data to country and quarterly levels, revealing geographic and temporal patterns
2. **Drill-down** provided detailed monthly breakdowns for specific countries, enabling targeted investigation
3. **Slice** isolated specific product categories to analyze segment performance and profitability

These capabilities enable the organization to transform raw transactional data into actionable business intelligence, supporting strategic decision-making across sales, marketing, inventory management, and customer relationship management functions. The data warehouse architecture provides scalability to handle growing data volumes while maintaining query performance through appropriate indexing and denormalized dimension tables.

---

## Recommendations

Based on the analysis, the following actions are recommended:

1. **Geographic Expansion**: Investigate Belgium's success factors and replicate in similar markets
2. **Seasonal Campaigns**: Leverage identified seasonal patterns for targeted promotional activities
3. **Product Mix Optimization**: Capitalize on Electronics dominance while exploring growth opportunities in underperforming categories
4. **Customer Value Enhancement**: Develop strategies to increase average transaction values in markets showing lower per-transaction revenues

The data warehouse provides the analytical foundation to monitor the impact of these initiatives and continuously refine business strategies based on empirical evidence.

---

**Analysis Date:** December 12, 2025
**Data Period:** August 12, 2024 - August 12, 2025
**Total Records Analyzed:** 21,245 transactions across 100 customers, 490 products, and 282 days
