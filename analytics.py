"""
analytics.py
------------
Module 3 — Analytics Engine

Purpose:
    Read the cleaned data from ecommerce.db and compute all business
    metrics needed by the dashboard. This module acts as the data
    intelligence layer between the database and the visualisation layer.

    Every function in this file answers one business question.
    Each returns a pandas DataFrame ready to be plotted.

    The dashboard (app.py) will simply call these functions -
    it never touches the database directly.

How data is loaded:
    Each function connects to ecommerce.db via SQLAlchemy (or sqlite3
    fallback), runs a SQL query, and returns the result as a DataFrame.
    This mirrors real company workflows where analysts query databases
    rather than reading flat CSV files.

Metrics computed:
    1.  Top 10 products by revenue
    2.  Monthly revenue trend
    3.  Revenue by region
    4.  Return rate by category
    5.  Customer purchase frequency segmentation
    6.  Average order value by payment method
    7.  Month-over-month revenue growth %
    8.  Top 5 customers by lifetime value
    9.  KPI summary (total revenue, orders, AOV, return rate)

Inputs:
    ecommerce.db  (table: orders)  — produced by data_cleaner.py

Outputs:
    pandas DataFrames (consumed by app.py / forecasting.py)

Author: E-Commerce Dashboard Project
"""

import sqlite3
import pandas as pd

# SQLAlchemy preferred for production — falls back to sqlite3 gracefully
try:
    from sqlalchemy import create_engine
    USE_SQLALCHEMY = True
except ImportError:
    USE_SQLALCHEMY = False


# =============================================================================
# CONFIGURATION
# Single place to change the database path if needed.
# =============================================================================

DB_PATH  = "ecommerce.db"
DB_TABLE = "orders"


# =============================================================================
# DATABASE CONNECTION HELPER
# Every analytics function calls this to get a consistent connection.
# Centralising it means if DB_PATH changes, we only update one line.
# =============================================================================

def get_connection():
    """
    Return a database connection object.

    Uses SQLAlchemy engine if available (production standard),
    otherwise falls back to Python's built-in sqlite3.

    Returns:
        SQLAlchemy engine  OR  sqlite3.Connection object
    """
    if USE_SQLALCHEMY:
        return create_engine(f"sqlite:///{DB_PATH}")
    else:
        return sqlite3.connect(DB_PATH)


def run_query(sql: str) -> pd.DataFrame:
    """
    Execute a SQL query and return results as a pandas DataFrame.

    This is the single entry point for all database reads in this module.
    Every analytics function uses this helper instead of managing
    connections individually.

    Args:
        sql: SQL query string to execute against ecommerce.db

    Returns:
        Query result as a pandas DataFrame

    Raises:
        SystemExit if database file is not found
    """
    import os
    if not os.path.exists(DB_PATH):
        print(f"\n  ERROR: Database not found at '{DB_PATH}'")
        print("  Please run data_cleaner.py first.\n")
        raise SystemExit(1)

    conn = get_connection()
    df   = pd.read_sql(sql, conn)

    # Clean up connection properly depending on type
    if USE_SQLALCHEMY:
        conn.dispose()
    else:
        conn.close()

    return df


# =============================================================================
# METRIC 1 — TOP 10 PRODUCTS BY REVENUE
# Business question: Which products generate the most money?
# =============================================================================

def get_top_products(n: int = 10) -> pd.DataFrame:
    """
    Return the top N products ranked by total revenue.

    Business use:
        Inventory decisions, promotional targeting, identifying best-sellers.
        A product with high revenue but also high return rate (cross-referenced
        with Metric 4) signals a quality problem worth investigating.

    SQL logic:
        Group all orders by product_name, sum their total_amount,
        then sort descending and take the top N.

    Args:
        n: Number of top products to return (default 10)

    Returns:
        DataFrame with columns: product_name, category, revenue, total_orders
    """
    sql = f"""
        SELECT
            product_name,
            category,
            ROUND(SUM(total_amount), 2)  AS revenue,
            COUNT(order_id)              AS total_orders
        FROM {DB_TABLE}
        GROUP BY product_name, category
        ORDER BY revenue DESC
        LIMIT {n}
    """
    df = run_query(sql)
    return df


# =============================================================================
# METRIC 2 — MONTHLY REVENUE TREND
# Business question: How does revenue change month by month?
# =============================================================================

def get_monthly_revenue() -> pd.DataFrame:
    """
    Return total revenue grouped by calendar month, sorted chronologically.

    Business use:
        Trend analysis, seasonality detection, goal tracking.
        This is the most commonly requested chart in any sales dashboard.

    Date handling:
        order_date is stored as 'YYYY-MM-DD HH:MM:SS' string in SQLite.
        We use SQLite's SUBSTR function to extract the YYYY-MM portion
        directly in SQL — no pandas date parsing needed at query time.

    Returns:
        DataFrame with columns: month (YYYY-MM str), revenue, total_orders
    """
    sql = f"""
        SELECT
            SUBSTR(order_date, 1, 7)         AS month,
            ROUND(SUM(total_amount), 2)       AS revenue,
            COUNT(order_id)                   AS total_orders
        FROM {DB_TABLE}
        GROUP BY SUBSTR(order_date, 1, 7)
        ORDER BY month ASC
    """
    df = run_query(sql)

    # Convert month string → proper datetime for correct chart ordering
    df["month"] = pd.to_datetime(df["month"], format="%Y-%m")

    return df


# =============================================================================
# METRIC 3 — REVENUE BY REGION
# Business question: Which geographical regions are performing best?
# =============================================================================

def get_revenue_by_region() -> pd.DataFrame:
    """
    Return revenue, order count, and average order value per region.

    Business use:
        Regional performance comparisons, resource allocation,
        identifying underperforming markets that need sales investment.

    Note:
        'Unknown' region appears because ~8% of region values were null
        in the raw data and filled with 'Unknown' by the cleaner.
        The dashboard can filter this out visually if desired.

    Returns:
        DataFrame with columns:
            region, revenue, total_orders, avg_order_value, return_count
    """
    sql = f"""
        SELECT
            region,
            ROUND(SUM(total_amount), 2)              AS revenue,
            COUNT(order_id)                          AS total_orders,
            ROUND(AVG(total_amount), 2)              AS avg_order_value,
            SUM(CASE WHEN return_status = 'Returned'
                     THEN 1 ELSE 0 END)              AS return_count
        FROM {DB_TABLE}
        GROUP BY region
        ORDER BY revenue DESC
    """
    df = run_query(sql)

    # Calculate return rate per region as a percentage
    df["return_rate_pct"] = (
        (df["return_count"] / df["total_orders"]) * 100
    ).round(2)

    return df


# =============================================================================
# METRIC 4 — RETURN RATE BY CATEGORY
# Business question: Which categories have quality or satisfaction issues?
# =============================================================================

def get_return_rate_by_category() -> pd.DataFrame:
    """
    Return the percentage of orders returned per product category.

    Business use:
        High return rates signal product defects, misleading descriptions,
        sizing issues (Clothing), or customer dissatisfaction.
        Industry benchmark: >10% return rate is a warning sign.

    Formula:
        return_rate = (returned_orders / total_orders) × 100

    Returns:
        DataFrame with columns:
            category, total_orders, returned_orders, return_rate_pct
    """
    sql = f"""
        SELECT
            category,
            COUNT(order_id)                          AS total_orders,
            SUM(CASE WHEN return_status = 'Returned'
                     THEN 1 ELSE 0 END)              AS returned_orders,
            ROUND(SUM(total_amount), 2)              AS revenue
        FROM {DB_TABLE}
        GROUP BY category
        ORDER BY category ASC
    """
    df = run_query(sql)

    # Compute return rate as percentage
    df["return_rate_pct"] = (
        (df["returned_orders"] / df["total_orders"]) * 100
    ).round(2)

    return df


# =============================================================================
# METRIC 5 — CUSTOMER SEGMENTATION
# Business question: How many customers are loyal vs one-time buyers?
# =============================================================================

def get_customer_segments() -> pd.DataFrame:
    """
    Segment customers by purchase frequency into 3 loyalty tiers.

    Segmentation rules:
        1 order          → 'One-time'   (acquired but not retained)
        2–4 orders       → 'Occasional' (returning but not loyal)
        5+ orders        → 'Loyal'      (high-value repeat customers)

    Business use:
        Marketing teams use this to target campaigns.
        One-time customers → win-back emails
        Occasional        → loyalty program invitations
        Loyal             → VIP rewards, early access

    Approach:
        Step 1: SQL counts orders per customer
        Step 2: pandas assigns segment labels based on count
        Step 3: Group and count customers per segment

    Returns:
        DataFrame with columns: segment, customer_count, pct_of_total
    """
    # Step 1 — Count orders per customer in SQL
    sql = f"""
        SELECT
            customer_id,
            customer_name,
            COUNT(order_id)             AS order_count,
            ROUND(SUM(total_amount), 2) AS lifetime_value
        FROM {DB_TABLE}
        GROUP BY customer_id
        ORDER BY order_count DESC
    """
    df = run_query(sql)

    # Step 2 — Assign segment labels in pandas
    def assign_segment(order_count: int) -> str:
        if order_count == 1:
            return "One-time"
        elif order_count <= 4:
            return "Occasional"
        else:
            return "Loyal"

    df["segment"] = df["order_count"].apply(assign_segment)

    # Step 3 — Count customers per segment
    segment_counts = (
        df.groupby("segment")
        .agg(customer_count=("customer_id", "count"))
        .reset_index()
    )

    # Add percentage of total customers
    total_customers = segment_counts["customer_count"].sum()
    segment_counts["pct_of_total"] = (
        (segment_counts["customer_count"] / total_customers) * 100
    ).round(1)

    # Enforce a logical display order for charts
    order_map = {"One-time": 0, "Occasional": 1, "Loyal": 2}
    segment_counts["sort_order"] = segment_counts["segment"].map(order_map)
    segment_counts = segment_counts.sort_values("sort_order").drop(columns="sort_order")

    return segment_counts


# =============================================================================
# METRIC 6 — AVERAGE ORDER VALUE BY PAYMENT METHOD
# Business question: Do customers spend more with certain payment methods?
# =============================================================================

def get_avg_order_value_by_payment() -> pd.DataFrame:
    """
    Return the average order value (AOV) grouped by payment method.

    Business use:
        Credit card users typically spend more than cash-on-delivery users.
        This insight informs checkout UX decisions — e.g. promoting
        credit card payments to increase average basket size.

    Formula:
        AOV = total_revenue / total_orders  (per payment method)

    Returns:
        DataFrame with columns:
            payment_method, avg_order_value, total_orders, total_revenue
    """
    sql = f"""
        SELECT
            payment_method,
            ROUND(AVG(total_amount), 2)  AS avg_order_value,
            COUNT(order_id)              AS total_orders,
            ROUND(SUM(total_amount), 2)  AS total_revenue
        FROM {DB_TABLE}
        GROUP BY payment_method
        ORDER BY avg_order_value DESC
    """
    df = run_query(sql)
    return df


# =============================================================================
# METRIC 7 — MONTH-OVER-MONTH REVENUE GROWTH
# Business question: Is the business accelerating or slowing down?
# =============================================================================

def get_mom_growth() -> pd.DataFrame:
    """
    Calculate month-over-month (MoM) revenue growth percentage.

    Formula:
        MoM Growth % = ((current_month - previous_month) / previous_month) × 100

    Business use:
        The single most important metric for tracking business momentum.
        Consistently positive MoM = healthy growth.
        Sudden negative MoM = needs investigation.

    Approach:
        We reuse get_monthly_revenue() to get the base data,
        then use pandas .shift(1) to get the previous month's revenue,
        then compute the percentage change.

    Returns:
        DataFrame with columns:
            month, revenue, total_orders, prev_revenue, mom_growth_pct
    """
    # Reuse the monthly revenue function — no need to re-query
    df = get_monthly_revenue()

    # .shift(1) moves each row's value down by 1 — giving us previous month
    df["prev_revenue"] = df["revenue"].shift(1)

    # Calculate MoM growth — first month will be NaN (no previous month)
    df["mom_growth_pct"] = (
        ((df["revenue"] - df["prev_revenue"]) / df["prev_revenue"]) * 100
    ).round(2)

    return df


# =============================================================================
# METRIC 8 — TOP CUSTOMERS BY LIFETIME VALUE (LTV)
# Business question: Who are our most valuable customers?
# =============================================================================

def get_top_customers(n: int = 5) -> pd.DataFrame:
    """
    Return the top N customers ranked by total lifetime spend.

    Lifetime Value (LTV):
        The total revenue a single customer has generated across
        all their orders. Higher LTV = more valuable customer.

    Business use:
        Companies use LTV to:
        - Design VIP reward programs
        - Prioritise customer service for high-value accounts
        - Model what a "good customer" looks like for acquisition targeting

    Args:
        n: Number of top customers to return (default 5)

    Returns:
        DataFrame with columns:
            customer_id, customer_name, lifetime_value,
            total_orders, avg_order_value
    """
    sql = f"""
        SELECT
            customer_id,
            customer_name,
            ROUND(SUM(total_amount), 2)  AS lifetime_value,
            COUNT(order_id)              AS total_orders,
            ROUND(AVG(total_amount), 2)  AS avg_order_value
        FROM {DB_TABLE}
        GROUP BY customer_id, customer_name
        ORDER BY lifetime_value DESC
        LIMIT {n}
    """
    df = run_query(sql)
    return df


# =============================================================================
# METRIC 9 — KPI SUMMARY CARD VALUES
# Four headline numbers shown at the top of the dashboard overview page.
# =============================================================================

def get_kpi_summary() -> dict:
    """
    Return the four top-level KPI metrics for the dashboard summary cards.

    KPIs:
        Total Revenue    — sum of all clean order amounts
        Total Orders     — count of unique orders
        Avg Order Value  — total revenue / total orders
        Return Rate      — % of orders that were returned

    Returns:
        Dictionary with keys:
            total_revenue    (float)
            total_orders     (int)
            avg_order_value  (float)
            return_rate_pct  (float)
    """
    sql = f"""
        SELECT
            ROUND(SUM(total_amount), 2)                          AS total_revenue,
            COUNT(order_id)                                      AS total_orders,
            ROUND(AVG(total_amount), 2)                          AS avg_order_value,
            ROUND(
                SUM(CASE WHEN return_status = 'Returned'
                         THEN 1.0 ELSE 0 END)
                / COUNT(order_id) * 100, 2
            )                                                    AS return_rate_pct
        FROM {DB_TABLE}
    """
    df = run_query(sql)

    # Return as a plain dictionary — easy to unpack in dashboard metric cards
    return {
        "total_revenue":   df["total_revenue"].iloc[0],
        "total_orders":    int(df["total_orders"].iloc[0]),
        "avg_order_value": df["avg_order_value"].iloc[0],
        "return_rate_pct": df["return_rate_pct"].iloc[0],
    }


# =============================================================================
# RUN ALL — Convenience function used when testing this module directly
# The dashboard calls individual functions, not this one.
# =============================================================================

def run_all_analytics() -> dict:
    """
    Run every analytics function and return all results in one dictionary.

    Used for:
        - Testing this module standalone (python analytics.py)
        - Passing all metrics to the AI insights module (ai_insights.py)
        - Verifying everything works before building the dashboard

    Returns:
        Dictionary with all metric DataFrames and KPI summary keyed by name
    """
    print("\n" + "=" * 55)
    print("   MODULE 3 — ANALYTICS ENGINE")
    print("=" * 55)

    results = {}

    print("\n  Computing metrics...\n")

    # --- KPI Summary
    print("  [1/9] KPI Summary...")
    results["kpi"] = get_kpi_summary()
    kpi = results["kpi"]
    print(f"         Total Revenue   : ${kpi['total_revenue']:,.2f}")
    print(f"         Total Orders    : {kpi['total_orders']:,}")
    print(f"         Avg Order Value : ${kpi['avg_order_value']:,.2f}")
    print(f"         Return Rate     : {kpi['return_rate_pct']}%")

    # --- Top Products
    print("\n  [2/9] Top 10 Products by Revenue...")
    results["top_products"] = get_top_products()
    print(results["top_products"][["product_name", "revenue"]].to_string(index=False))

    # --- Monthly Revenue
    print("\n  [3/9] Monthly Revenue Trend...")
    results["monthly_revenue"] = get_monthly_revenue()
    df_m = results["monthly_revenue"]
    print(f"         Months covered  : {len(df_m)}")
    print(f"         Date range      : {df_m['month'].min().strftime('%Y-%m')} → {df_m['month'].max().strftime('%Y-%m')}")

    # --- Revenue by Region
    print("\n  [4/9] Revenue by Region...")
    results["revenue_by_region"] = get_revenue_by_region()
    print(results["revenue_by_region"][["region", "revenue", "return_rate_pct"]].to_string(index=False))

    # --- Return Rate by Category
    print("\n  [5/9] Return Rate by Category...")
    results["return_by_category"] = get_return_rate_by_category()
    print(results["return_by_category"][["category", "return_rate_pct"]].to_string(index=False))

    # --- Customer Segments
    print("\n  [6/9] Customer Segmentation...")
    results["customer_segments"] = get_customer_segments()
    print(results["customer_segments"].to_string(index=False))

    # --- AOV by Payment
    print("\n  [7/9] Avg Order Value by Payment Method...")
    results["aov_by_payment"] = get_avg_order_value_by_payment()
    print(results["aov_by_payment"][["payment_method", "avg_order_value"]].to_string(index=False))

    # --- MoM Growth
    print("\n  [8/9] Month-over-Month Revenue Growth...")
    results["mom_growth"] = get_mom_growth()
    df_g = results["mom_growth"].dropna(subset=["mom_growth_pct"])
    avg_growth = df_g["mom_growth_pct"].mean()
    print(f"         Avg MoM Growth  : {avg_growth:.2f}%")
    print(results["mom_growth"][["month", "revenue", "mom_growth_pct"]].tail(6).to_string(index=False))

    # --- Top Customers
    print("\n  [9/9] Top 5 Customers by Lifetime Value...")
    results["top_customers"] = get_top_customers()
    print(results["top_customers"][["customer_name", "lifetime_value", "total_orders"]].to_string(index=False))

    print("\n" + "=" * 55)
    print("   ALL METRICS COMPUTED SUCCESSFULLY")
    print("=" * 55)
    print("\n  Module 3 complete. Run forecasting.py next.\n")

    return results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_all_analytics()