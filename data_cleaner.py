"""
data_cleaner.py
---------------
Module 2 — Data Cleaning Pipeline

Purpose:
    Take the messy raw dataset from Module 1 and transform it into
    a clean, reliable dataset ready for analytics, forecasting, and
    dashboard visualisation.

    Without this step:
        - Revenue totals would be wrong (outliers, negatives)
        - Time-series charts would crash (invalid dates)
        - Duplicate rows would inflate all metrics

Pipeline Order (order matters — each step depends on the previous):
    1.  Load raw data            understand what we have
    2.  Remove duplicates        one order_id = one row
    3.  Handle missing values    no silent NaN gaps
    4.  Fix invalid dates        parse then drop unparseable
    5.  Fix negative quantities  absolute value conversion
    6.  Remove price outliers    IQR method
    7.  Recalculate totals       ensure financial consistency
    8.  Save clean CSV           data/clean_orders.csv
    9.  Load into SQLite DB      ecommerce.db (for analytics SQL queries)

Inputs:
    data/raw_orders.csv

Outputs:
    data/clean_orders.csv
    ecommerce.db  (table: orders)
"""

import os
import sqlite3
import numpy as np
import pandas as pd

# SQLAlchemy is the production-grade way to interact with databases in Python.
# It is used when available (pip install sqlalchemy).
# We gracefully fall back to Python's built-in sqlite3 if not yet installed —
# both produce an identical ecommerce.db file.
try:
    from sqlalchemy import create_engine, text
    USE_SQLALCHEMY = True
except ImportError:
    USE_SQLALCHEMY = False


# =============================================================================
# CONFIGURATION
# Centralise all file paths in one place — easy to change later.
# =============================================================================

RAW_PATH   = "data/raw_orders.csv"
CLEAN_PATH = "data/clean_orders.csv"
DB_PATH    = "ecommerce.db"
DB_TABLE   = "orders"


# =============================================================================
# LOGGING HELPER
# Every cleaning step prints a before/after log so the pipeline is
# fully traceable — important in production data engineering.
# =============================================================================

def log(step: str, message: str) -> None:
    """
    Print a formatted log line for a cleaning step.

    Args:
        step:    Short step label  e.g. 'DUPLICATES'
        message: Detail message    e.g. 'Removed 100 rows'
    """
    print(f"  [{step}] {message}")


def log_row_count(df: pd.DataFrame, label: str) -> None:
    """
    Print the current row count with a label.
    Called before and after each major step to track row loss.

    Args:
        df:    Current DataFrame
        label: Context label e.g. 'After duplicate removal'
    """
    print(f"         → {label}: {len(df):,} rows")


# =============================================================================
# STEP 1 — LOAD RAW DATA
# Load the CSV and immediately report its quality state.
# =============================================================================

def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load raw_orders.csv and print an initial data quality report.

    Why we report first:
        Before touching anything, we need to understand the scale of
        problems. This mirrors what a data engineer does on day 1 with
        a new dataset.

    Args:
        path: File path to raw_orders.csv

    Returns:
        Raw DataFrame, completely unmodified

    Raises:
        SystemExit if the file does not exist
    """
    print("\n" + "=" * 55)
    print("  STEP 1 — LOADING RAW DATA")
    print("=" * 55)

    if not os.path.exists(path):
        print(f"\n  ERROR: Raw data file not found at '{path}'")
        print("  Please run generate_data.py first.\n")
        raise SystemExit(1)

    df = pd.read_csv(path)

    # --- Initial quality report
    print(f"\n  File loaded: {path}")
    log_row_count(df, "Total rows in raw file")
    log("INFO", f"Columns ({len(df.columns)}): {list(df.columns)}")

    print("\n  Null values per column:")
    null_counts = df.isnull().sum()
    for col, count in null_counts.items():
        if count > 0:
            pct = (count / len(df)) * 100
            print(f"         → {col}: {count} nulls ({pct:.1f}%)")

    total_nulls = null_counts.sum()
    log("INFO", f"Total null values across all columns: {total_nulls:,}")
    log("INFO", f"Duplicate order_ids detected: {df.duplicated(subset='order_id').sum():,}")
    log("INFO", f"Negative quantities detected: {(pd.to_numeric(df['quantity'], errors='coerce') < 0).sum():,}")

    return df


# =============================================================================
# STEP 2 — REMOVE DUPLICATE ORDERS
# Keep the first occurrence of each order_id, drop the rest.
# =============================================================================

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows based on order_id.

    Business rule:
        Each order_id must be unique. In Module 1 we duplicated ~2%
        of rows to simulate API retry errors. Here we keep only the
        first occurrence — on the assumption the first record is the
        original and later ones are retries.

    Args:
        df: Raw DataFrame

    Returns:
        DataFrame with duplicate order_ids removed
    """
    print("\n" + "=" * 55)
    print("  STEP 2 — REMOVING DUPLICATE ORDERS")
    print("=" * 55)

    rows_before = len(df)
    log_row_count(df, "Before duplicate removal")

    # keep='first' → retain the first occurrence, drop all subsequent
    df = df.drop_duplicates(subset="order_id", keep="first")

    rows_removed = rows_before - len(df)
    log("DUPLICATES", f"Removed {rows_removed:,} duplicate rows")
    log_row_count(df, "After duplicate removal")

    return df


# =============================================================================
# STEP 3 — HANDLE MISSING VALUES
# Different columns need different strategies — not a one-size-fits-all.
# =============================================================================

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill or drop null values with column-appropriate strategies.

    Strategy per column:
        customer_name  → fill with 'Unknown Customer'
                         (name is informational, not required for metrics)

        product_name   → fill with 'Unknown Product'
                         (we still want to count the order)

        payment_method → fill with 'Unknown'
                         (payment analysis will have an 'Unknown' category)

        discount_pct   → fill with 0.0
                         (safest assumption: no discount was applied)

        region         → fill with 'Unknown'
                         (regional analysis will show an 'Unknown' group)

        unit_price     → DROP rows where null
                         (cannot calculate revenue without a price)

        quantity       → DROP rows where null
                         (cannot calculate revenue without quantity)

    Args:
        df: DataFrame after duplicate removal

    Returns:
        DataFrame with nulls resolved
    """
    print("\n" + "=" * 55)
    print("  STEP 3 — HANDLING MISSING VALUES")
    print("=" * 55)

    rows_before = len(df)
    log_row_count(df, "Before null handling")

    # --- Strategy A: Fill with placeholder strings (informational columns)
    fill_map = {
        "customer_name":  "Unknown Customer",
        "product_name":   "Unknown Product",
        "payment_method": "Unknown",
        "region":         "Unknown",
    }
    for col, fill_value in fill_map.items():
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                df[col] = df[col].fillna(fill_value)
                log("NULLS", f"'{col}': filled {null_count:,} nulls → '{fill_value}'")

    # --- Strategy B: Fill discount_pct with 0 (numeric — assume no discount)
    if "discount_pct" in df.columns:
        null_count = df["discount_pct"].isnull().sum()
        if null_count > 0:
            df["discount_pct"] = df["discount_pct"].fillna(0.0)
            log("NULLS", f"'discount_pct': filled {null_count:,} nulls → 0.0")

    # --- Strategy C: Drop rows where critical numeric fields are null
    # We cannot compute total_amount without both price and quantity.
    critical_cols = ["unit_price", "quantity"]
    for col in critical_cols:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                df = df.dropna(subset=[col])
                log("NULLS", f"'{col}': dropped {null_count:,} rows with null values")

    rows_removed = rows_before - len(df)
    log("NULLS", f"Total rows removed in this step: {rows_removed:,}")
    log_row_count(df, "After null handling")

    return df


# =============================================================================
# STEP 4 — FIX INVALID DATES
# Parse dates strictly, then drop rows that cannot be parsed.
# =============================================================================

def fix_invalid_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert order_date to datetime, then remove unparseable rows.

    What we injected in Module 1:
        '2023-99-12', 'invalid_date', 'N/A', '00-00-0000', '2024-13-45'

    Approach:
        pd.to_datetime with errors='coerce' converts unparseable
        values to NaT (Not a Time) silently. We then drop NaT rows.

    Why we drop instead of fill:
        Dates are essential for all time-series analytics (monthly
        revenue trends, forecasting). A fake/guessed date is worse
        than no date because it would silently distort charts.

    Args:
        df: DataFrame after null handling

    Returns:
        DataFrame with order_date as proper datetime, bad rows dropped
    """
    print("\n" + "=" * 55)
    print("  STEP 4 — FIXING INVALID DATES")
    print("=" * 55)

    rows_before = len(df)
    log_row_count(df, "Before date fixing")

    # Convert to datetime — unparseable → NaT
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    # Count how many became NaT
    nat_count = df["order_date"].isnull().sum()
    log("DATES", f"Found {nat_count:,} unparseable dates → converting to NaT")

    # Drop rows with NaT dates
    df = df.dropna(subset=["order_date"])

    rows_removed = rows_before - len(df)
    log("DATES", f"Removed {rows_removed:,} rows with invalid dates")
    log("DATES", f"Date range after fix: {df['order_date'].min().date()} → {df['order_date'].max().date()}")
    log_row_count(df, "After date fixing")

    return df


# =============================================================================
# STEP 5 — FIX NEGATIVE QUANTITIES
# Convert negative values to their absolute value.
# =============================================================================

def fix_negative_quantities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert negative quantity values to positive using abs().

    Why abs() instead of dropping:
        Dropping would lose the order entirely. A negative quantity
        is most likely a data entry error (typed -3 instead of 3),
        so the order itself is valid — only the sign is wrong.
        Converting keeps more data in the pipeline.

    Note:
        quantity must be numeric first. We coerce just in case
        any string values slipped through previous steps.

    Args:
        df: DataFrame after date fixing

    Returns:
        DataFrame with all positive quantities
    """
    print("\n" + "=" * 55)
    print("  STEP 5 — FIXING NEGATIVE QUANTITIES")
    print("=" * 55)

    # Coerce to numeric first — any non-numeric becomes NaN
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

    # Count negatives before fixing
    neg_count = (df["quantity"] < 0).sum()
    log("QUANTITY", f"Found {neg_count:,} negative quantities")

    # Convert negative → positive using absolute value
    df["quantity"] = df["quantity"].abs()

    log("QUANTITY", f"Converted {neg_count:,} values using abs()")
    log("QUANTITY", f"Quantity range after fix: {df['quantity'].min()} → {df['quantity'].max()}")

    # Drop any rows where quantity became NaN after coercion
    nan_qty = df["quantity"].isnull().sum()
    if nan_qty > 0:
        df = df.dropna(subset=["quantity"])
        log("QUANTITY", f"Dropped {nan_qty:,} rows where quantity was non-numeric")

    log_row_count(df, "After quantity fix")

    return df


# =============================================================================
# STEP 6 — REMOVE PRICE OUTLIERS (IQR METHOD)
# Remove statistically extreme unit_price values.
# =============================================================================

def remove_price_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with extreme unit_price values using the IQR method.

    IQR (Interquartile Range) method — industry standard:
        Q1  = 25th percentile of unit_price
        Q3  = 75th percentile of unit_price
        IQR = Q3 - Q1

        Lower bound = Q1 - (1.5 × IQR)
        Upper bound = Q3 + (1.5 × IQR)

        Rows outside [lower, upper] are considered outliers.

    Why 1.5 × IQR?
        This is the standard Tukey fence. It flags values that are
        unusually far from the bulk of the data without being too
        aggressive on natural variation.

    What we injected in Module 1:
        unit_prices like 4999.99, 7500.00, 9999.99, 12000.00, 15000.00
        These are 40–100x normal prices, clearly anomalous.

    Args:
        df: DataFrame after quantity fixing

    Returns:
        DataFrame with outlier rows removed from unit_price
    """
    print("\n" + "=" * 55)
    print("  STEP 6 — REMOVING PRICE OUTLIERS (IQR)")
    print("=" * 55)

    rows_before = len(df)

    # Ensure unit_price is numeric
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    df = df.dropna(subset=["unit_price"])

    # --- Calculate IQR boundaries
    Q1  = df["unit_price"].quantile(0.25)
    Q3  = df["unit_price"].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)

    log("OUTLIERS", f"unit_price Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
    log("OUTLIERS", f"Acceptable range: {lower_bound:.2f} → {upper_bound:.2f}")

    # --- Count and remove outliers
    outlier_mask = (df["unit_price"] < lower_bound) | (df["unit_price"] > upper_bound)
    outlier_count = outlier_mask.sum()

    log("OUTLIERS", f"Found {outlier_count:,} price outlier rows")
    df = df[~outlier_mask]  # keep only non-outlier rows

    rows_removed = rows_before - len(df)
    log("OUTLIERS", f"Removed {rows_removed:,} rows total")
    log("OUTLIERS", f"unit_price range after fix: {df['unit_price'].min():.2f} → {df['unit_price'].max():.2f}")
    log_row_count(df, "After outlier removal")

    return df


# =============================================================================
# STEP 7 — RECALCULATE TOTAL AMOUNT
# After fixing quantities and prices, recompute the revenue column.
# =============================================================================

def recalculate_totals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute total_amount from scratch using clean quantity, unit_price,
    and discount_pct values.

    Formula (same as Module 1):
        total_amount = quantity × unit_price × (1 - discount_pct / 100)

    Why recalculate?
        The total_amount values in raw data may be wrong because:
          - Quantities were negative (now fixed to positive)
          - Prices had outliers (now removed)
          - discount_pct had nulls (now filled with 0)
        So any total_amount from the raw file cannot be trusted.
        We throw it away and compute it fresh from clean columns.

    Args:
        df: DataFrame after all individual column fixes

    Returns:
        DataFrame with corrected total_amount column
    """
    print("\n" + "=" * 55)
    print("  STEP 7 — RECALCULATING TOTAL AMOUNTS")
    print("=" * 55)

    # Store old totals for comparison
    old_total_revenue = df["total_amount"].sum() if "total_amount" in df.columns else 0

    # Recalculate using the clean column values
    df["total_amount"] = (
        df["quantity"].astype(float)
        * df["unit_price"].astype(float)
        * (1 - df["discount_pct"].astype(float) / 100)
    ).round(2)

    new_total_revenue = df["total_amount"].sum()

    log("TOTALS", f"Revenue before recalculation : ${old_total_revenue:,.2f}")
    log("TOTALS", f"Revenue after  recalculation : ${new_total_revenue:,.2f}")
    log("TOTALS", f"Difference                   : ${abs(new_total_revenue - old_total_revenue):,.2f}")
    log("TOTALS", "total_amount column is now consistent with cleaned data")

    return df


# =============================================================================
# STEP 8 — SAVE CLEAN DATASET
# Write the final clean DataFrame to CSV.
# =============================================================================

def save_clean_data(df: pd.DataFrame, path: str) -> None:
    """
    Save the cleaned DataFrame to a CSV file.

    Args:
        df:   Cleaned DataFrame
        path: Output file path (data/clean_orders.csv)
    """
    print("\n" + "=" * 55)
    print("  STEP 8 — SAVING CLEAN DATASET")
    print("=" * 55)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

    file_size_kb = os.path.getsize(path) / 1024
    log("SAVE", f"Clean CSV saved → {path}")
    log("SAVE", f"File size: {file_size_kb:.1f} KB")
    log("SAVE", f"Rows saved: {len(df):,}")


# =============================================================================
# STEP 9 — LOAD DATA INTO SQLITE DATABASE
# Store the cleaned data in a relational database for SQL queries.
# =============================================================================

def load_into_database(df: pd.DataFrame, db_path: str, table_name: str) -> None:
    """
    Load the cleaned DataFrame into a SQLite database.

    Uses SQLAlchemy if available (recommended for production).
    Falls back to Python's built-in sqlite3 automatically.

    Why SQLite + SQLAlchemy?
        - SQLite is file-based — no server setup needed
        - SQLAlchemy is the industry standard ORM for Python
        - Later modules (analytics.py) will run SQL queries on this DB
        - This simulates real company infrastructure where data lives
          in a relational database, not just a CSV file

    The table is replaced on every run so that re-running the cleaner
    always gives you a fresh, consistent database state.

    Args:
        df:         Cleaned DataFrame to store
        db_path:    Path for the SQLite file (ecommerce.db)
        table_name: Name of the table inside the DB (orders)
    """
    print("\n" + "=" * 55)
    print("  STEP 9 — LOADING INTO SQLITE DATABASE")
    print("=" * 55)

    if USE_SQLALCHEMY:
        # --- Production path: SQLAlchemy (install with: pip install sqlalchemy)
        engine = create_engine(f"sqlite:///{db_path}")
        df.to_sql(name=table_name, con=engine, if_exists="replace", index=False)

        # Verify row count
        with engine.connect() as conn:
            result    = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            row_count = result.fetchone()[0]
        engine.dispose()
        log("DATABASE", "Using SQLAlchemy engine")

    else:
        # --- Fallback path: built-in sqlite3 (no install needed)
        # pandas can write directly to a sqlite3 connection object
        conn = sqlite3.connect(db_path)
        df.to_sql(name=table_name, con=conn, if_exists="replace", index=False)

        # Verify row count
        cursor    = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        conn.close()
        log("DATABASE", "Using built-in sqlite3 (install sqlalchemy for production)")

    log("DATABASE", f"Table '{table_name}' created in {db_path}")
    log("DATABASE", f"Rows confirmed in database: {row_count:,}")
    log("DATABASE", "Ready for SQL queries in analytics.py")


# =============================================================================
# FINAL SUMMARY — Print a clean before/after report
# =============================================================================

def print_cleaning_summary(raw_rows: int, df_clean: pd.DataFrame) -> None:
    """
    Print a final before/after summary of the entire cleaning pipeline.

    Args:
        raw_rows: Row count of the original raw file
        df_clean: Final cleaned DataFrame
    """
    clean_rows    = len(df_clean)
    rows_removed  = raw_rows - clean_rows
    pct_retained  = (clean_rows / raw_rows) * 100

    print("\n" + "=" * 55)
    print("   CLEANING PIPELINE SUMMARY")
    print("=" * 55)
    print(f"  Raw rows loaded          : {raw_rows:,}")
    print(f"  Clean rows retained      : {clean_rows:,}")
    print(f"  Total rows removed       : {rows_removed:,}")
    print(f"  Data retention rate      : {pct_retained:.1f}%")
    print(f"  Remaining nulls          : {df_clean.isnull().sum().sum()}")
    print(f"  Negative quantities left : {(df_clean['quantity'] < 0).sum()}")
    print(f"  Date range               : {df_clean['order_date'].min().date()} → {df_clean['order_date'].max().date()}")
    print(f"  Total clean revenue      : ${df_clean['total_amount'].sum():,.2f}")
    print(f"  Output CSV               : {CLEAN_PATH}")
    print(f"  Output Database          : {DB_PATH}  (table: {DB_TABLE})")
    print("=" * 55)


# =============================================================================
# MAIN PIPELINE — Orchestrates all 9 steps in order
# =============================================================================

def run_cleaning_pipeline() -> pd.DataFrame:
    """
    Execute the full data cleaning pipeline from raw CSV to clean DB.

    Each step is isolated in its own function so it can be:
        - tested independently
        - reused in other scripts
        - easily debugged if one step fails

    Returns:
        Final cleaned DataFrame (also saved to CSV and SQLite)
    """
    print("\n" + "=" * 55)
    print("   MODULE 2 — DATA CLEANING PIPELINE")
    print("=" * 55)

    # Step 1 — Load
    df = load_raw_data(RAW_PATH)
    raw_rows = len(df)          # save original count for final summary

    # Step 2 — Duplicates
    df = remove_duplicates(df)

    # Step 3 — Missing values
    df = handle_missing_values(df)

    # Step 4 — Invalid dates
    df = fix_invalid_dates(df)

    # Step 5 — Negative quantities
    df = fix_negative_quantities(df)

    # Step 6 — Price outliers
    df = remove_price_outliers(df)

    # Step 7 — Recalculate totals (must come after all column fixes)
    df = recalculate_totals(df)

    # Step 8 — Save clean CSV
    save_clean_data(df, CLEAN_PATH)

    # Step 9 — Load into SQLite
    load_into_database(df, DB_PATH, DB_TABLE)

    # Final report
    print_cleaning_summary(raw_rows, df)

    return df


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    print("\n  Starting data cleaning pipeline...")
    print("  Make sure generate_data.py has been run first.\n")

    df_clean = run_cleaning_pipeline()

    print("\n  Module 2 complete.")
    print("  Next step: run analytics.py\n")