"""
generate_data.py
----------------
Module 1 — Synthetic Data Generation

Purpose:
    Generate a realistic, messy e-commerce dataset that mimics
    real-world production data. This forms the foundation for all
    later modules (cleaning, analytics, dashboard, forecasting).

Output:
    data/raw_orders.csv  
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# =============================================================================
# STEP 1 — DEFINE REALISTIC VALUE POOLS
# These are the ingredients we'll randomly sample from.
# =============================================================================

# Products grouped by category
PRODUCTS = {
    "Electronics": [
        "Wireless Headphones", "Bluetooth Speaker", "USB-C Hub",
        "Mechanical Keyboard", "Gaming Mouse", "Webcam HD",
        "Laptop Stand", "Phone Charger", "Smart Watch", "Earbuds Pro"
    ],
    "Clothing": [
        "Men's Jacket", "Women's Dress", "Running Shoes",
        "Casual T-Shirt", "Denim Jeans", "Winter Scarf",
        "Sports Leggings", "Formal Shirt", "Sneakers", "Hoodie"
    ],
    "Home & Kitchen": [
        "Coffee Maker", "Air Fryer", "Blender Pro",
        "Non-stick Pan", "Knife Set", "Food Containers",
        "Electric Kettle", "Toaster Oven", "Rice Cooker", "Dish Rack"
    ],
    "Books": [
        "Python Crash Course", "Atomic Habits", "Clean Code",
        "The Lean Startup", "Deep Work", "Sapiens",
        "Data Science Handbook", "Zero to One", "The Pragmatic Programmer", "Rich Dad Poor Dad"
    ],
    "Sports": [
        "Yoga Mat", "Resistance Bands", "Dumbbells Set",
        "Jump Rope", "Water Bottle", "Gym Gloves",
        "Foam Roller", "Pull-up Bar", "Cycling Helmet", "Tennis Racket"
    ]
}

# Realistic unit prices per category (min, max)
PRICE_RANGES = {
    "Electronics": (15,  250),
    "Clothing":    (10,  120),
    "Home & Kitchen": (12, 180),
    "Books":       (8,   45),
    "Sports":      (5,   150)
}

REGIONS = ["North", "South", "East", "West", "Central"]

PAYMENT_METHODS = ["Credit Card", "Debit Card", "PayPal", "Bank Transfer", "Cash on Delivery"]

RETURN_STATUSES = ["Not Returned", "Returned"]
# ~15% return rate is realistic for e-commerce
RETURN_WEIGHTS = [0.85, 0.15]

# Sample customer first and last names
FIRST_NAMES = [
    "Ali", "Sara", "Ahmed", "Fatima", "Omar", "Aisha", "Hassan",
    "Zara", "Bilal", "Mariam", "Usman", "Hina", "Kamran", "Sana",
    "Tariq", "Nadia", "Fahad", "Amna", "Imran", "Rabia"
]
LAST_NAMES = [
    "Khan", "Ahmed", "Ali", "Hassan", "Malik", "Raza", "Sheikh",
    "Qureshi", "Siddiqui", "Chaudhry", "Mirza", "Baig", "Shah",
    "Akhtar", "Hussain", "Butt", "Ansari", "Rizvi", "Zaidi", "Nawaz"
]


# =============================================================================
# STEP 2 — HELPER FUNCTIONS
# =============================================================================

def generate_customer_pool(n_customers: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    #Create a fixed pool of customers so the same customer can appear
    multiple times (realistic — customers make repeat purchases).

    Args:
        n_customers: How many unique customers to create
        rng: NumPy random generator for reproducibility

    Returns:
        DataFrame with customer_id and customer_name columns
    """
    first = rng.choice(FIRST_NAMES, size=n_customers)
    last  = rng.choice(LAST_NAMES,  size=n_customers)
    names = [f"{f} {l}" for f, l in zip(first, last)]
    ids   = [f"CUST-{str(i).zfill(4)}" for i in range(1, n_customers + 1)]
    return pd.DataFrame({"customer_id": ids, "customer_name": names})


def generate_date_range(start: str, end: str, n: int, rng: np.random.Generator) -> list:
    """
    Generate n random dates between start and end as date strings.

    Args:
        start: Start date string 'YYYY-MM-DD'
        end:   End date string   'YYYY-MM-DD'
        n:     Number of dates to generate
        rng:   NumPy random generator

    Returns:
        List of date strings in 'YYYY-MM-DD' format
    """
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")
    delta    = (end_dt - start_dt).days
    offsets  = rng.integers(0, delta, size=n)
    dates    = [(start_dt + timedelta(days=int(d))).strftime("%Y-%m-%d") for d in offsets]
    return dates


# =============================================================================
# STEP 3 — GENERATE CLEAN BASE DATASET
# Start with 5000 perfectly valid rows, then corrupt them.
# =============================================================================

def generate_clean_data(n_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate a clean, valid e-commerce dataset before any corruption.

    Business logic:
        total_amount = quantity * unit_price * (1 - discount_pct / 100)

    Args:
        n_rows: Number of rows to generate
        rng:    NumPy random generator

    Returns:
        Clean DataFrame with all columns populated and valid
    """
    print(f"  Generating {n_rows} clean base rows...")

    # --- Order IDs: sequential, zero-padded  e.g. ORD-00001
    order_ids = [f"ORD-{str(i).zfill(5)}" for i in range(1, n_rows + 1)]

    # --- Customer pool: 500 unique customers making repeat purchases
    customer_pool = generate_customer_pool(500, rng)
    customer_idx  = rng.integers(0, len(customer_pool), size=n_rows)
    customer_ids  = customer_pool["customer_id"].iloc[customer_idx].values
    customer_names = customer_pool["customer_name"].iloc[customer_idx].values

    # --- Products and categories: pick a category first, then a product from it
    categories   = rng.choice(list(PRODUCTS.keys()), size=n_rows)
    product_names = np.array([
        rng.choice(PRODUCTS[cat]) for cat in categories
    ])

    # --- Quantity: mostly 1–5, occasionally higher (bulk orders)
    quantities = rng.integers(1, 6, size=n_rows)
    # ~5% of orders are bulk (qty 6–20)
    bulk_mask = rng.random(n_rows) < 0.05
    quantities[bulk_mask] = rng.integers(6, 21, size=bulk_mask.sum())

    # --- Unit price: based on category price range
    unit_prices = np.array([
        round(rng.uniform(*PRICE_RANGES[cat]), 2) for cat in categories
    ])

    # --- Discount: 0–30%, most orders have small or no discount
    discount_pcts = np.round(rng.choice(
        [0, 5, 10, 15, 20, 25, 30],
        size=n_rows,
        p=[0.40, 0.20, 0.15, 0.10, 0.08, 0.04, 0.03]
    ), 2)

    # --- Total amount: core business calculation
    total_amounts = np.round(
        quantities * unit_prices * (1 - discount_pcts / 100), 2
    )

    # --- Order dates: spread across 2 years (2023–2024)
    order_dates = generate_date_range("2023-01-01", "2024-12-31", n_rows, rng)

    # --- Region: random from pool
    regions = rng.choice(REGIONS, size=n_rows)

    # --- Return status: realistic 15% return rate
    return_statuses = rng.choice(
        RETURN_STATUSES, size=n_rows, p=RETURN_WEIGHTS
    )

    # --- Payment method
    payment_methods = rng.choice(PAYMENT_METHODS, size=n_rows)

    # --- Assemble into DataFrame
    df = pd.DataFrame({
        "order_id":       order_ids,
        "customer_id":    customer_ids,
        "customer_name":  customer_names,
        "product_name":   product_names,
        "category":       categories,
        "quantity":       quantities,
        "unit_price":     unit_prices,
        "discount_pct":   discount_pcts,
        "total_amount":   total_amounts,
        "order_date":     order_dates,
        "region":         regions,
        "return_status":  return_statuses,
        "payment_method": payment_methods
    })

    print(f"  Clean data generated: {len(df)} rows, {len(df.columns)} columns")
    return df


# =============================================================================
# STEP 4 — INTRODUCE DATA PROBLEMS (5 LAYERS OF CORRUPTION)
# Each function handles one specific type of real-world data issue.
# =============================================================================

def add_null_values(df: pd.DataFrame, rng: np.random.Generator, null_rate: float = 0.08) -> pd.DataFrame:
    """
    Layer 1 — Scatter ~8% null values across selected columns.

    Why this happens in real data:
        - Users skip optional form fields
        - System failures during data capture
        - API timeouts causing partial records

    Columns affected: customer_name, product_name, payment_method,
                      discount_pct, region (more realistic targets)
    """
    nullable_columns = ["customer_name", "product_name", "payment_method", "discount_pct", "region"]

    df = df.copy()
    n_rows = len(df)

    for col in nullable_columns:
        # Randomly select row indices to nullify for this column
        null_count = int(n_rows * null_rate)
        null_indices = rng.choice(df.index, size=null_count, replace=False)
        df.loc[null_indices, col] = np.nan

    # Count total nulls introduced
    total_nulls = df[nullable_columns].isnull().sum().sum()
    print(f"  [Nulls] Introduced {total_nulls} null values across {len(nullable_columns)} columns")
    return df


def add_duplicate_order_ids(df: pd.DataFrame, rng: np.random.Generator, dup_rate: float = 0.02) -> pd.DataFrame:
    """
    Layer 2 — Duplicate ~2% of order IDs.

    Why this happens in real data:
        - API retry logic sending the same order twice
        - ETL pipeline bugs re-inserting records
        - Database sync failures creating double entries

    We copy entire rows and re-append them — the order_id stays the same
    but other columns may have slight variation (realistic).
    """
    df = df.copy()
    n_duplicates = int(len(df) * dup_rate)

    # Pick random rows to duplicate
    dup_indices = rng.choice(df.index, size=n_duplicates, replace=False)
    dup_rows    = df.loc[dup_indices].copy()

    # Append duplicates to the bottom of the dataframe
    df = pd.concat([df, dup_rows], ignore_index=True)

    print(f"  [Duplicates] Added {n_duplicates} duplicate order rows (total rows now: {len(df)})")
    return df


def add_invalid_dates(df: pd.DataFrame, rng: np.random.Generator, invalid_rate: float = 0.01) -> pd.DataFrame:
    """
    Layer 3 — Replace ~1% of dates with invalid strings.

    Why this happens in real data:
        - Manual data entry errors
        - Different date formats from different source systems
        - Legacy system exports with corrupt date fields

    Examples of bad dates we'll inject:
        '2023-99-12'  → invalid month
        '2024-06-00'  → invalid day
        'N/A'         → missing value encoded as text
        'invalid_date'→ corrupted string
    """
    df = df.copy()
    n_invalid = int(len(df) * invalid_rate)

    # Pool of bad date strings to randomly pick from
    bad_dates = ["2023-99-12", "2024-06-00", "N/A", "invalid_date", "00-00-0000", "2024-13-45"]

    bad_indices = rng.choice(df.index, size=n_invalid, replace=False)
    df.loc[bad_indices, "order_date"] = rng.choice(bad_dates, size=n_invalid)

    print(f"  [Invalid Dates] Replaced {n_invalid} dates with invalid strings")
    return df


def add_negative_quantities(df: pd.DataFrame, rng: np.random.Generator, neg_rate: float = 0.01) -> pd.DataFrame:
    """
    Layer 4 — Make ~1% of quantities negative.

    Why this happens in real data:
        - System bugs recording returns as negative stock movements
        - Data entry errors (typed -3 instead of 3)
        - Accounting adjustments incorrectly stored in orders table
    """
    df = df.copy()
    n_negative = int(len(df) * neg_rate)

    neg_indices = rng.choice(df.index, size=n_negative, replace=False)
    df.loc[neg_indices, "quantity"] = df.loc[neg_indices, "quantity"] * -1

    print(f"  [Negative Quantities] Flipped {n_negative} quantities to negative")
    return df


def add_price_outliers(df: pd.DataFrame, rng: np.random.Generator, outlier_rate: float = 0.005) -> pd.DataFrame:
    """
    Layer 5 — Inject extreme price outliers into unit_price and total_amount.

    Why this happens in real data:
        - Test orders accidentally pushed to production
        - Currency conversion bugs (multiplied by 100x)
        - Manual price overrides with typos (e.g., 9999 instead of 99.99)
    """
    df = df.copy()
    n_outliers = int(len(df) * outlier_rate)

    # Extreme prices — clearly anomalous for any category
    extreme_prices = [4999.99, 7500.00, 9999.99, 12000.00, 15000.00]

    outlier_indices = rng.choice(df.index, size=n_outliers, replace=False)
    df.loc[outlier_indices, "unit_price"]   = rng.choice(extreme_prices, size=n_outliers)
    df.loc[outlier_indices, "total_amount"] = rng.choice(extreme_prices, size=n_outliers)

    print(f"  [Price Outliers] Injected {n_outliers} extreme price values")
    return df


# =============================================================================
# STEP 5 — MAIN PIPELINE
# Orchestrates all steps in the correct order.
# =============================================================================

def generate_dataset(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Main pipeline: generates clean data then corrupts it in 5 layers.

    Args:
        n_rows: Target number of base rows (duplicates will add ~2% more)
        seed:   Random seed for reproducibility

    Returns:
        Final messy DataFrame ready to save as CSV
    """
    print("=" * 55)
    print("   MODULE 1 — SYNTHETIC DATA GENERATION")
    print("=" * 55)

    # Seeded random generator — same seed = same dataset every run
    rng = np.random.default_rng(seed)

    # --- Step 2: Generate clean base
    print("\n[1/6] Generating clean base dataset...")
    df = generate_clean_data(n_rows, rng)

    # --- Step 3–7: Corrupt the data layer by layer
    print("\n[2/6] Introducing null values...")
    df = add_null_values(df, rng)

    print("\n[3/6] Adding duplicate order IDs...")
    df = add_duplicate_order_ids(df, rng)

    print("\n[4/6] Injecting invalid dates...")
    df = add_invalid_dates(df, rng)

    print("\n[5/6] Adding negative quantities...")
    df = add_negative_quantities(df, rng)

    print("\n[6/6] Injecting price outliers...")
    df = add_price_outliers(df, rng)

    # --- Shuffle rows so corrupted data is not clustered at the bottom
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"\n  Rows shuffled — corruption is now naturally scattered")

    return df


def save_dataset(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the final dataset to CSV.

    Args:
        df:          DataFrame to save
        output_path: File path for the CSV output
    """
    # Create the data/ directory if it doesn't exist yet
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"\n  Saved → {output_path}")


def print_summary(df: pd.DataFrame) -> None:
    """
    Print a quick summary so we can verify the dataset looks right.
    """
    print("\n" + "=" * 55)
    print("   DATASET SUMMARY")
    print("=" * 55)
    print(f"  Total rows        : {len(df)}")
    print(f"  Total columns     : {len(df.columns)}")
    print(f"  Null values total : {df.isnull().sum().sum()}")
    print(f"  Duplicate order_ids: {df.duplicated(subset='order_id').sum()}")
    print(f"  Negative quantities: {(df['quantity'] < 0).sum()}")
    print(f"  Date range        : {df['order_date'].min()} → {df['order_date'].max()}")
    print(f"  Columns           : {list(df.columns)}")
    print("=" * 55)


# =============================================================================
# STEP 6 — USER FILE LOADER
# Handles the case where the user provides their own CSV file.
# =============================================================================

# Columns the rest of the pipeline expects to work with.
# A user file doesn't need ALL of these — but needs enough to be useful.
REQUIRED_COLUMNS = ["order_id", "quantity", "unit_price", "order_date"]

EXPECTED_COLUMNS = [
    "order_id", "customer_id", "customer_name", "product_name",
    "category", "quantity", "unit_price", "discount_pct",
    "total_amount", "order_date", "region", "return_status", "payment_method"
]


def load_user_file(file_path: str) -> pd.DataFrame:
    """
    Load and lightly validate a user-provided CSV file.

    What we check:
        1. The file path actually exists on disk
        2. The file is a .csv (not Excel, JSON, etc.)
        3. The file is not empty
        4. It contains the minimum required columns

    We do NOT clean the data here — that is Module 2's job.
    We just make sure the file is readable and usable.

    Args:
        file_path: Path string the user typed in the terminal

    Returns:
        Raw DataFrame loaded from the user's file

    Raises:
        SystemExit if any validation check fails (with a clear message)
    """

    print(f"\n  Checking file: {file_path}")

    # --- Check 1: Does the file exist?
    if not os.path.exists(file_path):
        print(f"\n  ERROR: File not found at path: '{file_path}'")
        print("  Please check the path and try again.")
        raise SystemExit(1)

    # --- Check 2: Is it a CSV file?
    if not file_path.lower().endswith(".csv"):
        print(f"\n  ERROR: File must be a .csv file.")
        print(f"  You provided: '{file_path}'")
        print("  If you have an Excel file, export it as CSV first.")
        raise SystemExit(1)

    # --- Check 3: Try loading it
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"\n  ERROR: Could not read the file. Reason: {e}")
        raise SystemExit(1)

    # --- Check 4: Is it empty?
    if df.empty:
        print("\n  ERROR: The file is empty (0 rows).")
        raise SystemExit(1)

    # --- Check 5: Does it have the minimum required columns?
    user_cols   = [c.lower().strip() for c in df.columns]
    missing     = [c for c in REQUIRED_COLUMNS if c not in user_cols]

    if missing:
        print(f"\n  ERROR: Your file is missing these required columns: {missing}")
        print(f"  Columns found in your file: {list(df.columns)}")
        print("\n  Minimum required columns are:")
        for col in REQUIRED_COLUMNS:
            print(f"    - {col}")
        raise SystemExit(1)

    # --- All checks passed — show the user what was found
    found_expected    = [c for c in EXPECTED_COLUMNS if c in user_cols]
    not_found         = [c for c in EXPECTED_COLUMNS if c not in user_cols]

    print(f"\n  File loaded successfully!")
    print(f"  Rows             : {len(df)}")
    print(f"  Columns found    : {len(df.columns)}")
    print(f"  Matched columns  : {found_expected}")

    if not_found:
        # Not an error — just inform the user which columns are absent.
        # Module 2 (data_cleaner.py) will handle missing columns gracefully.
        print(f"  Missing columns  : {not_found}")
        print("  (These will be handled in the cleaning step)")

    return df


# =============================================================================
# STEP 7 — INTERACTIVE MENU
# Asks the user which path they want to take.
# =============================================================================

def show_menu() -> str:
    """
    Display the main menu and return the user's choice.

    Returns:
        '1' for generate synthetic data
        '2' for load user's own file
    """
    print("\n" + "=" * 55)
    print("   MODULE 1 — DATA INPUT")
    print("=" * 55)
    print("\n  How would you like to provide the dataset?\n")
    print("  [1]  Generate synthetic data (5000 rows, auto-created)")
    print("  [2]  Use my own CSV file (real-world data)\n")

    while True:
        choice = input("  Enter 1 or 2: ").strip()
        if choice in ("1", "2"):
            return choice
        print("  Invalid input. Please enter 1 or 2.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    OUTPUT_PATH = "data/raw_orders.csv"

    # Show menu and get user's choice
    choice = show_menu()

    if choice == "1":
        # --- PATH A: Generate synthetic messy dataset
        print("\n  Option 1 selected — Generating synthetic dataset...\n")
        df = generate_dataset(n_rows=5000, seed=42)

    else:
        # --- PATH B: Load user's own CSV file
        print("\n  Option 2 selected — Load your own CSV file")
        print("  Example path: data/my_sales.csv  or  C:/Users/you/sales.csv\n")
        file_path = input("  Enter full path to your CSV file: ").strip()
        df = load_user_file(file_path)

    # Both paths converge here — save to the same output location
    print("\nSaving to output location...")
    save_dataset(df, OUTPUT_PATH)

    # Show summary of whatever data we ended up with
    print_summary(df)

    print("\n  Module 1 complete. Run data_cleaner.py next.\n")