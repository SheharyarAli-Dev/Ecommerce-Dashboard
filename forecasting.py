"""
forecasting.py
--------------
Module 4 — Revenue Forecasting

Purpose:
    Add predictive analytics on top of the descriptive analytics
    already computed in Module 3. While analytics.py answers
    "what happened?", this module answers "what will likely happen next?"

    We use Linear Regression to learn the revenue trend over time and
    project it forward by 3 months. The result includes a confidence
    interval band to communicate prediction uncertainty honestly.

Forecasting approach: why Linear Regression?
    - Simple and interpretable (we can explain the slope in plain English)
    - Works well for short datasets (we have 24 months)
    - No hyperparameter tuning needed
    - Perfect for a dashboard where business users need to understand it
    Advanced models (ARIMA, Prophet, LSTM) are more powerful but need
    larger datasets and are overkill for this project at this stage.

Model formula:
    Revenue = (slope × month_index) + intercept
    Where month_index is a simple integer: 1 for Jan 2023, 24 for Dec 2024

Input:
    Monthly revenue DataFrame from analytics.get_monthly_revenue()
    Columns: month (datetime64), revenue (float64), total_orders (int64)

Output:
    forecast_df - DataFrame with columns:
        month           (datetime64)  - the future month
        forecast        (float)       - predicted revenue
        lower_ci        (float)       - lower confidence bound
        upper_ci        (float)       - upper confidence bound
        is_forecast     (bool)        - True = predicted, False = historical

    model_summary - dict with slope, intercept, R² score, RMSE

Author: E-Commerce Dashboard Project
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Import the monthly revenue function from Module 3
# This keeps forecasting.py clean — it doesn't touch the DB directly
from analytics import get_monthly_revenue


# =============================================================================
# CONFIGURATION
# =============================================================================

FORECAST_MONTHS = 3      # How many future months to predict
CONFIDENCE_LEVEL = 1.96  # 95% confidence interval multiplier (z-score)


# =============================================================================
# STEP 1 - PREPARE MONTHLY DATA
# Convert the datetime-indexed revenue series into ML-ready format.
# =============================================================================

def prepare_monthly_data() -> pd.DataFrame:
    """
    Load monthly revenue and add a numeric time index for regression.

    Machine learning models cannot work with dates directly.
    We convert each month into a plain integer index:
        2023-01 → 1
        2023-02 → 2
        ...
        2024-12 → 24

    This integer becomes our X (feature).
    Revenue becomes our y (target).

    Returns:
        DataFrame with columns:
            month         (datetime64) — original date
            revenue       (float)      — actual monthly revenue
            total_orders  (int)        — orders that month
            time_index    (int)        — 1-based integer for regression
    """
    df = get_monthly_revenue()

    # Ensure data is sorted chronologically — critical for time-series
    df = df.sort_values("month").reset_index(drop=True)

    # Create 1-based integer index: month 1, 2, 3 ... N
    df["time_index"] = np.arange(1, len(df) + 1)

    return df


# =============================================================================
# STEP 2 - TRAIN REGRESSION MODEL
# Fit a LinearRegression on the historical time → revenue data.
# =============================================================================

def train_forecast_model(df: pd.DataFrame) -> tuple:
    """
    Train a Linear Regression model on the monthly revenue data.

    The model learns the overall trend line:
        Revenue = slope × month_index + intercept

    The slope tells us how much revenue grows (or shrinks) per month.
    For example, slope = 800 means revenue grows ~$800 per month on average.

    Args:
        df: Prepared DataFrame from prepare_monthly_data()
            Must contain columns: time_index, revenue

    Returns:
        Tuple of:
            model       — trained LinearRegression object
            residuals   — array of (actual - predicted) for each training month
                          used later to compute confidence intervals
            r2          — R² score (0–1, higher = better fit)
            rmse        — Root Mean Squared Error in dollars
    """
    # Reshape X to 2D array — scikit-learn requires shape (n_samples, n_features)
    X = df["time_index"].values.reshape(-1, 1)
    y = df["revenue"].values

    # Train the model
    model = LinearRegression()
    model.fit(X, y)

    # Predictions on training data (to evaluate model quality)
    y_pred = model.predict(X)

    # Residuals: how far off each training prediction was
    # These capture the typical "error" the model makes
    residuals = y - y_pred

    # Model quality metrics
    r2   = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    return model, residuals, r2, rmse


# =============================================================================
# STEP 3 — PREDICT FUTURE MONTHS
# Use the trained model to forecast the next N months.
# =============================================================================

def predict_future_revenue(
    model: LinearRegression,
    last_time_index: int,
    last_date: pd.Timestamp,
    n_months: int = FORECAST_MONTHS
) -> pd.DataFrame:
    """
    Generate revenue predictions for the next N months after the data ends.

    Args:
        model:           Trained LinearRegression model
        last_time_index: The time_index of the last historical month (e.g. 24)
        last_date:       The date of the last historical month (e.g. 2024-12)
        n_months:        Number of future months to forecast (default 3)

    Returns:
        DataFrame with columns:
            month       (datetime64) — future month dates
            time_index  (int)        — integer index continuing from history
            forecast    (float)      — predicted revenue for that month
    """
    # Build future time indices continuing from where history ends
    # e.g. if history ends at index 24, future = [25, 26, 27]
    future_indices = np.arange(
        last_time_index + 1,
        last_time_index + n_months + 1
    ).reshape(-1, 1)

    # Predict revenue for those future indices
    future_revenue = model.predict(future_indices)

    # Generate the actual future month dates using pandas date offset
    # pd.DateOffset(months=i) correctly handles month-end boundaries
    future_dates = [
        last_date + pd.DateOffset(months=i)
        for i in range(1, n_months + 1)
    ]

    forecast_df = pd.DataFrame({
        "month":      future_dates,
        "time_index": future_indices.flatten(),
        "forecast":   np.round(future_revenue, 2)
    })

    return forecast_df


# =============================================================================
# STEP 4 — CALCULATE CONFIDENCE INTERVAL
# Add lower and upper bounds to communicate prediction uncertainty.
# =============================================================================

def add_confidence_interval(
    forecast_df: pd.DataFrame,
    residuals: np.ndarray,
    confidence_level: float = CONFIDENCE_LEVEL
) -> pd.DataFrame:
    """
    Add a confidence interval band to the forecast predictions.

    How confidence intervals work here:
        We look at how wrong the model was on historical data (residuals).
        The standard deviation of those residuals tells us the typical
        prediction error. We multiply by the z-score for 95% confidence
        (1.96) to get the interval width.

        lower_ci = forecast - (1.96 × std_of_residuals)
        upper_ci = forecast + (1.96 × std_of_residuals)

    This means: "we are 95% confident the true revenue will fall
    between lower_ci and upper_ci."

    Why std of residuals instead of a fixed %?
        Because the interval should reflect how well the model actually
        fits the data. A volatile dataset produces wider intervals.
        A stable trend produces narrower intervals.

    Args:
        forecast_df:      DataFrame from predict_future_revenue()
        residuals:        Array of (actual - predicted) on training data
        confidence_level: Z-score multiplier (1.96 = 95%, 2.576 = 99%)

    Returns:
        forecast_df with added columns: lower_ci, upper_ci
    """
    # Standard deviation of residuals = typical prediction error in dollars
    std_residuals = np.std(residuals)

    margin = confidence_level * std_residuals

    forecast_df = forecast_df.copy()
    forecast_df["lower_ci"] = np.round(forecast_df["forecast"] - margin, 2)
    forecast_df["upper_ci"] = np.round(forecast_df["forecast"] + margin, 2)

    # Lower bound should not go below zero — revenue cannot be negative
    forecast_df["lower_ci"] = forecast_df["lower_ci"].clip(lower=0)

    return forecast_df


# =============================================================================
# STEP 5 — BUILD COMBINED DATAFRAME
# Merge historical actuals + forecast into one unified DataFrame.
# The dashboard uses this single DataFrame to draw the full chart.
# =============================================================================

def build_combined_dataframe(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine historical monthly revenue with the forecast into one DataFrame.

    The dashboard needs a single DataFrame to render the chart — both the
    solid historical line and the dashed forecast line with shaded CI band.

    Structure of combined output:
        Historical rows: month, revenue, forecast=NaN, lower_ci=NaN,
                         upper_ci=NaN, is_forecast=False
        Forecast rows:   month, revenue=NaN, forecast, lower_ci,
                         upper_ci, is_forecast=True

    Args:
        historical_df: From prepare_monthly_data() — actual revenue by month
        forecast_df:   From add_confidence_interval() — predicted future months

    Returns:
        Combined DataFrame sorted by month with is_forecast flag column
    """
    # --- Historical portion
    hist = historical_df[["month", "revenue"]].copy()
    hist["forecast"]    = np.nan
    hist["lower_ci"]    = np.nan
    hist["upper_ci"]    = np.nan
    hist["is_forecast"] = False

    # --- Forecast portion
    fcast = forecast_df[["month", "forecast", "lower_ci", "upper_ci"]].copy()
    fcast["revenue"]     = np.nan
    fcast["is_forecast"] = True

    # --- Combine and sort chronologically
    combined = pd.concat([hist, fcast], ignore_index=True)
    combined  = combined.sort_values("month").reset_index(drop=True)

    return combined


# =============================================================================
# MAIN FUNCTION — get_forecast()
# The single public function that app.py and forecasting tests call.
# Runs the entire pipeline and returns everything the dashboard needs.
# =============================================================================

def get_forecast() -> tuple:
    """
    Run the complete forecasting pipeline and return results.

    This is the only function app.py needs to call from this module.
    It orchestrates all steps internally.

    Pipeline:
        1. Load + prepare monthly data
        2. Train Linear Regression model
        3. Predict next 3 months
        4. Add confidence intervals
        5. Build combined historical + forecast DataFrame

    Returns:
        Tuple of:
            combined_df    — full DataFrame (historical + forecast) for charting
            forecast_only  — just the 3 forecast rows (for summary display)
            model_summary  — dict with slope, intercept, R², RMSE, trend label
    """
    # Step 1 — Prepare
    df = prepare_monthly_data()

    # Step 2 — Train
    model, residuals, r2, rmse = train_forecast_model(df)

    # Step 3 — Predict
    last_index = int(df["time_index"].iloc[-1])
    last_date  = df["month"].iloc[-1]

    forecast_df = predict_future_revenue(model, last_index, last_date)

    # Step 4 — Confidence intervals
    forecast_df = add_confidence_interval(forecast_df, residuals)

    # Step 5 — Combined DataFrame for the chart
    combined_df = build_combined_dataframe(df, forecast_df)

    # --- Model summary: plain English interpretation of the model
    slope     = model.coef_[0]
    intercept = model.intercept_

    # Describe the trend direction in plain English for dashboard display
    if slope > 500:
        trend_label = "Strong Growth"
    elif slope > 0:
        trend_label = "Moderate Growth"
    elif slope > -500:
        trend_label = "Slight Decline"
    else:
        trend_label = "Strong Decline"

    model_summary = {
        "slope":          round(slope, 2),
        "intercept":      round(intercept, 2),
        "r2_score":       round(r2, 4),
        "rmse":           round(rmse, 2),
        "trend_label":    trend_label,
        "monthly_growth": round(slope, 2),   # $ per month growth rate
        "std_error":      round(np.std(residuals), 2),
    }

    return combined_df, forecast_df, model_summary


# =============================================================================
# ENTRY POINT — run standalone to verify module works
# =============================================================================

if __name__ == "__main__":

    print("\n" + "=" * 55)
    print("   MODULE 4 — REVENUE FORECASTING")
    print("=" * 55)

    print("\n  Running forecasting pipeline...")

    combined_df, forecast_df, summary = get_forecast()

    # --- Model quality report
    print("\n  Model Summary")
    print("  " + "-" * 40)
    print(f"  Trend Direction  : {summary['trend_label']}")
    print(f"  Monthly Growth   : ${summary['monthly_growth']:,.2f} per month")
    print(f"  R² Score         : {summary['r2_score']}  (1.0 = perfect fit)")
    print(f"  RMSE             : ${summary['rmse']:,.2f}  (avg prediction error)")
    print(f"  Std Error (CI)   : ${summary['std_error']:,.2f}")

    # --- Forecast results
    print("\n  3-Month Revenue Forecast")
    print("  " + "-" * 40)
    print(f"  {'Month':<12} {'Forecast':>12} {'Lower (95%)':>14} {'Upper (95%)':>14}")
    print("  " + "-" * 55)
    for _, row in forecast_df.iterrows():
        month_str = row["month"].strftime("%Y-%m")
        print(
            f"  {month_str:<12}"
            f" ${row['forecast']:>11,.2f}"
            f" ${row['lower_ci']:>13,.2f}"
            f" ${row['upper_ci']:>13,.2f}"
        )

    # --- Historical context (last 6 months for comparison)
    print("\n  Last 6 Months (Actual) vs Trend")
    print("  " + "-" * 40)
    hist_rows = combined_df[combined_df["is_forecast"] == False].tail(6)
    print(f"  {'Month':<12} {'Actual Revenue':>16}")
    print("  " + "-" * 30)
    for _, row in hist_rows.iterrows():
        month_str = row["month"].strftime("%Y-%m")
        print(f"  {month_str:<12} ${row['revenue']:>15,.2f}")

    # --- Combined DataFrame structure
    print(f"\n  Combined DataFrame shape : {combined_df.shape}")
    print(f"  Historical rows          : {(combined_df['is_forecast'] == False).sum()}")
    print(f"  Forecast rows            : {(combined_df['is_forecast'] == True).sum()}")
    print(f"  Columns                  : {list(combined_df.columns)}")

    print("\n" + "=" * 55)
    print("   FORECASTING COMPLETE")
    print("=" * 55)
    print("\n  Module 4 complete. Run ai_insights.py next.\n")