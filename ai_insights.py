"""
ai_insights.py
--------------
Module 6 — AI-Powered Insights (Google Gemini)

Purpose:
    Use Google Gemini API (gemini-2.0-flash) to generate intelligent,
    plain-English insights from the analytics data computed in Module 3.

    Three capabilities:
        1. Executive summary   — 3-5 sentence business overview
        2. Anomaly detection   — bullet-point flags for unusual patterns
        3. Q&A                 — natural language answers about the data

    This module is completely optional. If GEMINI_API_KEY is missing
    or invalid, every function returns a graceful fallback message.
    The rest of the dashboard continues working normally.

How Gemini is called:
    Direct HTTP POST to https://api.anthropic.com/v1/messages
    (The Gemini REST API — no SDK needed, just the requests library)
    API key is loaded from .env file using python-dotenv.

Input:
    metrics_dict — a plain dictionary of key business metrics built
    from analytics.py and forecasting.py output. Passed as context
    inside the prompt so Gemini can reason about real data.

Output:
    Plain strings (summaries, bullet lists, Q&A answers)
    Ready to render directly in Streamlit.

Setup:
    1. Get free Gemini API key at https://aistudio.google.com/app/apikey
    2. Create .env file in project root: GEMINI_API_KEY=your_key_here
    3. pip install python-dotenv requests

Author: E-Commerce Dashboard Project
"""

import os
import json
import requests

# python-dotenv loads the .env file so GEMINI_API_KEY becomes available
# via os.getenv(). Fails silently if .env doesn't exist.
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

# Gemini REST API endpoint
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models"
    "/gemini-2.0-flash:generateContent"
)

# Load API key from environment (set via .env file)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Request timeout in seconds — prevents dashboard hanging if API is slow
REQUEST_TIMEOUT = 30

# Fallback messages shown when API key is missing
NO_KEY_MESSAGE = (
    "🔑 **Gemini API key not configured.**\n\n"
    "To enable AI insights:\n"
    "1. Get a free key at https://aistudio.google.com/app/apikey\n"
    "2. Create a `.env` file in the project root\n"
    "3. Add: `GEMINI_API_KEY=your_key_here`\n"
    "4. Restart the dashboard\n\n"
    "All other dashboard pages work without this key."
)


# =============================================================================
# CONTEXT BUILDER
# Converts raw analytics DataFrames + dicts → a clean text block
# that Gemini can understand and reason about.
# =============================================================================

def build_metrics_context(metrics_dict: dict) -> str:
    """
    Convert the analytics metrics dictionary into a structured text block.

    Gemini is a language model — it works with text, not DataFrames.
    This function serialises all key metrics into a readable format
    that forms the "data context" injected into every prompt.

    Args:
        metrics_dict: Dictionary produced by collect_metrics_for_ai()
                      Contains KPIs, top products, region data, etc.

    Returns:
        Multi-line string describing the business data clearly
    """
    kpi      = metrics_dict.get("kpi", {})
    products = metrics_dict.get("top_products", [])
    regions  = metrics_dict.get("regions", [])
    returns  = metrics_dict.get("returns", [])
    segments = metrics_dict.get("segments", [])
    forecast = metrics_dict.get("forecast", [])
    mom      = metrics_dict.get("recent_mom", [])
    trend    = metrics_dict.get("trend_label", "Unknown")
    growth   = metrics_dict.get("monthly_growth", 0)

    lines = [
        "=== E-COMMERCE BUSINESS DATA ===",
        "",
        "KEY PERFORMANCE INDICATORS:",
        f"  - Total Revenue    : ${kpi.get('total_revenue', 0):,.2f}",
        f"  - Total Orders     : {kpi.get('total_orders', 0):,}",
        f"  - Avg Order Value  : ${kpi.get('avg_order_value', 0):,.2f}",
        f"  - Return Rate      : {kpi.get('return_rate_pct', 0):.2f}%",
        "",
        "TOP 5 PRODUCTS BY REVENUE:",
    ]

    for p in products[:5]:
        lines.append(f"  - {p['product_name']}: ${p['revenue']:,.2f}")

    lines += [
        "",
        "REVENUE BY REGION:",
    ]
    for r in regions:
        if r["region"] != "Unknown":
            lines.append(
                f"  - {r['region']}: ${r['revenue']:,.2f} "
                f"({r['return_rate_pct']:.1f}% return rate)"
            )

    lines += [
        "",
        "RETURN RATE BY CATEGORY:",
    ]
    for r in returns:
        flag = " ⚠️ HIGH" if r["return_rate_pct"] > 10 else ""
        lines.append(
            f"  - {r['category']}: {r['return_rate_pct']:.1f}%{flag}"
        )

    lines += [
        "",
        "CUSTOMER SEGMENTS:",
    ]
    for s in segments:
        lines.append(
            f"  - {s['segment']}: {s['customer_count']} customers "
            f"({s['pct_of_total']:.1f}%)"
        )

    lines += [
        "",
        "REVENUE FORECAST (next 3 months):",
        f"  - Trend: {trend} (${growth:,.2f}/month growth rate)",
    ]
    for f in forecast:
        lines.append(
            f"  - {f['month']}: ${f['forecast']:,.2f} "
            f"(range ${f['lower_ci']:,.2f}–${f['upper_ci']:,.2f})"
        )

    lines += [
        "",
        "RECENT MONTH-OVER-MONTH GROWTH (last 3 months):",
    ]
    for m in mom:
        direction = "▲" if m["growth"] >= 0 else "▼"
        lines.append(f"  - {m['month']}: {direction} {m['growth']:+.2f}%")

    lines.append("")
    lines.append("=================================")

    return "\n".join(lines)


# =============================================================================
# GEMINI API CALLER
# Low-level function that sends a prompt to Gemini and returns the text.
# All three public functions use this internally.
# =============================================================================

def call_gemini(prompt: str) -> str:
    """
    Send a prompt to the Gemini API and return the response text.

    Handles all failure modes gracefully:
        - Missing API key    → returns setup instructions
        - Network error      → returns friendly error message
        - API error response → returns the error detail
        - Timeout            → returns timeout message

    Args:
        prompt: The full prompt string to send to Gemini

    Returns:
        Response text string, or a fallback error message
    """
    # Guard: no key → no API call
    if not GEMINI_API_KEY:
        return NO_KEY_MESSAGE

    headers = {"Content-Type": "application/json"}
    params  = {"key": GEMINI_API_KEY}

    # Gemini REST API request body
    body = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature":     0.4,   # lower = more factual, less creative
            "maxOutputTokens": 1024,  # enough for summaries and Q&A
            "topP":            0.8,
        }
    }

    try:
        response = requests.post(
            GEMINI_API_URL,
            headers = headers,
            params  = params,
            json    = body,
            timeout = REQUEST_TIMEOUT
        )

        # Check HTTP status
        if response.status_code != 200:
            return (
                f"⚠️ Gemini API error (HTTP {response.status_code}).\n"
                f"Detail: {response.text[:300]}"
            )

        # Parse response JSON
        data = response.json()

        # Extract the text from the nested response structure
        text = (
            data
            .get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )

        if not text:
            return "⚠️ Gemini returned an empty response. Please try again."

        return text.strip()

    except requests.exceptions.Timeout:
        return (
            f"⏱️ Request timed out after {REQUEST_TIMEOUT}s. "
            "Gemini API may be slow — please try again."
        )
    except requests.exceptions.ConnectionError:
        return (
            "🌐 Network error — could not reach Gemini API. "
            "Check your internet connection."
        )
    except Exception as e:
        return f"⚠️ Unexpected error: {str(e)}"


# =============================================================================
# PUBLIC FUNCTION 1 — EXECUTIVE SUMMARY
# Generates a 3-5 sentence overview of the business performance.
# =============================================================================

def generate_dashboard_summary(metrics_dict: dict) -> str:
    """
    Generate a 3-5 sentence executive summary of the business data.

    The summary is written as if by a senior business analyst presenting
    to a management team. It covers revenue performance, top sellers,
    regional highlights, and the revenue forecast outlook.

    Args:
        metrics_dict: Dictionary from collect_metrics_for_ai()

    Returns:
        3-5 sentence executive summary string
    """
    context = build_metrics_context(metrics_dict)

    prompt = f"""You are a senior business analyst presenting to the executive team.

Based on the following e-commerce data, write a concise executive summary of
3-5 sentences. Cover: overall revenue performance, top-selling products,
regional standout, and forward-looking forecast. Be specific — use the actual
numbers. Write in confident, professional business language.

{context}

Executive Summary:"""

    return call_gemini(prompt)


# =============================================================================
# PUBLIC FUNCTION 2 — ANOMALY DETECTION
# Scans the metrics and flags anything unusual or worth investigating.
# =============================================================================

def detect_anomalies(metrics_dict: dict) -> str:
    """
    Scan the business metrics and return bullet-point anomaly flags.

    Looks for patterns that are unusual or potentially problematic:
        - Return rates above industry threshold (>10%)
        - Large negative MoM revenue swings
        - Underperforming regions
        - Declining forecast trend

    Args:
        metrics_dict: Dictionary from collect_metrics_for_ai()

    Returns:
        Bullet-point string of anomalies found, or "No anomalies detected."
    """
    context = build_metrics_context(metrics_dict)

    prompt = f"""You are a data analyst performing a business health check.

Review the following e-commerce metrics and identify any anomalies, warning
signs, or patterns that deserve management attention.

Rules for your response:
- Return ONLY bullet points (start each with •)
- Each bullet = one specific finding with the actual number
- Flag return rates above 10% as concerning
- Flag any month with revenue drop > 10% MoM
- Flag underperforming regions compared to average
- If something looks healthy, note it positively
- Maximum 6 bullets
- Be specific and direct — no generic statements

{context}

Anomalies and Observations:"""

    return call_gemini(prompt)


# =============================================================================
# PUBLIC FUNCTION 3 — NATURAL LANGUAGE Q&A
# Answers a user's question about the data using Gemini.
# =============================================================================

def answer_question(question: str, metrics_dict: dict) -> str:
    """
    Answer a natural language question about the e-commerce data.

    The user types a question in plain English. This function injects
    the full metrics context + the question into a prompt and returns
    Gemini's answer.

    Example questions:
        "Which region should we invest more in?"
        "Why might our return rate be so high?"
        "What does the forecast tell us about Q1 2025?"
        "Which customer segment should we focus on?"

    Args:
        question:     The user's question string
        metrics_dict: Dictionary from collect_metrics_for_ai()

    Returns:
        Answer string in plain English
    """
    if not question or not question.strip():
        return "Please type a question to get an answer."

    context = build_metrics_context(metrics_dict)

    prompt = f"""You are an expert business analyst for an e-commerce company.

You have access to the following business data:

{context}

A stakeholder asks: "{question.strip()}"

Answer the question directly and specifically using the data provided.
Be concise (2-4 sentences). Use actual numbers from the data.
If the data doesn't contain enough information to answer fully, say so honestly.

Answer:"""

    return call_gemini(prompt)


# =============================================================================
# METRICS COLLECTOR
# Builds the metrics_dict that all three functions above expect.
# Called once by the dashboard page and reused for all three features.
# =============================================================================

def collect_metrics_for_ai() -> dict:
    """
    Collect all analytics data into one flat dictionary for AI context.

    Converts pandas DataFrames → plain Python lists/dicts so they can
    be easily serialised into prompt text by build_metrics_context().

    Returns:
        Dictionary with all key metrics in JSON-serialisable format
    """
    from analytics import (
        get_kpi_summary,
        get_top_products,
        get_revenue_by_region,
        get_return_rate_by_category,
        get_customer_segments,
        get_mom_growth,
    )
    from forecasting import get_forecast

    # KPI — already a dict
    kpi = get_kpi_summary()

    # Top products — convert to list of dicts
    df_products = get_top_products()
    top_products = df_products[["product_name", "revenue"]].to_dict("records")

    # Region — convert to list of dicts
    df_region = get_revenue_by_region()
    regions = df_region[["region", "revenue", "return_rate_pct"]].to_dict("records")

    # Return rate by category
    df_returns = get_return_rate_by_category()
    returns = df_returns[["category", "return_rate_pct"]].to_dict("records")

    # Customer segments
    df_segments = get_customer_segments()
    segments = df_segments[["segment", "customer_count", "pct_of_total"]].to_dict("records")

    # Forecast
    _, forecast_df, summary = get_forecast()
    forecast = []
    for _, row in forecast_df.iterrows():
        forecast.append({
            "month":     row["month"].strftime("%B %Y"),
            "forecast":  float(row["forecast"]),
            "lower_ci":  float(row["lower_ci"]),
            "upper_ci":  float(row["upper_ci"]),
        })

    # Recent MoM growth (last 3 months only — enough context for AI)
    df_mom = get_mom_growth().dropna(subset=["mom_growth_pct"]).tail(3)
    recent_mom = []
    for _, row in df_mom.iterrows():
        recent_mom.append({
            "month":  row["month"].strftime("%b %Y"),
            "growth": float(row["mom_growth_pct"]),
        })

    return {
        "kpi":            kpi,
        "top_products":   top_products,
        "regions":        regions,
        "returns":        returns,
        "segments":       segments,
        "forecast":       forecast,
        "recent_mom":     recent_mom,
        "trend_label":    summary["trend_label"],
        "monthly_growth": summary["monthly_growth"],
    }


# =============================================================================
# ENTRY POINT — Test the module standalone
# =============================================================================

if __name__ == "__main__":

    print("\n" + "=" * 55)
    print("   MODULE 6 — AI INSIGHTS (Gemini)")
    print("=" * 55)

    if not GEMINI_API_KEY:
        print("\n  ⚠️  GEMINI_API_KEY not found in environment.")
        print("  Create a .env file with: GEMINI_API_KEY=your_key_here")
        print("  Get a free key at: https://aistudio.google.com/app/apikey")
        print("\n  Testing with API key absent — fallback messages will show.\n")

    print("\n  Collecting metrics...")
    metrics = collect_metrics_for_ai()
    print(f"  Metrics collected: {list(metrics.keys())}")

    print("\n  [1/3] Generating executive summary...")
    summary = generate_dashboard_summary(metrics)
    print(f"\n  {summary}")

    print("\n  [2/3] Detecting anomalies...")
    anomalies = detect_anomalies(metrics)
    print(f"\n  {anomalies}")

    print("\n  [3/3] Answering sample question...")
    answer = answer_question(
        "Which product category should we prioritise for growth?",
        metrics
    )
    print(f"\n  {answer}")

    print("\n" + "=" * 55)
    print("   MODULE 6 COMPLETE")
    print("=" * 55)
    print("\n  Next step: run app.py with AI page enabled.\n")