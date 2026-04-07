"""
app.py
------
Module 5 — Interactive Analytics Dashboard

Purpose:
    The final layer of the project. Combines all previous modules into
    a multi-page interactive dashboard that a business user can read
    and act on without any coding knowledge.

    This file is purely a presentation layer. It calls analytics.py and
    forecasting.py for all data — it never queries the database directly.

Pages:
    1. Overview      — KPI cards + revenue trend + forecast chart
    2. Products      — Top products, return rates, category breakdown
    3. Customers     — Segmentation, LTV table, payment AOV
    4. Regional      — Revenue by region, performance table
    5. Forecasting   — Detailed forecast view with model stats

How to run:
    streamlit run app.py

Requirements:
    pip install streamlit plotly pandas

Author: E-Commerce Dashboard Project
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os

# =============================================================================
# PAGE CONFIGURATION
# Must be the very first Streamlit call in the file.
# =============================================================================

st.set_page_config(
    page_title  = "E-Commerce Analytics",
    page_icon   = "📊",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)


# =============================================================================
# CUSTOM CSS — Professional dark-accented theme
# Injects styling that Streamlit's default theme doesn't provide.
# =============================================================================

st.markdown("""
<style>
    /* Main background */
    .main { background-color: #0f1117; }

    /* KPI metric cards */
    .kpi-card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border: 1px solid #2d3250;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .kpi-label {
        font-size: 13px;
        color: #8b92a5;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 32px;
        font-weight: 700;
        color: #e8eaf6;
        margin: 4px 0;
    }
    .kpi-icon {
        font-size: 22px;
        margin-bottom: 6px;
    }

    /* Section headers */
    .section-header {
        font-size: 18px;
        font-weight: 600;
        color: #c5cae9;
        padding: 8px 0 4px 0;
        border-bottom: 2px solid #3d4270;
        margin-bottom: 16px;
    }

    /* Sidebar styling */
    .css-1d391kg { background-color: #1a1d2e; }

    /* Alert / info boxes */
    .info-box {
        background-color: #1e2130;
        border-left: 4px solid #5c6bc0;
        border-radius: 4px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #c5cae9;
        font-size: 14px;
    }

    /* Table styling */
    .dataframe { font-size: 13px; }

    /* Hide Streamlit default footer */
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CACHED DATA LOADERS
# st.cache_data means the function only runs ONCE per session.
# Every page re-render reuses the cached result — no repeated DB queries.
# =============================================================================

@st.cache_data
def load_kpi():
    """Load KPI summary — cached for the session."""
    from analytics import get_kpi_summary
    return get_kpi_summary()


@st.cache_data
def load_top_products():
    """Load top 10 products by revenue — cached."""
    from analytics import get_top_products
    return get_top_products()


@st.cache_data
def load_monthly_revenue():
    """Load monthly revenue trend — cached."""
    from analytics import get_monthly_revenue
    return get_monthly_revenue()


@st.cache_data
def load_revenue_by_region():
    """Load revenue by region — cached."""
    from analytics import get_revenue_by_region
    return get_revenue_by_region()


@st.cache_data
def load_return_by_category():
    """Load return rate by category — cached."""
    from analytics import get_return_rate_by_category
    return get_return_rate_by_category()


@st.cache_data
def load_customer_segments():
    """Load customer segmentation — cached."""
    from analytics import get_customer_segments
    return get_customer_segments()


@st.cache_data
def load_aov_by_payment():
    """Load average order value by payment method — cached."""
    from analytics import get_avg_order_value_by_payment
    return get_avg_order_value_by_payment()


@st.cache_data
def load_mom_growth():
    """Load month-over-month revenue growth — cached."""
    from analytics import get_mom_growth
    return get_mom_growth()


@st.cache_data
def load_top_customers():
    """Load top 5 customers by lifetime value — cached."""
    from analytics import get_top_customers
    return get_top_customers()


@st.cache_data
def load_forecast():
    """Load revenue forecast — cached."""
    from forecasting import get_forecast
    return get_forecast()


# =============================================================================
# COLOUR PALETTE
# Consistent colours used across all charts.
# =============================================================================

COLORS = {
    "primary":    "#5c6bc0",   # indigo — main accent
    "secondary":  "#26c6da",   # cyan — secondary accent
    "success":    "#66bb6a",   # green — positive values
    "warning":    "#ffa726",   # amber — caution values
    "danger":     "#ef5350",   # red — alerts / high returns
    "forecast":   "#ab47bc",   # purple — forecast line
    "ci_fill":    "rgba(171, 71, 188, 0.15)",  # purple transparent CI band
    "chart_bg":   "#1a1d2e",
    "grid":       "#2d3250",
    "text":       "#c5cae9",
}

# Category colour map — consistent across all charts
CATEGORY_COLORS = {
    "Electronics":    "#5c6bc0",
    "Clothing":       "#26c6da",
    "Home & Kitchen": "#66bb6a",
    "Books":          "#ffa726",
    "Sports":         "#ef5350",
}


# =============================================================================
# CHART HELPER — Consistent Plotly layout applied to every figure
# =============================================================================

def apply_dark_theme(fig: go.Figure, title: str = "") -> go.Figure:
    """
    Apply the dark professional theme to any Plotly figure.

    Args:
        fig:   Plotly figure object
        title: Chart title string

    Returns:
        Figure with dark theme applied
    """
    fig.update_layout(
        title           = dict(text=title, font=dict(color=COLORS["text"], size=15)),
        paper_bgcolor   = COLORS["chart_bg"],
        plot_bgcolor    = COLORS["chart_bg"],
        font            = dict(color=COLORS["text"], size=12),
        margin          = dict(l=40, r=20, t=50, b=40),
        legend          = dict(
            bgcolor     = COLORS["chart_bg"],
            bordercolor = COLORS["grid"],
            borderwidth = 1,
        ),
        xaxis = dict(
            gridcolor   = COLORS["grid"],
            linecolor   = COLORS["grid"],
            tickcolor   = COLORS["grid"],
        ),
        yaxis = dict(
            gridcolor   = COLORS["grid"],
            linecolor   = COLORS["grid"],
            tickcolor   = COLORS["grid"],
        ),
    )
    return fig


# =============================================================================
# KPI CARD RENDERER
# Renders a single styled metric card using HTML.
# =============================================================================

def render_kpi_card(icon: str, label: str, value: str, col) -> None:
    """
    Render a KPI metric card in the given Streamlit column.

    Args:
        icon:  Emoji icon e.g. '💰'
        label: Metric label e.g. 'Total Revenue'
        value: Formatted value string e.g. '$1,227,986'
        col:   Streamlit column object to render into
    """
    col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
    """, unsafe_allow_html=True)


# =============================================================================
# DATABASE GUARD
# Check the DB exists before rendering anything — shows a helpful error
# instead of a cryptic Python traceback if the user hasn't run the pipeline.
# =============================================================================

def check_database() -> bool:
    """
    Verify ecommerce.db exists. If not, show setup instructions.

    Returns:
        True if DB exists, False otherwise
    """
    if not os.path.exists("ecommerce.db"):
        st.error("⚠️ **Database not found** — `ecommerce.db` is missing.")
        st.markdown("""
        **Run the pipeline first:**
        ```bash
        python generate_data.py   # Step 1 — create raw dataset
        python data_cleaner.py    # Step 2 — clean and load into DB
        streamlit run app.py      # Step 3 — launch dashboard
        ```
        """)
        return False
    return True


# =============================================================================
# PAGE 1 — OVERVIEW
# KPI cards + Monthly revenue trend + Forecast preview
# =============================================================================

def page_overview():
    """Render the Overview page — KPIs, revenue trend, forecast."""

    st.markdown("## 📊 Business Overview")
    st.markdown("Key performance indicators and revenue trends at a glance.")
    st.markdown("---")

    # --- KPI Cards Row
    kpi = load_kpi()

    st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    render_kpi_card("💰", "Total Revenue",
                    f"${kpi['total_revenue']:,.0f}", c1)
    render_kpi_card("📦", "Total Orders",
                    f"{kpi['total_orders']:,}", c2)
    render_kpi_card("🛒", "Avg Order Value",
                    f"${kpi['avg_order_value']:,.2f}", c3)
    render_kpi_card("↩️", "Return Rate",
                    f"{kpi['return_rate_pct']}%", c4)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Monthly Revenue Trend
    st.markdown('<div class="section-header">Monthly Revenue Trend</div>',
                unsafe_allow_html=True)

    df_monthly = load_monthly_revenue()

    fig_trend = go.Figure()

    fig_trend.add_trace(go.Scatter(
        x    = df_monthly["month"],
        y    = df_monthly["revenue"],
        mode = "lines+markers",
        name = "Monthly Revenue",
        line = dict(color=COLORS["primary"], width=2.5),
        marker = dict(size=5, color=COLORS["primary"]),
        hovertemplate = "<b>%{x|%b %Y}</b><br>Revenue: $%{y:,.2f}<extra></extra>"
    ))

    apply_dark_theme(fig_trend, "Monthly Revenue (2023–2024)")
    fig_trend.update_layout(height=350, showlegend=False)
    fig_trend.update_yaxes(tickprefix="$", tickformat=",.0f")

    st.plotly_chart(fig_trend, use_container_width=True)

    # --- Forecast Preview
    st.markdown('<div class="section-header">Revenue Forecast — Next 3 Months</div>',
                unsafe_allow_html=True)

    combined_df, forecast_df, summary = load_forecast()

    fig_fc = go.Figure()

    # Historical line
    hist = combined_df[combined_df["is_forecast"] == False]
    fig_fc.add_trace(go.Scatter(
        x    = hist["month"],
        y    = hist["revenue"],
        mode = "lines",
        name = "Actual Revenue",
        line = dict(color=COLORS["primary"], width=2),
        hovertemplate = "<b>%{x|%b %Y}</b><br>Actual: $%{y:,.2f}<extra></extra>"
    ))

    # Confidence interval shaded band
    fcast = combined_df[combined_df["is_forecast"] == True]
    fig_fc.add_trace(go.Scatter(
        x    = pd.concat([fcast["month"], fcast["month"].iloc[::-1]]),
        y    = pd.concat([fcast["upper_ci"], fcast["lower_ci"].iloc[::-1]]),
        fill = "toself",
        fillcolor = COLORS["ci_fill"],
        line = dict(color="rgba(0,0,0,0)"),
        name = "95% Confidence Band",
        hoverinfo = "skip"
    ))

    # Forecast line
    fig_fc.add_trace(go.Scatter(
        x    = fcast["month"],
        y    = fcast["forecast"],
        mode = "lines+markers",
        name = "Forecast",
        line = dict(color=COLORS["forecast"], width=2.5, dash="dash"),
        marker = dict(size=7, color=COLORS["forecast"], symbol="diamond"),
        hovertemplate = "<b>%{x|%b %Y}</b><br>Forecast: $%{y:,.2f}<extra></extra>"
    ))

    apply_dark_theme(fig_fc, "Revenue Forecast with 95% Confidence Interval")
    fig_fc.update_layout(height=370)
    fig_fc.update_yaxes(tickprefix="$", tickformat=",.0f")

    st.plotly_chart(fig_fc, use_container_width=True)

    # Small forecast summary below chart
    fc1, fc2, fc3 = st.columns(3)
    for col, (_, row) in zip([fc1, fc2, fc3], forecast_df.iterrows()):
        month_label = row["month"].strftime("%B %Y")
        col.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{month_label}</div>
                <div class="kpi-value">${row['forecast']:,.0f}</div>
                <div style="color:#8b92a5;font-size:12px;">
                    ${row['lower_ci']:,.0f} – ${row['upper_ci']:,.0f}
                </div>
            </div>
        """, unsafe_allow_html=True)


# =============================================================================
# PAGE 2 — PRODUCTS & CATEGORIES
# Top products bar chart, return rate by category, category revenue donut
# =============================================================================

def page_products():
    """Render the Products & Categories page."""

    st.markdown("## 🛍️ Products & Categories")
    st.markdown("Revenue performance and return analysis by product and category.")
    st.markdown("---")

    # --- Top 10 Products Bar Chart
    st.markdown('<div class="section-header">Top 10 Products by Revenue</div>',
                unsafe_allow_html=True)

    df_products = load_top_products()

    # Map each product to its category colour
    bar_colors = [
        CATEGORY_COLORS.get(cat, COLORS["primary"])
        for cat in df_products["category"]
    ]

    fig_products = go.Figure(go.Bar(
        x             = df_products["revenue"],
        y             = df_products["product_name"],
        orientation   = "h",                 # horizontal bars
        marker_color  = bar_colors,
        customdata    = df_products[["category", "total_orders"]],
        hovertemplate = (
            "<b>%{y}</b><br>"
            "Category: %{customdata[0]}<br>"
            "Revenue: $%{x:,.2f}<br>"
            "Orders: %{customdata[1]}<extra></extra>"
        )
    ))

    apply_dark_theme(fig_products, "Top 10 Products by Revenue")
    fig_products.update_layout(height=420, yaxis=dict(autorange="reversed"))
    fig_products.update_xaxes(tickprefix="$", tickformat=",.0f")

    st.plotly_chart(fig_products, use_container_width=True)

    # --- Return Rate + Category Donut side by side
    col_left, col_right = st.columns(2)

    # Return Rate by Category
    with col_left:
        st.markdown('<div class="section-header">Return Rate by Category</div>',
                    unsafe_allow_html=True)

        df_returns = load_return_by_category()

        # Colour bars red if above 10% threshold (industry warning level)
        return_colors = [
            COLORS["danger"] if r > 10 else COLORS["primary"]
            for r in df_returns["return_rate_pct"]
        ]

        fig_returns = go.Figure(go.Bar(
            x            = df_returns["category"],
            y            = df_returns["return_rate_pct"],
            marker_color = return_colors,
            customdata   = df_returns[["returned_orders", "total_orders"]],
            hovertemplate = (
                "<b>%{x}</b><br>"
                "Return Rate: %{y:.1f}%<br>"
                "Returned: %{customdata[0]}<br>"
                "Total Orders: %{customdata[1]}<extra></extra>"
            )
        ))

        # 10% threshold red line — industry benchmark
        fig_returns.add_hline(
            y           = 10,
            line_dash   = "dash",
            line_color  = COLORS["danger"],
            annotation_text = "10% threshold",
            annotation_font_color = COLORS["danger"]
        )

        apply_dark_theme(fig_returns, "Return Rate by Category (%)")
        fig_returns.update_layout(height=350, showlegend=False)
        fig_returns.update_yaxes(ticksuffix="%")

        st.plotly_chart(fig_returns, use_container_width=True)

    # Category Revenue Donut
    with col_right:
        st.markdown('<div class="section-header">Revenue Share by Category</div>',
                    unsafe_allow_html=True)

        # Aggregate top products by category for the donut
        df_cat_rev = (
            df_products.groupby("category")["revenue"]
            .sum()
            .reset_index()
            .sort_values("revenue", ascending=False)
        )

        fig_donut = go.Figure(go.Pie(
            labels    = df_cat_rev["category"],
            values    = df_cat_rev["revenue"],
            hole      = 0.55,
            marker    = dict(colors=[
                CATEGORY_COLORS.get(c, COLORS["primary"])
                for c in df_cat_rev["category"]
            ]),
            textinfo  = "label+percent",
            hovertemplate = (
                "<b>%{label}</b><br>"
                "Revenue: $%{value:,.2f}<br>"
                "Share: %{percent}<extra></extra>"
            )
        ))

        apply_dark_theme(fig_donut, "Category Revenue Distribution")
        fig_donut.update_layout(height=350)

        st.plotly_chart(fig_donut, use_container_width=True)

    # --- Return Rate Info Box
    st.markdown("""
        <div class="info-box">
            ℹ️ <strong>Industry benchmark:</strong> A return rate above 10%
            typically signals product quality issues, misleading descriptions,
            or sizing problems (especially in Clothing). Bars shown in red
            exceed this threshold.
        </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGE 3 — CUSTOMER ANALYSIS
# Segmentation pie, top customers table, AOV by payment method
# =============================================================================

def page_customers():
    """Render the Customer Analysis page."""

    st.markdown("## 👥 Customer Analysis")
    st.markdown("Customer loyalty segmentation, lifetime value, and payment behaviour.")
    st.markdown("---")

    col_left, col_right = st.columns(2)

    # --- Customer Segmentation Pie
    with col_left:
        st.markdown('<div class="section-header">Customer Loyalty Segments</div>',
                    unsafe_allow_html=True)

        df_segments = load_customer_segments()

        seg_colors = {
            "One-time":   COLORS["danger"],
            "Occasional": COLORS["warning"],
            "Loyal":      COLORS["success"],
        }

        fig_seg = go.Figure(go.Pie(
            labels    = df_segments["segment"],
            values    = df_segments["customer_count"],
            hole      = 0.5,
            marker    = dict(colors=[
                seg_colors.get(s, COLORS["primary"])
                for s in df_segments["segment"]
            ]),
            textinfo  = "label+percent+value",
            hovertemplate = (
                "<b>%{label}</b><br>"
                "Customers: %{value}<br>"
                "Share: %{percent}<extra></extra>"
            )
        ))

        apply_dark_theme(fig_seg, "Customer Segments")
        fig_seg.update_layout(height=380)

        st.plotly_chart(fig_seg, use_container_width=True)

        # Segment definitions
        st.markdown("""
            <div class="info-box">
                <strong>Segment rules:</strong><br>
                🔴 <strong>One-time</strong>  — 1 order only<br>
                🟡 <strong>Occasional</strong> — 2–4 orders<br>
                🟢 <strong>Loyal</strong>      — 5+ orders
            </div>
        """, unsafe_allow_html=True)

    # --- AOV by Payment Method
    with col_right:
        st.markdown('<div class="section-header">Avg Order Value by Payment Method</div>',
                    unsafe_allow_html=True)

        df_payment = load_aov_by_payment()

        fig_pay = go.Figure(go.Bar(
            x            = df_payment["avg_order_value"],
            y            = df_payment["payment_method"],
            orientation  = "h",
            marker_color = COLORS["secondary"],
            customdata   = df_payment[["total_orders", "total_revenue"]],
            hovertemplate = (
                "<b>%{y}</b><br>"
                "Avg Order Value: $%{x:,.2f}<br>"
                "Total Orders: %{customdata[0]:,}<br>"
                "Total Revenue: $%{customdata[1]:,.2f}<extra></extra>"
            )
        ))

        apply_dark_theme(fig_pay, "Average Order Value by Payment Method")
        fig_pay.update_layout(height=380, yaxis=dict(autorange="reversed"))
        fig_pay.update_xaxes(tickprefix="$", tickformat=",.0f")

        st.plotly_chart(fig_pay, use_container_width=True)

    # --- Top 5 Customers Table
    st.markdown('<div class="section-header">Top 5 Customers by Lifetime Value</div>',
                unsafe_allow_html=True)

    df_top_customers = load_top_customers()

    # Format columns for display
    display_customers = df_top_customers.copy()
    display_customers["lifetime_value"]  = display_customers["lifetime_value"].apply(
        lambda x: f"${x:,.2f}"
    )
    display_customers["avg_order_value"] = display_customers["avg_order_value"].apply(
        lambda x: f"${x:,.2f}"
    )
    display_customers.columns = [
        "Customer ID", "Customer Name", "Lifetime Value",
        "Total Orders", "Avg Order Value"
    ]
    display_customers = display_customers.drop(columns=["Customer ID"])

    st.dataframe(
        display_customers,
        use_container_width = True,
        hide_index          = True
    )

    # --- MoM Growth Bar
    st.markdown('<div class="section-header">Month-over-Month Revenue Growth (%)</div>',
                unsafe_allow_html=True)

    df_mom = load_mom_growth().dropna(subset=["mom_growth_pct"])

    mom_colors = [
        COLORS["success"] if g >= 0 else COLORS["danger"]
        for g in df_mom["mom_growth_pct"]
    ]

    fig_mom = go.Figure(go.Bar(
        x            = df_mom["month"],
        y            = df_mom["mom_growth_pct"],
        marker_color = mom_colors,
        hovertemplate = (
            "<b>%{x|%b %Y}</b><br>"
            "MoM Growth: %{y:.2f}%<extra></extra>"
        )
    ))

    # Zero reference line
    fig_mom.add_hline(y=0, line_color=COLORS["grid"], line_width=1)

    apply_dark_theme(fig_mom, "Month-over-Month Revenue Growth (%)")
    fig_mom.update_layout(height=300, showlegend=False)
    fig_mom.update_yaxes(ticksuffix="%")

    st.plotly_chart(fig_mom, use_container_width=True)


# =============================================================================
# PAGE 4 — REGIONAL PERFORMANCE
# Revenue by region bar chart + detailed table
# =============================================================================

def page_regional():
    """Render the Regional Performance page."""

    st.markdown("## 🌍 Regional Performance")
    st.markdown("Revenue, orders, and return rates across all sales regions.")
    st.markdown("---")

    df_region = load_revenue_by_region()

    # --- Revenue by Region Bar Chart
    st.markdown('<div class="section-header">Revenue by Region</div>',
                unsafe_allow_html=True)

    region_colors = [
        COLORS["primary"], COLORS["secondary"], COLORS["success"],
        COLORS["warning"], COLORS["forecast"], COLORS["danger"]
    ]

    fig_region = go.Figure(go.Bar(
        x            = df_region["region"],
        y            = df_region["revenue"],
        marker_color = region_colors[:len(df_region)],
        customdata   = df_region[["total_orders", "avg_order_value", "return_rate_pct"]],
        hovertemplate = (
            "<b>%{x}</b><br>"
            "Revenue: $%{y:,.2f}<br>"
            "Orders: %{customdata[0]:,}<br>"
            "Avg Order Value: $%{customdata[1]:,.2f}<br>"
            "Return Rate: %{customdata[2]:.1f}%<extra></extra>"
        )
    ))

    apply_dark_theme(fig_region, "Revenue by Region")
    fig_region.update_layout(height=380, showlegend=False)
    fig_region.update_yaxes(tickprefix="$", tickformat=",.0f")

    st.plotly_chart(fig_region, use_container_width=True)

    # --- Region KPI mini-cards
    st.markdown('<div class="section-header">Region Breakdown</div>',
                unsafe_allow_html=True)

    # Show top 5 known regions (exclude Unknown)
    known_regions = df_region[df_region["region"] != "Unknown"].head(5)
    cols = st.columns(len(known_regions))

    for col, (_, row) in zip(cols, known_regions.iterrows()):
        col.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{row['region']}</div>
                <div class="kpi-value">${row['revenue']:,.0f}</div>
                <div style="color:#8b92a5;font-size:12px;">
                    {row['total_orders']:,} orders
                    · {row['return_rate_pct']:.1f}% returns
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Full Region Table
    st.markdown('<div class="section-header">Detailed Region Statistics</div>',
                unsafe_allow_html=True)

    display_region = df_region.copy()
    display_region["revenue"]          = display_region["revenue"].apply(lambda x: f"${x:,.2f}")
    display_region["avg_order_value"]  = display_region["avg_order_value"].apply(lambda x: f"${x:,.2f}")
    display_region["return_rate_pct"]  = display_region["return_rate_pct"].apply(lambda x: f"{x:.1f}%")
    display_region = display_region.drop(columns=["return_count"])
    display_region.columns = [
        "Region", "Revenue", "Total Orders", "Avg Order Value", "Return Rate"
    ]

    st.dataframe(display_region, use_container_width=True, hide_index=True)

    # Note about Unknown region
    st.markdown("""
        <div class="info-box">
            ℹ️ <strong>'Unknown' region</strong> represents orders where the region
            field was missing in the raw data (~8% null rate). These were filled
            during the cleaning step in Module 2 to avoid data loss.
        </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGE 5 — FORECASTING
# Detailed forecast chart + model stats + forecast table
# =============================================================================

def page_forecasting():
    """Render the Forecasting page — detailed forecast view."""

    st.markdown("## 🔮 Revenue Forecasting")
    st.markdown("Linear regression model trained on 24 months of historical data.")
    st.markdown("---")

    combined_df, forecast_df, summary = load_forecast()

    # --- Model Stats Cards
    st.markdown('<div class="section-header">Model Performance</div>',
                unsafe_allow_html=True)

    ms1, ms2, ms3, ms4 = st.columns(4)

    render_kpi_card("📈", "Trend",       summary["trend_label"],              ms1)
    render_kpi_card("💹", "Monthly Growth", f"${summary['monthly_growth']:,.2f}", ms2)
    render_kpi_card("🎯", "R² Score",    str(summary["r2_score"]),            ms3)
    render_kpi_card("⚡", "RMSE",         f"${summary['rmse']:,.2f}",          ms4)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Full Forecast Chart
    st.markdown('<div class="section-header">Historical Revenue + 3-Month Forecast</div>',
                unsafe_allow_html=True)

    fig = go.Figure()

    # Historical line
    hist = combined_df[combined_df["is_forecast"] == False]
    fig.add_trace(go.Scatter(
        x    = hist["month"],
        y    = hist["revenue"],
        mode = "lines+markers",
        name = "Actual Revenue",
        line = dict(color=COLORS["primary"], width=2.5),
        marker = dict(size=4),
        hovertemplate = "<b>%{x|%b %Y}</b><br>Revenue: $%{y:,.2f}<extra></extra>"
    ))

    # Confidence interval band
    fcast = combined_df[combined_df["is_forecast"] == True]
    fig.add_trace(go.Scatter(
        x         = pd.concat([fcast["month"], fcast["month"].iloc[::-1]]),
        y         = pd.concat([fcast["upper_ci"], fcast["lower_ci"].iloc[::-1]]),
        fill      = "toself",
        fillcolor = COLORS["ci_fill"],
        line      = dict(color="rgba(0,0,0,0)"),
        name      = "95% Confidence Interval",
        hoverinfo = "skip"
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x    = fcast["month"],
        y    = fcast["forecast"],
        mode = "lines+markers",
        name = "Forecast",
        line = dict(color=COLORS["forecast"], width=3, dash="dash"),
        marker = dict(size=10, color=COLORS["forecast"], symbol="diamond"),
        hovertemplate = (
            "<b>%{x|%b %Y}</b><br>"
            "Forecast: $%{y:,.2f}<extra></extra>"
        )
    ))

    # Vertical divider line between history and forecast
    last_hist_date = hist["month"].iloc[-1]
    x_millis = last_hist_date.value // 10**6  
    fig.add_vline(
        x           = x_millis,
        line_dash   = "dot",
        line_color  = COLORS["grid"],
        annotation_text = "Forecast →",
        annotation_font_color = COLORS["text"],
        annotation_position   = "top right"
    )

    apply_dark_theme(fig, "Revenue Forecast — Actuals + Prediction + 95% CI Band")
    fig.update_layout(height=500)
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")

    st.plotly_chart(fig, use_container_width=True)

    # --- Forecast Table + Explanation side by side
    col_table, col_explain = st.columns([1, 1])

    with col_table:
        st.markdown('<div class="section-header">Forecast Table</div>',
                    unsafe_allow_html=True)

        display_fc = forecast_df[["month", "forecast", "lower_ci", "upper_ci"]].copy()
        display_fc["month"]     = display_fc["month"].dt.strftime("%B %Y")
        display_fc["forecast"]  = display_fc["forecast"].apply(lambda x: f"${x:,.2f}")
        display_fc["lower_ci"]  = display_fc["lower_ci"].apply(lambda x: f"${x:,.2f}")
        display_fc["upper_ci"]  = display_fc["upper_ci"].apply(lambda x: f"${x:,.2f}")
        display_fc.columns      = ["Month", "Forecast", "Lower (95%)", "Upper (95%)"]

        st.dataframe(display_fc, use_container_width=True, hide_index=True)

    with col_explain:
        st.markdown('<div class="section-header">How This Works</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
            <div class="info-box">
                <strong>Model:</strong> Linear Regression<br><br>
                <strong>Formula:</strong><br>
                Revenue = {summary['monthly_growth']:,.2f} × month + {summary['intercept']:,.2f}<br><br>
                <strong>Interpretation:</strong><br>
                Revenue grows approximately
                <strong>${summary['monthly_growth']:,.2f}</strong> per month
                based on the 24-month trend.<br><br>
                <strong>R² = {summary['r2_score']}</strong> — the model explains
                {summary['r2_score']*100:.1f}% of revenue variance.<br><br>
                <strong>RMSE = ${summary['rmse']:,.2f}</strong> — typical
                prediction error per month.<br><br>
                <strong>Confidence Interval:</strong> 95% — there is a 95%
                probability that actual revenue will fall within the shaded band.
            </div>
        """, unsafe_allow_html=True)


# =============================================================================
# PAGE 6 — AI INSIGHTS (Gemini)
# Executive summary, anomaly detection, and natural language Q&A
# =============================================================================

def page_ai_insights():
    """
    Render the AI Insights page powered by Google Gemini.

    Three sections:
        1. Executive Summary  — auto-generated on page load
        2. Anomaly Detection  — flagged patterns worth investigating
        3. Ask the Data       — user types a question, Gemini answers

    The page degrades gracefully if GEMINI_API_KEY is not set —
    shows setup instructions instead of crashing.
    """
    from ai_insights import (
        collect_metrics_for_ai,
        generate_dashboard_summary,
        detect_anomalies,
        answer_question,
        GEMINI_API_KEY,
    )

    st.markdown("## 🤖 AI Insights")
    st.markdown("Powered by Google Gemini — natural language analysis of your data.")
    st.markdown("---")

    # --- API key status banner
    if not GEMINI_API_KEY:
        st.warning(
            "⚠️ **Gemini API key not found.** "
            "Add `GEMINI_API_KEY=your_key` to a `.env` file and restart. "
            "Get a free key at https://aistudio.google.com/app/apikey"
        )
        st.markdown("""
        <div class="info-box">
            <strong>Setup steps:</strong><br>
            1. Visit <a href="https://aistudio.google.com/app/apikey"
               style="color:#5c6bc0;">aistudio.google.com/app/apikey</a>
               and create a free key<br>
            2. Create a file called <code>.env</code> in your project root<br>
            3. Add one line: <code>GEMINI_API_KEY=your_key_here</code><br>
            4. Run <code>pip install python-dotenv</code> if not installed<br>
            5. Restart: <code>streamlit run app.py</code>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.info(
            "💡 All other dashboard pages (Overview, Products, Customers, "
            "Regional, Forecasting) work fully without this key."
        )
        return   # stop rendering the rest of the page

    # --- Load metrics once — reused for all three AI features
    with st.spinner("Loading business data for AI analysis..."):
        try:
            metrics = collect_metrics_for_ai()
        except Exception as e:
            st.error(f"Failed to load metrics: {e}")
            return

    # =========================================================================
    # SECTION 1 — EXECUTIVE SUMMARY
    # =========================================================================
    st.markdown('<div class="section-header">📋 Executive Summary</div>',
                unsafe_allow_html=True)

    st.markdown(
        "*Auto-generated overview of your business performance — "
        "as if written by a senior analyst.*"
    )

    with st.spinner("Gemini is writing your executive summary..."):
        summary_text = generate_dashboard_summary(metrics)

    # Render the summary in a styled card
    st.markdown(f"""
        <div class="info-box" style="border-left-color: #5c6bc0; font-size: 15px;
             line-height: 1.7; padding: 20px 24px;">
            {summary_text.replace(chr(10), '<br>')}
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================================================================
    # SECTION 2 — ANOMALY DETECTION
    # =========================================================================
    st.markdown('<div class="section-header">⚠️ Anomaly Detection</div>',
                unsafe_allow_html=True)

    st.markdown(
        "*Patterns that deserve management attention — flagged automatically.*"
    )

    with st.spinner("Scanning for anomalies..."):
        anomaly_text = detect_anomalies(metrics)

    # Parse bullet points and render as individual alert cards
    lines = [
        line.strip()
        for line in anomaly_text.split("\n")
        if line.strip().startswith("•") or line.strip().startswith("-")
    ]

    if lines:
        for line in lines:
            # Clean the bullet character
            clean = line.lstrip("•-").strip()

            # Choose card colour based on content keywords
            if any(w in clean.lower() for w in ["high", "drop", "decline", "concern", "warning", "above"]):
                border_color = COLORS["danger"]
                icon = "🔴"
            elif any(w in clean.lower() for w in ["growth", "strong", "healthy", "positive"]):
                border_color = COLORS["success"]
                icon = "🟢"
            else:
                border_color = COLORS["warning"]
                icon = "🟡"

            st.markdown(f"""
                <div class="info-box" style="border-left-color: {border_color};
                     margin-bottom: 8px;">
                    {icon} {clean}
                </div>
            """, unsafe_allow_html=True)
    else:
        # Fallback if Gemini didn't return bullet points — show raw text
        st.markdown(f"""
            <div class="info-box">{anomaly_text.replace(chr(10), '<br>')}</div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================================================================
    # SECTION 3 — NATURAL LANGUAGE Q&A
    # =========================================================================
    st.markdown('<div class="section-header">💬 Ask the Data</div>',
                unsafe_allow_html=True)

    st.markdown(
        "*Type any business question about your data and Gemini will answer "
        "using the actual metrics.*"
    )

    # Suggested question chips to help users get started
    st.markdown("**Suggested questions:**")
    chip_cols = st.columns(3)

    suggestions = [
        "Which region should we invest more in?",
        "Why might our return rate be high?",
        "Which customer segment should we target?",
        "What does the forecast say about Q1 2025?",
        "Which product category has the most growth potential?",
        "How is our month-over-month performance trending?",
    ]

    # Clicking a chip pre-fills the text input
    if "ai_question" not in st.session_state:
        st.session_state["ai_question"] = ""

    for i, (col, suggestion) in enumerate(zip(chip_cols * 2, suggestions)):
        if col.button(suggestion, key=f"chip_{i}", use_container_width=True):
            st.session_state["ai_question"] = suggestion

    st.markdown("<br>", unsafe_allow_html=True)

    # Text input box — uses session state so chip clicks pre-fill it
    question = st.text_input(
        label       = "Your question",
        value       = st.session_state["ai_question"],
        placeholder = "e.g. Which region has the most growth potential?",
        label_visibility = "collapsed",
        key         = "question_input"
    )

    ask_col, _ = st.columns([1, 4])
    ask_clicked = ask_col.button("🔍 Ask Gemini", type="primary", use_container_width=True)

    if ask_clicked and question.strip():
        with st.spinner("Gemini is thinking..."):
            answer = answer_question(question.strip(), metrics)

        # Display the question + answer together
        st.markdown(f"""
            <div class="info-box" style="border-left-color: {COLORS['secondary']};
                 margin-top: 16px;">
                <div style="color: #8b92a5; font-size: 12px; margin-bottom: 6px;">
                    YOUR QUESTION
                </div>
                <div style="font-size: 15px; margin-bottom: 14px; color: #e8eaf6;">
                    {question.strip()}
                </div>
                <div style="color: #8b92a5; font-size: 12px; margin-bottom: 6px;">
                    GEMINI'S ANSWER
                </div>
                <div style="font-size: 15px; line-height: 1.7; color: #c5cae9;">
                    {answer.replace(chr(10), '<br>')}
                </div>
            </div>
        """, unsafe_allow_html=True)

    elif ask_clicked and not question.strip():
        st.warning("Please type a question before clicking Ask Gemini.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="color:#8b92a5; font-size:12px; text-align:center;">'
        'AI answers are generated from your actual data. '
        'Always verify important business decisions independently.'
        '</div>',
        unsafe_allow_html=True
    )


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================

def render_sidebar() -> str:
    """
    Render the sidebar with navigation and project info.

    Returns:
        Selected page name string
    """
    with st.sidebar:
        st.markdown("## 📊 E-Commerce Analytics")
        st.markdown("*End-to-end analytics dashboard*")
        st.markdown("---")

        page = st.radio(
            "Navigate to",
            options = [
                "📊 Overview",
                "🛍️ Products & Categories",
                "👥 Customer Analysis",
                "🌍 Regional Performance",
                "🔮 Forecasting",
                "🤖 AI Insights",
            ],
            label_visibility = "collapsed"
        )

        st.markdown("---")
        st.markdown("**Pipeline**")
        st.markdown("""
        ```
        generate_data.py
              ↓
        data_cleaner.py
              ↓
        analytics.py
              ↓
        forecasting.py
              ↓
        ai_insights.py
              ↓
        app.py  ← you are here
        ```
        """)

        st.markdown("---")
        st.caption("Built with Python · Streamlit · Plotly · SQLite")

    return page


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Main function — renders sidebar, checks DB, routes to correct page.
    """
    # Guard: make sure the database is present before rendering anything
    if not check_database():
        st.stop()

    # Render sidebar and get the selected page
    page = render_sidebar()

    # Route to the correct page function
    if page == "📊 Overview":
        page_overview()
    elif page == "🛍️ Products & Categories":
        page_products()
    elif page == "👥 Customer Analysis":
        page_customers()
    elif page == "🌍 Regional Performance":
        page_regional()
    elif page == "🔮 Forecasting":
        page_forecasting()
    elif page == "🤖 AI Insights":
        page_ai_insights()


if __name__ == "__main__":
    main()