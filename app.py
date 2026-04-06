"""
app.py
------
Module 5 — Interactive Analytics Dashboard (Redesigned)

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
# CUSTOM CSS — Refined monochrome editorial theme
# Clean, high-contrast, typographically driven design.
# =============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

    /* ── Global reset ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* ── App background ── */
    .stApp {
        background-color: #0a0b0f;
    }
    .main .block-container {
        padding: 2rem 2.5rem 3rem 2.5rem;
        max-width: 1400px;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #0e0f14;
        border-right: 1px solid #1c1e2a;
    }
    [data-testid="stSidebar"] .stRadio > label {
        color: #555975;
        font-size: 11px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    [data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
        font-size: 13.5px !important;
        color: #8b90a8 !important;
        padding: 4px 0;
    }

    /* ── KPI Cards ── */
    .kpi-card {
        background: #11121a;
        border: 1px solid #1c1e2a;
        border-radius: 16px;
        padding: 24px 28px;
        position: relative;
        overflow: hidden;
        transition: border-color 0.2s ease;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #4f5bd5, #8b5cf6);
        opacity: 0.6;
    }
    .kpi-card:hover {
        border-color: #2d3050;
    }
    .kpi-label {
        font-size: 11px;
        font-weight: 500;
        color: #555975;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 10px;
    }
    .kpi-value {
        font-size: 30px;
        font-weight: 600;
        color: #e8eaf6;
        font-variant-numeric: tabular-nums;
        letter-spacing: -0.5px;
        line-height: 1;
    }
    .kpi-sub {
        font-size: 12px;
        color: #3d4166;
        margin-top: 8px;
        font-family: 'DM Mono', monospace;
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 12px;
        font-weight: 500;
        color: #454870;
        text-transform: uppercase;
        letter-spacing: 2px;
        padding: 0 0 12px 0;
        margin: 4px 0 20px 0;
        border-bottom: 1px solid #1c1e2a;
    }

    /* ── Page title ── */
    .page-title {
        font-size: 28px;
        font-weight: 600;
        color: #e8eaf6;
        letter-spacing: -0.5px;
        margin: 0 0 6px 0;
        line-height: 1.2;
    }
    .page-sub {
        font-size: 14px;
        color: #454870;
        margin: 0 0 28px 0;
        font-weight: 400;
    }

    /* ── Divider ── */
    hr {
        border: none;
        border-top: 1px solid #1c1e2a;
        margin: 20px 0 28px 0;
    }

    /* ── Info / alert boxes ── */
    .info-box {
        background-color: #0e0f14;
        border: 1px solid #1c1e2a;
        border-left: 3px solid #4f5bd5;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 12px 0;
        color: #8b90a8;
        font-size: 13px;
        line-height: 1.6;
    }
    .info-box strong {
        color: #c5c9e0;
        font-weight: 500;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        border: 1px solid #1c1e2a;
        border-radius: 12px;
        overflow: hidden;
    }

    /* ── Forecast mini cards ── */
    .fc-card {
        background: #11121a;
        border: 1px solid #1c1e2a;
        border-radius: 14px;
        padding: 18px 22px;
        text-align: left;
    }
    .fc-month {
        font-size: 11px;
        font-weight: 500;
        color: #454870;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
    }
    .fc-value {
        font-size: 24px;
        font-weight: 600;
        color: #c8aff0;
        font-variant-numeric: tabular-nums;
    }
    .fc-range {
        font-size: 11px;
        color: #3d4166;
        margin-top: 6px;
        font-family: 'DM Mono', monospace;
    }

    /* ── Region mini cards ── */
    .region-card {
        background: #11121a;
        border: 1px solid #1c1e2a;
        border-radius: 14px;
        padding: 18px 20px;
    }
    .region-name {
        font-size: 11px;
        font-weight: 500;
        color: #454870;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 10px;
    }
    .region-value {
        font-size: 22px;
        font-weight: 600;
        color: #e8eaf6;
        font-variant-numeric: tabular-nums;
    }
    .region-meta {
        font-size: 11px;
        color: #3d4166;
        margin-top: 6px;
        font-family: 'DM Mono', monospace;
    }

    /* ── Segment info ── */
    .seg-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        margin-right: 6px;
        vertical-align: middle;
    }

    /* ── Hide Streamlit default footer / header cruft ── */
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; }

    /* ── Sidebar pipeline ── */
    .pipeline-step {
        font-family: 'DM Mono', monospace;
        font-size: 12px;
        color: #454870;
        padding: 3px 0;
    }
    .pipeline-step.active {
        color: #8b90a8;
        font-weight: 500;
    }
    .pipeline-arrow {
        font-size: 11px;
        color: #2a2d40;
        padding-left: 14px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# COLOUR PALETTE
# =============================================================================

COLORS = {
    "primary":    "#4f5bd5",
    "secondary":  "#22d3ee",
    "success":    "#34d399",
    "warning":    "#fbbf24",
    "danger":     "#f87171",
    "forecast":   "#c084fc",
    "ci_fill":    "rgba(192, 132, 252, 0.08)",
    "chart_bg":   "#0e0f14",
    "grid":       "#1a1c27",
    "text":       "#8b90a8",
    "text_bright":"#c5c9e0",
}

CATEGORY_COLORS = {
    "Electronics":    "#4f5bd5",
    "Clothing":       "#22d3ee",
    "Home & Kitchen": "#34d399",
    "Books":          "#fbbf24",
    "Sports":         "#f87171",
}


# =============================================================================
# CHART HELPER
# =============================================================================

def apply_dark_theme(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(
        title = dict(
            text  = title,
            font  = dict(color=COLORS["text_bright"], size=14, family="DM Sans"),
            x     = 0,
            xanchor = "left",
            pad   = dict(l=0, b=12),
        ),
        paper_bgcolor = COLORS["chart_bg"],
        plot_bgcolor  = COLORS["chart_bg"],
        font          = dict(color=COLORS["text"], size=12, family="DM Sans"),
        margin        = dict(l=48, r=20, t=52, b=40),
        legend        = dict(
            bgcolor     = "rgba(0,0,0,0)",
            bordercolor = COLORS["grid"],
            borderwidth = 0,
            font        = dict(size=12),
        ),
        xaxis = dict(
            gridcolor    = COLORS["grid"],
            linecolor    = COLORS["grid"],
            tickcolor    = "rgba(0,0,0,0)",
            showgrid     = True,
            zeroline     = False,
            tickfont     = dict(size=11),
        ),
        yaxis = dict(
            gridcolor    = COLORS["grid"],
            linecolor    = "rgba(0,0,0,0)",
            tickcolor    = "rgba(0,0,0,0)",
            showgrid     = True,
            zeroline     = False,
            tickfont     = dict(size=11),
        ),
    )
    return fig


# =============================================================================
# KPI CARD RENDERER
# =============================================================================

def render_kpi_card(label: str, value: str, col, sub: str = "") -> None:
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {sub_html}
        </div>
    """, unsafe_allow_html=True)


# =============================================================================
# CACHED DATA LOADERS
# =============================================================================

@st.cache_data
def load_kpi():
    from analytics import get_kpi_summary
    return get_kpi_summary()

@st.cache_data
def load_top_products():
    from analytics import get_top_products
    return get_top_products()

@st.cache_data
def load_monthly_revenue():
    from analytics import get_monthly_revenue
    return get_monthly_revenue()

@st.cache_data
def load_revenue_by_region():
    from analytics import get_revenue_by_region
    return get_revenue_by_region()

@st.cache_data
def load_return_by_category():
    from analytics import get_return_rate_by_category
    return get_return_rate_by_category()

@st.cache_data
def load_customer_segments():
    from analytics import get_customer_segments
    return get_customer_segments()

@st.cache_data
def load_aov_by_payment():
    from analytics import get_avg_order_value_by_payment
    return get_avg_order_value_by_payment()

@st.cache_data
def load_mom_growth():
    from analytics import get_mom_growth
    return get_mom_growth()

@st.cache_data
def load_top_customers():
    from analytics import get_top_customers
    return get_top_customers()

@st.cache_data
def load_forecast():
    from forecasting import get_forecast
    return get_forecast()


# =============================================================================
# DATABASE GUARD
# =============================================================================

def check_database() -> bool:
    if not os.path.exists("ecommerce.db"):
        st.error("⚠️ **Database not found** — `ecommerce.db` is missing.")
        st.markdown("""
        **Run the pipeline first:**
        ```bash
        python generate_data.py
        python data_cleaner.py
        streamlit run app.py
        ```
        """)
        return False
    return True


# =============================================================================
# PAGE 1 — OVERVIEW
# =============================================================================

def page_overview():
    st.markdown('<div class="page-title">Business Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Key performance indicators and revenue trends at a glance.</div>', unsafe_allow_html=True)

    kpi = load_kpi()

    st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    render_kpi_card("Total Revenue",    f"${kpi['total_revenue']:,.0f}",         c1)
    render_kpi_card("Total Orders",     f"{kpi['total_orders']:,}",              c2)
    render_kpi_card("Avg Order Value",  f"${kpi['avg_order_value']:,.2f}",       c3)
    render_kpi_card("Return Rate",      f"{kpi['return_rate_pct']}%",           c4)

    st.markdown("<br>", unsafe_allow_html=True)

    # Monthly Revenue Trend
    st.markdown('<div class="section-header">Monthly Revenue Trend</div>', unsafe_allow_html=True)

    df_monthly = load_monthly_revenue()

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x    = df_monthly["month"],
        y    = df_monthly["revenue"],
        mode = "lines",
        name = "Revenue",
        fill = "tozeroy",
        fillcolor = "rgba(79, 91, 213, 0.06)",
        line = dict(color=COLORS["primary"], width=2),
        hovertemplate = "<b>%{x|%b %Y}</b><br>Revenue: $%{y:,.0f}<extra></extra>"
    ))

    apply_dark_theme(fig_trend, "Monthly Revenue  ·  2023–2024")
    fig_trend.update_layout(height=300, showlegend=False)
    fig_trend.update_yaxes(tickprefix="$", tickformat=",.0f")
    st.plotly_chart(fig_trend, use_container_width=True)

    # Forecast Preview
    st.markdown('<div class="section-header">Revenue Forecast — Next 3 Months</div>', unsafe_allow_html=True)

    combined_df, forecast_df, summary = load_forecast()
    hist  = combined_df[combined_df["is_forecast"] == False]
    fcast = combined_df[combined_df["is_forecast"] == True]

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x    = hist["month"],
        y    = hist["revenue"],
        mode = "lines",
        name = "Actual",
        line = dict(color=COLORS["primary"], width=2),
        hovertemplate = "<b>%{x|%b %Y}</b><br>Actual: $%{y:,.0f}<extra></extra>"
    ))
    fig_fc.add_trace(go.Scatter(
        x         = pd.concat([fcast["month"], fcast["month"].iloc[::-1]]),
        y         = pd.concat([fcast["upper_ci"], fcast["lower_ci"].iloc[::-1]]),
        fill      = "toself",
        fillcolor = COLORS["ci_fill"],
        line      = dict(color="rgba(0,0,0,0)"),
        name      = "95% CI",
        hoverinfo = "skip"
    ))
    fig_fc.add_trace(go.Scatter(
        x    = fcast["month"],
        y    = fcast["forecast"],
        mode = "lines+markers",
        name = "Forecast",
        line = dict(color=COLORS["forecast"], width=2, dash="dot"),
        marker = dict(size=8, color=COLORS["forecast"], symbol="diamond",
                      line=dict(color=COLORS["chart_bg"], width=1.5)),
        hovertemplate = "<b>%{x|%b %Y}</b><br>Forecast: $%{y:,.0f}<extra></extra>"
    ))

    apply_dark_theme(fig_fc, "Revenue Forecast  ·  95% Confidence Interval")
    fig_fc.update_layout(height=320)
    fig_fc.update_yaxes(tickprefix="$", tickformat=",.0f")
    st.plotly_chart(fig_fc, use_container_width=True)

    # Forecast mini cards
    fc1, fc2, fc3 = st.columns(3)
    for col, (_, row) in zip([fc1, fc2, fc3], forecast_df.iterrows()):
        label = row["month"].strftime("%B %Y")
        col.markdown(f"""
            <div class="fc-card">
                <div class="fc-month">{label}</div>
                <div class="fc-value">${row['forecast']:,.0f}</div>
                <div class="fc-range">${row['lower_ci']:,.0f} – ${row['upper_ci']:,.0f}</div>
            </div>
        """, unsafe_allow_html=True)


# =============================================================================
# PAGE 2 — PRODUCTS & CATEGORIES
# =============================================================================

def page_products():
    st.markdown('<div class="page-title">Products & Categories</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Revenue performance and return analysis by product and category.</div>', unsafe_allow_html=True)

    df_products = load_top_products()

    st.markdown('<div class="section-header">Top 10 Products by Revenue</div>', unsafe_allow_html=True)

    bar_colors = [CATEGORY_COLORS.get(c, COLORS["primary"]) for c in df_products["category"]]

    fig_products = go.Figure(go.Bar(
        x            = df_products["revenue"],
        y            = df_products["product_name"],
        orientation  = "h",
        marker       = dict(
            color   = bar_colors,
            opacity = 0.85,
            line    = dict(width=0),
        ),
        customdata   = df_products[["category", "total_orders"]],
        hovertemplate = (
            "<b>%{y}</b><br>"
            "Category: %{customdata[0]}<br>"
            "Revenue: $%{x:,.0f}<br>"
            "Orders: %{customdata[1]}<extra></extra>"
        )
    ))

    apply_dark_theme(fig_products, "Top 10 Products by Revenue")
    fig_products.update_layout(height=400, yaxis=dict(autorange="reversed"))
    fig_products.update_xaxes(tickprefix="$", tickformat=",.0f")
    st.plotly_chart(fig_products, use_container_width=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-header">Return Rate by Category</div>', unsafe_allow_html=True)
        df_returns = load_return_by_category()

        return_colors = [
            COLORS["danger"] if r > 10 else COLORS["primary"]
            for r in df_returns["return_rate_pct"]
        ]

        fig_returns = go.Figure(go.Bar(
            x            = df_returns["category"],
            y            = df_returns["return_rate_pct"],
            marker       = dict(color=return_colors, opacity=0.85, line=dict(width=0)),
            customdata   = df_returns[["returned_orders", "total_orders"]],
            hovertemplate = (
                "<b>%{x}</b><br>Return Rate: %{y:.1f}%<br>"
                "Returned: %{customdata[0]}<br>Total: %{customdata[1]}<extra></extra>"
            )
        ))
        fig_returns.add_hline(
            y=10, line_dash="dot", line_color=COLORS["danger"], line_width=1,
            annotation_text="10% benchmark",
            annotation_font_color=COLORS["danger"],
            annotation_font_size=11,
        )
        apply_dark_theme(fig_returns, "Return Rate by Category")
        fig_returns.update_layout(height=320, showlegend=False)
        fig_returns.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig_returns, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Revenue Share by Category</div>', unsafe_allow_html=True)
        df_cat_rev = (
            df_products.groupby("category")["revenue"]
            .sum().reset_index().sort_values("revenue", ascending=False)
        )

        fig_donut = go.Figure(go.Pie(
            labels       = df_cat_rev["category"],
            values       = df_cat_rev["revenue"],
            hole         = 0.62,
            marker       = dict(
                colors = [CATEGORY_COLORS.get(c, COLORS["primary"]) for c in df_cat_rev["category"]],
                line   = dict(color=COLORS["chart_bg"], width=2),
            ),
            textinfo     = "label+percent",
            textfont     = dict(size=11),
            hovertemplate = (
                "<b>%{label}</b><br>Revenue: $%{value:,.0f}<br>Share: %{percent}<extra></extra>"
            )
        ))
        apply_dark_theme(fig_donut, "Category Revenue Distribution")
        fig_donut.update_layout(height=320)
        st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown("""
        <div class="info-box">
            <strong>Industry benchmark:</strong> Return rates above 10% typically signal product quality issues,
            misleading descriptions, or sizing problems (especially Clothing). Categories shown in red exceed this threshold.
        </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGE 3 — CUSTOMER ANALYSIS
# =============================================================================

def page_customers():
    st.markdown('<div class="page-title">Customer Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Customer loyalty segmentation, lifetime value, and payment behaviour.</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-header">Customer Loyalty Segments</div>', unsafe_allow_html=True)
        df_segments = load_customer_segments()

        seg_colors = {
            "One-time":   COLORS["danger"],
            "Occasional": COLORS["warning"],
            "Loyal":      COLORS["success"],
        }

        fig_seg = go.Figure(go.Pie(
            labels       = df_segments["segment"],
            values       = df_segments["customer_count"],
            hole         = 0.58,
            marker       = dict(
                colors = [seg_colors.get(s, COLORS["primary"]) for s in df_segments["segment"]],
                line   = dict(color=COLORS["chart_bg"], width=2),
            ),
            textinfo     = "label+percent",
            textfont     = dict(size=11),
            hovertemplate = (
                "<b>%{label}</b><br>Customers: %{value:,}<br>Share: %{percent}<extra></extra>"
            )
        ))
        apply_dark_theme(fig_seg, "Customer Segments")
        fig_seg.update_layout(height=340)
        st.plotly_chart(fig_seg, use_container_width=True)

        st.markdown("""
            <div class="info-box">
                <strong>Segment rules</strong><br>
                <span class="seg-dot" style="background:#f87171;"></span><strong>One-time</strong> — 1 order only<br>
                <span class="seg-dot" style="background:#fbbf24;"></span><strong>Occasional</strong> — 2–4 orders<br>
                <span class="seg-dot" style="background:#34d399;"></span><strong>Loyal</strong> — 5+ orders
            </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-header">Avg Order Value by Payment Method</div>', unsafe_allow_html=True)
        df_payment = load_aov_by_payment()

        fig_pay = go.Figure(go.Bar(
            x            = df_payment["avg_order_value"],
            y            = df_payment["payment_method"],
            orientation  = "h",
            marker       = dict(color=COLORS["secondary"], opacity=0.8, line=dict(width=0)),
            customdata   = df_payment[["total_orders", "total_revenue"]],
            hovertemplate = (
                "<b>%{y}</b><br>AOV: $%{x:,.2f}<br>"
                "Orders: %{customdata[0]:,}<br>Revenue: $%{customdata[1]:,.0f}<extra></extra>"
            )
        ))
        apply_dark_theme(fig_pay, "Average Order Value by Payment Method")
        fig_pay.update_layout(height=340, yaxis=dict(autorange="reversed"))
        fig_pay.update_xaxes(tickprefix="$", tickformat=",.0f")
        st.plotly_chart(fig_pay, use_container_width=True)

    # Top Customers Table
    st.markdown('<div class="section-header">Top 5 Customers by Lifetime Value</div>', unsafe_allow_html=True)
    df_top = load_top_customers().copy()
    df_top["lifetime_value"]  = df_top["lifetime_value"].apply(lambda x: f"${x:,.2f}")
    df_top["avg_order_value"] = df_top["avg_order_value"].apply(lambda x: f"${x:,.2f}")
    df_top.columns = ["Customer ID", "Customer Name", "Lifetime Value", "Total Orders", "Avg Order Value"]
    df_top = df_top.drop(columns=["Customer ID"])
    st.dataframe(df_top, use_container_width=True, hide_index=True)

    # MoM Growth
    st.markdown('<div class="section-header">Month-over-Month Revenue Growth</div>', unsafe_allow_html=True)
    df_mom = load_mom_growth().dropna(subset=["mom_growth_pct"])

    mom_colors = [COLORS["success"] if g >= 0 else COLORS["danger"] for g in df_mom["mom_growth_pct"]]

    fig_mom = go.Figure(go.Bar(
        x            = df_mom["month"],
        y            = df_mom["mom_growth_pct"],
        marker       = dict(color=mom_colors, opacity=0.85, line=dict(width=0)),
        hovertemplate = "<b>%{x|%b %Y}</b><br>MoM Growth: %{y:.2f}%<extra></extra>"
    ))
    fig_mom.add_hline(y=0, line_color=COLORS["grid"], line_width=1)
    apply_dark_theme(fig_mom, "Month-over-Month Revenue Growth (%)")
    fig_mom.update_layout(height=260, showlegend=False)
    fig_mom.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig_mom, use_container_width=True)


# =============================================================================
# PAGE 4 — REGIONAL PERFORMANCE
# =============================================================================

def page_regional():
    st.markdown('<div class="page-title">Regional Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Revenue, orders, and return rates across all sales regions.</div>', unsafe_allow_html=True)

    df_region = load_revenue_by_region()

    st.markdown('<div class="section-header">Revenue by Region</div>', unsafe_allow_html=True)

    palette = [COLORS["primary"], COLORS["secondary"], COLORS["success"],
               COLORS["warning"], COLORS["forecast"], COLORS["danger"]]

    fig_region = go.Figure(go.Bar(
        x            = df_region["region"],
        y            = df_region["revenue"],
        marker       = dict(
            color   = palette[:len(df_region)],
            opacity = 0.85,
            line    = dict(width=0),
        ),
        customdata   = df_region[["total_orders", "avg_order_value", "return_rate_pct"]],
        hovertemplate = (
            "<b>%{x}</b><br>Revenue: $%{y:,.0f}<br>"
            "Orders: %{customdata[0]:,}<br>"
            "AOV: $%{customdata[1]:,.2f}<br>"
            "Returns: %{customdata[2]:.1f}%<extra></extra>"
        )
    ))
    apply_dark_theme(fig_region, "Revenue by Region")
    fig_region.update_layout(height=340, showlegend=False)
    fig_region.update_yaxes(tickprefix="$", tickformat=",.0f")
    st.plotly_chart(fig_region, use_container_width=True)

    # Region mini-cards
    st.markdown('<div class="section-header">Region Breakdown</div>', unsafe_allow_html=True)
    known = df_region[df_region["region"] != "Unknown"].head(5)
    cols  = st.columns(len(known))

    for col, (_, row) in zip(cols, known.iterrows()):
        col.markdown(f"""
            <div class="region-card">
                <div class="region-name">{row['region']}</div>
                <div class="region-value">${row['revenue']:,.0f}</div>
                <div class="region-meta">{row['total_orders']:,} orders · {row['return_rate_pct']:.1f}% returns</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Full table
    st.markdown('<div class="section-header">Detailed Region Statistics</div>', unsafe_allow_html=True)
    display_region = df_region.copy()
    display_region["revenue"]         = display_region["revenue"].apply(lambda x: f"${x:,.2f}")
    display_region["avg_order_value"] = display_region["avg_order_value"].apply(lambda x: f"${x:,.2f}")
    display_region["return_rate_pct"] = display_region["return_rate_pct"].apply(lambda x: f"{x:.1f}%")
    display_region = display_region.drop(columns=["return_count"])
    display_region.columns = ["Region", "Revenue", "Total Orders", "Avg Order Value", "Return Rate"]
    st.dataframe(display_region, use_container_width=True, hide_index=True)

    st.markdown("""
        <div class="info-box">
            <strong>'Unknown' region</strong> represents orders where the region field was missing in the raw data (~8% null rate).
            These were retained during the cleaning step to avoid data loss.
        </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGE 5 — FORECASTING
# =============================================================================

def page_forecasting():
    st.markdown('<div class="page-title">Revenue Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Linear regression model trained on 24 months of historical data.</div>', unsafe_allow_html=True)

    combined_df, forecast_df, summary = load_forecast()

    # Model stats
    st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
    ms1, ms2, ms3, ms4 = st.columns(4)
    render_kpi_card("Trend Direction",   summary["trend_label"],               ms1)
    render_kpi_card("Monthly Growth",    f"${summary['monthly_growth']:,.2f}", ms2)
    render_kpi_card("R² Score",          str(summary["r2_score"]),             ms3)
    render_kpi_card("RMSE",              f"${summary['rmse']:,.2f}",           ms4)

    st.markdown("<br>", unsafe_allow_html=True)

    # Full forecast chart
    st.markdown('<div class="section-header">Historical Revenue + 3-Month Forecast</div>', unsafe_allow_html=True)

    hist  = combined_df[combined_df["is_forecast"] == False]
    fcast = combined_df[combined_df["is_forecast"] == True]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x    = hist["month"],
        y    = hist["revenue"],
        mode = "lines",
        name = "Actual Revenue",
        fill = "tozeroy",
        fillcolor = "rgba(79, 91, 213, 0.05)",
        line = dict(color=COLORS["primary"], width=2),
        hovertemplate = "<b>%{x|%b %Y}</b><br>Revenue: $%{y:,.0f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x         = pd.concat([fcast["month"], fcast["month"].iloc[::-1]]),
        y         = pd.concat([fcast["upper_ci"], fcast["lower_ci"].iloc[::-1]]),
        fill      = "toself",
        fillcolor = COLORS["ci_fill"],
        line      = dict(color="rgba(0,0,0,0)"),
        name      = "95% Confidence Interval",
        hoverinfo = "skip"
    ))
    fig.add_trace(go.Scatter(
        x    = fcast["month"],
        y    = fcast["forecast"],
        mode = "lines+markers",
        name = "Forecast",
        line = dict(color=COLORS["forecast"], width=2.5, dash="dot"),
        marker = dict(size=10, color=COLORS["forecast"], symbol="diamond",
                      line=dict(color=COLORS["chart_bg"], width=2)),
        hovertemplate = "<b>%{x|%b %Y}</b><br>Forecast: $%{y:,.0f}<extra></extra>"
    ))

    last_hist_date = hist["month"].max()
    fig.add_shape(
        type="line",
        x0=last_hist_date, x1=last_hist_date,
        y0=0, y1=1, xref="x", yref="paper",
        line=dict(color="#2d3050", width=1.5, dash="dot")
    )
    fig.add_annotation(
        x=last_hist_date, y=0.97, xref="x", yref="paper",
        text="Forecast begins",
        showarrow=False, xanchor="left", yanchor="top",
        font=dict(color="#454870", size=11)
    )

    apply_dark_theme(fig, "Revenue Forecast  ·  Actuals + Prediction + 95% CI")
    
    fig.update_layout(height=460)
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)

    # Table + Explanation
    col_table, col_explain = st.columns([1, 1])

    with col_table:
        st.markdown('<div class="section-header">Forecast Table</div>', unsafe_allow_html=True)
        display_fc = forecast_df[["month", "forecast", "lower_ci", "upper_ci"]].copy()
        display_fc["month"]    = display_fc["month"].dt.strftime("%B %Y")
        display_fc["forecast"] = display_fc["forecast"].apply(lambda x: f"${x:,.2f}")
        display_fc["lower_ci"] = display_fc["lower_ci"].apply(lambda x: f"${x:,.2f}")
        display_fc["upper_ci"] = display_fc["upper_ci"].apply(lambda x: f"${x:,.2f}")
        display_fc.columns     = ["Month", "Forecast", "Lower (95%)", "Upper (95%)"]
        st.dataframe(display_fc, use_container_width=True, hide_index=True)

    with col_explain:
        st.markdown('<div class="section-header">How This Works</div>', unsafe_allow_html=True)
        st.markdown(f"""
            <div class="info-box">
                <strong>Model:</strong> Linear Regression<br><br>
                <strong>Formula:</strong><br>
                Revenue = {summary['monthly_growth']:,.2f} × month + {summary['intercept']:,.2f}<br><br>
                <strong>Interpretation:</strong><br>
                Revenue grows approximately <strong>${summary['monthly_growth']:,.2f}</strong>
                per month based on the 24-month trend.<br><br>
                <strong>R² = {summary['r2_score']}</strong> — the model explains
                {summary['r2_score']*100:.1f}% of revenue variance.<br><br>
                <strong>RMSE = ${summary['rmse']:,.2f}</strong> — typical prediction error per month.<br><br>
                <strong>Confidence Interval:</strong> 95% — actual revenue will fall within
                the shaded band with 95% probability.
            </div>
        """, unsafe_allow_html=True)


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("""
            <div style="padding: 8px 0 20px 0;">
                <div style="font-size: 13px; font-weight: 600; color: #c5c9e0; letter-spacing: 0.5px;">
                    E-Commerce Analytics
                </div>
                <div style="font-size: 11px; color: #454870; margin-top: 4px; letter-spacing: 0.3px;">
                    Performance Dashboard
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="border-top: 1px solid #1c1e2a; margin-bottom: 20px;"></div>', unsafe_allow_html=True)

        page = st.radio(
            "Navigate",
            options=[
                "📊  Overview",
                "🛍  Products & Categories",
                "👥  Customer Analysis",
                "🌍  Regional Performance",
                "🔮  Forecasting",
            ],
            label_visibility="collapsed"
        )

        st.markdown('<div style="border-top: 1px solid #1c1e2a; margin: 24px 0 16px 0;"></div>', unsafe_allow_html=True)

        st.markdown("""
            <div style="font-size: 10px; font-weight: 500; color: #454870; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 12px;">
                Pipeline
            </div>
            <div style="font-family: 'DM Mono', monospace; font-size: 11px; line-height: 2; color: #2d3050;">
                generate_data.py<br>
                <span style="color:#1c1e2a;">↓</span><br>
                data_cleaner.py<br>
                <span style="color:#1c1e2a;">↓</span><br>
                analytics.py<br>
                <span style="color:#1c1e2a;">↓</span><br>
                forecasting.py<br>
                <span style="color:#1c1e2a;">↓</span><br>
                <span style="color:#8b90a8;">app.py  ← here</span>
            </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="border-top: 1px solid #1c1e2a; margin: 20px 0 12px 0;"></div>', unsafe_allow_html=True)
        st.markdown("""
            <div style="font-size: 11px; color: #2d3050; line-height: 1.8;">
                Python · Streamlit · Plotly · SQLite
            </div>
        """, unsafe_allow_html=True)

    return page


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    if not check_database():
        st.stop()

    page = render_sidebar()

    if page == "📊  Overview":
        page_overview()
    elif page == "🛍  Products & Categories":
        page_products()
    elif page == "👥  Customer Analysis":
        page_customers()
    elif page == "🌍  Regional Performance":
        page_regional()
    elif page == "🔮  Forecasting":
        page_forecasting()


if __name__ == "__main__":
    main()