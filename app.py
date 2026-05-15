"""
FleetWise SA: Should I Build an Uber Fleet?
A Data Science Portfolio Project
Author: Ngobe | ALX Africa Data Science Certification
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os

# ── PATH FIX ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from data.vehicles import build_dataset, build_comparison, profit_delta, city_config, YEAR_CONFIG
from models.fleet_models import (
    train_models, predict_custom_car, fleet_growth_simulator,
    get_top_cars, feature_importance_df
)

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FleetWise SA",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── STYLING ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

    /* Header — always dark with gradient, text always white inside it */
    .main-header {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #0f3460 100%);
        border-radius: 16px; padding: 2.5rem 3rem; margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .main-header h1 {
        font-size: 3rem; font-weight: 700;
        background: linear-gradient(90deg, #e94560, #f5a623, #4ecdc4);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0 0 0.5rem 0;
    }
    .main-header p { color: rgba(255,255,255,0.65); font-size: 1.05rem; margin: 0; }

    /* Section titles — use Streamlit body text colour so they always show */
    .section-title {
        font-size: 1.4rem; font-weight: 600;
        color: var(--text-color, inherit);
        border-left: 3px solid #e94560;
        padding-left: 0.8rem; margin: 2rem 0 1rem 0;
    }

    /* Insight boxes — adaptive background + text, teal border stays */
    .insight-box {
        background: rgba(78,205,196,0.10);
        border: 1px solid rgba(78,205,196,0.40);
        border-radius: 10px; padding: 1rem 1.3rem; font-size: 0.92rem;
        color: var(--text-color, inherit);
        margin: 0.8rem 0;
    }

    /* Risk colours — same in both modes */
    .risk-low  { color: #1aad9c; font-weight: 600; }
    .risk-med  { color: #d4890a; font-weight: 600; }
    .risk-high { color: #c0303e; font-weight: 600; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 0.5rem 1.2rem; font-weight: 500; }

    /* Metric cards — use Streamlit surface colour, no hardcoded dark */
    div[data-testid="metric-container"] {
        background: var(--background-color, transparent);
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 12px; padding: 1rem;
    }

    /* Footer — adapt text colour */
    .fw-footer {
        text-align: center; font-size: 0.8rem; padding: 1rem;
        color: rgba(128,128,128,0.7);
    }
</style>
""", unsafe_allow_html=True)

# ─── CACHE ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(city, year):
    return build_dataset(city, year)

@st.cache_resource
def load_models(city, year):
    df = build_dataset(city, year)
    return train_models(df), df

@st.cache_data
def load_comparison(city):
    return build_comparison(city)

@st.cache_data
def load_delta(city):
    return profit_delta(city)

# ─── CHART THEME ─────────────────────────────────────────────────────────────
PLOTLY_CONFIG = {"displayModeBar": False}

def chart_layout(fig, height=None, **kwargs):
    """Apply a clean neutral theme optimised for both desktop and mobile.

    Key mobile fixes:
    - title_x/title_xanchor: left-align title so it sits below (not behind) the modebar
    - margin t=56: gives the modebar its own 44px zone before the title starts
    - legend orientation=h + y=-0.28: moves legend below chart, freeing horizontal space
    - font size=11: tighter axis labels on narrow viewports
    """
    updates = dict(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_family="Space Grotesk",
        font_color=None,
        title_x=0.0,
        title_xanchor="left",
        title_pad=dict(l=4, t=4),
        margin=dict(l=10, r=10, t=56, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.28,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        font=dict(size=11),
    )
    if height:
        updates["height"] = height
    updates.update(kwargs)
    fig.update_layout(**updates)
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.15)", zerolinecolor="rgba(128,128,128,0.25)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zerolinecolor="rgba(128,128,128,0.25)")
    return fig

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    selected_city = st.selectbox("City", list(city_config.keys()), index=0,
                                 help="City affects demand and profitability")
    selected_year = st.radio(
        "Market Year",
        [2026, 2024],
        format_func=lambda y: YEAR_CONFIG[y]["label"],
        help="2024 = baseline before fuel crisis. 2026 = current market conditions."
    )
    budget    = st.slider("Your Budget (R)", 100_000, 800_000, 200_000, 10_000, format="R%d")
    ownership = st.radio("Ownership Model", ["Cash Purchase", "Bank Finance"])
    st.markdown("---")
    st.markdown("### 📊 Filter Vehicles")
    tiers      = st.multiselect("Uber Tier", ["UberX", "UberComfort", "UberXL"],
                                default=["UberX", "UberComfort"])
    categories = st.multiselect("Vehicle Type",
                                ["Hatchback", "Sedan", "SUV", "MPV", "Bakkie"],
                                default=["Hatchback", "Sedan"])
    st.markdown("---")
    st.caption("FleetWise SA v1.1 | NGOBE")

# ─── LOAD ────────────────────────────────────────────────────────────────────
models, df_full = load_models(selected_city, selected_year)
df = df_full.copy()
if tiers:
    df = df[df["Uber_Tier"].isin(tiers)]
if categories:
    df = df[df["Category"].isin(categories)]

df_budget  = df[df["Price_R"] <= budget]
profit_col = "Net_Profit_Financed" if ownership == "Bank Finance" else "Net_Profit_Cash"

# Pre-load comparison data (used in Year Comparison tab)
df_compare = load_comparison(selected_city)
df_delta   = load_delta(selected_city)
year_cfg   = YEAR_CONFIG[selected_year]

# ─── HEADER ──────────────────────────────────────────────────────────────────
year_badge = "🟢 2026 Current" if selected_year == 2026 else "🟡 2024 Baseline"
fuel_badge = f"R{year_cfg['fuel_price']:.2f}/l"
rate_badge = f"{year_cfg['financing_rate']*100:.2f}% finance"
st.markdown(f"""
<div class="main-header">
    <h1>🚗 FleetWise SA</h1>
    <p>Should I Build an Uber Fleet? · A Data-Driven Investment Analysis for South African Markets</p>
    <p style="margin-top:0.6rem; font-size:0.85rem; color:rgba(255,255,255,0.45);">
        {year_badge} &nbsp;·&nbsp; Fuel: <strong style="color:#f5a623">{fuel_badge}</strong>
        &nbsp;·&nbsp; Finance: <strong style="color:#4ecdc4">{rate_badge}</strong>
        &nbsp;·&nbsp; City: <strong>{selected_city}</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# ─── TOP METRICS ─────────────────────────────────────────────────────────────
best_car = df_budget.loc[df_budget[profit_col].idxmax()] if not df_budget.empty else None
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Vehicles Analysed", len(df_full), delta=f"{len(df_budget)} in budget")
with col2:
    if best_car is not None:
        st.metric("Best Monthly Profit", f"R {best_car[profit_col]:,.0f}",
                  delta=best_car["Model"].split()[1])
with col3:
    avg_roi = df_budget["Annual_ROI_Pct"].mean() if not df_budget.empty else 0
    st.metric("Avg Annual ROI", f"{avg_roi:.1f}%", delta=selected_city)
with col4:
    avg_be = df_budget["Breakeven_Months"].median() if not df_budget.empty else 0
    st.metric("Median Breakeven", f"{avg_be:.0f} months")
with col5:
    profitable = (df_budget[profit_col] > 0).sum() if not df_budget.empty else 0
    st.metric("Profitable Vehicles", f"{profitable}/{len(df_budget)}")

st.markdown("---")

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 EDA & Market Overview", "🏆 Vehicle Rankings", "🤖 ML Profit Predictor",
    "📈 Fleet Growth Simulator", "⚠️ Risk Dashboard",
    "📅 2024 vs 2026 Comparison", "📋 Full Dataset"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 · EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-title">Market Overview · Exploratory Data Analysis</p>', unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        fig = px.box(df_full, x="Uber_Tier", y="Net_Profit_Cash", color="Uber_Tier",
                     color_discrete_sequence=["#4ecdc4", "#e94560", "#f5a623"],
                     points="outliers",
                     title="Monthly Net Profit by Uber Tier",
                     labels={"Net_Profit_Cash": "Monthly Net Profit (R)", "Uber_Tier": "Tier"})
        fig.update_layout(showlegend=False)
        chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    with col_r:
        fig2 = px.scatter(df_full, x="Price_R", y="Net_Profit_Cash",
                          color="Uber_Tier", size="Annual_ROI_Pct", hover_name="Model",
                          color_discrete_sequence=["#4ecdc4", "#e94560", "#f5a623"],
                          title="Price vs Monthly Profit (bubble = ROI %)",
                          labels={"Price_R": "Vehicle Price (R)", "Net_Profit_Cash": "Monthly Net Profit (R)"})
        chart_layout(fig2)
        st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG)

    col_l2, col_r2 = st.columns(2)

    with col_l2:
        cost_summary = df_full.groupby("Category")[
            ["Fuel_Cost_Monthly", "Insurance_Monthly", "Maintenance_Monthly"]
        ].mean().round(0)
        cost_melted = cost_summary.reset_index().melt(id_vars="Category")
        cost_melted["variable"] = cost_melted["variable"].map({
            "Fuel_Cost_Monthly":    "Fuel",
            "Insurance_Monthly":    "Insurance",
            "Maintenance_Monthly":  "Maintenance",
        })
        fig3 = px.bar(
            cost_melted,
            y="Category", x="value", color="variable", barmode="stack",
            orientation="h",
            color_discrete_sequence=["#e94560", "#f5a623", "#a8dadc"],
            title="Avg Monthly Cost Breakdown by Vehicle Type",
            labels={"value": "Monthly Cost (R)", "variable": "", "Category": ""},
        )
        fig3.update_layout(yaxis={"categoryorder": "total ascending"})
        chart_layout(fig3, height=340)
        # Override: give x-axis title its own row, legend sits below that
        fig3.update_layout(
            margin=dict(l=10, r=10, t=56, b=90),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.38,
                xanchor="center",
                x=0.5,
                font=dict(size=11),
            ),
        )
        st.plotly_chart(fig3, use_container_width=True, config=PLOTLY_CONFIG)

    with col_r2:
        city_data = []
        for city_name in list(city_config.keys())[:6]:
            cdf = build_dataset(city_name, selected_year)
            city_data.append({"City": city_name, "Avg_ROI": cdf["Annual_ROI_Pct"].mean()})
        city_compare = pd.DataFrame(city_data).sort_values("Avg_ROI", ascending=True)
        fig4 = px.bar(city_compare, x="Avg_ROI", y="City", orientation="h",
                      color="Avg_ROI", color_continuous_scale=["#e94560", "#f5a623", "#4ecdc4"],
                      title="Average Annual ROI by City", labels={"Avg_ROI": "Annual ROI (%)"})
        chart_layout(fig4, showlegend=False)
        st.plotly_chart(fig4, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown('<p class="section-title">Key Insights from the Data</p>', unsafe_allow_html=True)
    best_tier      = df_full.groupby("Uber_Tier")["Net_Profit_Cash"].mean().idxmax()
    best_cat       = df_full.groupby("Category")["Annual_ROI_Pct"].mean().idxmax()
    worst_cost_cat = df_full.groupby("Category")["Fuel_Cost_Monthly"].mean().idxmax()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="insight-box">💡 <strong>{best_tier}</strong> vehicles generate the highest average monthly profit — driven by higher fare rates and consistent demand.</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="insight-box">📈 <strong>{best_cat}</strong> vehicles deliver the best average ROI — balancing low purchase price with adequate earning potential.</div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="insight-box">⛽ <strong>{worst_cost_cat}</strong> vehicles carry the highest fuel costs — fuel efficiency is a critical variable in net margin.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 · RANKINGS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">Vehicle Rankings · Within Your Budget</p>', unsafe_allow_html=True)

    if df_budget.empty:
        st.warning("No vehicles found within your budget and selected filters. Adjust the sidebar.")
    else:
        sort_by = st.selectbox("Rank by",
            ["Net_Profit_Cash", "Annual_ROI_Pct", "Breakeven_Months", "Risk_Score"],
            format_func=lambda x: {
                "Net_Profit_Cash":  "Monthly Profit",
                "Annual_ROI_Pct":   "Annual ROI %",
                "Breakeven_Months": "Breakeven Speed (shortest first)",
                "Risk_Score":       "Lowest Risk"
            }[x])

        ascending = sort_by in ["Breakeven_Months", "Risk_Score"]
        top_df    = df_budget.sort_values(sort_by, ascending=ascending).head(10).reset_index(drop=True)
        top_df.index += 1

        fig_rank = px.bar(top_df, x=sort_by, y="Model", orientation="h",
                          color="Uber_Tier",
                          color_discrete_sequence=["#4ecdc4", "#e94560", "#f5a623"],
                          text=sort_by,
                          title=f"Top 10 Vehicles Ranked by {sort_by.replace('_', ' ')}",
                          hover_data=["Price_R", "Net_Profit_Cash", "Annual_ROI_Pct", "Risk_Score"])
        fig_rank.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        chart_layout(fig_rank, height=450, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_rank, use_container_width=True, config=PLOTLY_CONFIG)

        display_cols = {
            "Model": "Vehicle", "Category": "Type", "Uber_Tier": "Tier",
            "Price_R": "Price (R)", "Net_Profit_Cash": "Monthly Profit",
            "Annual_ROI_Pct": "ROI %", "Breakeven_Months": "Breakeven (mo)",
            "Risk_Score": "Risk Score", "Fuel_L100km": "Fuel (L/100km)"
        }
        st.dataframe(top_df[list(display_cols.keys())].rename(columns=display_cols),
                     use_container_width=True, height=350)

        st.markdown('<p class="section-title">Head-to-Head: Top 3 Vehicles</p>', unsafe_allow_html=True)
        top3 = top_df.head(3)
        categories_radar = ["Monthly Profit", "ROI %", "Breakeven Speed", "Low Risk", "Low Fuel Cost"]

        def normalize_series(s, invert=False):
            mn, mx = s.min(), s.max()
            norm = (s - mn) / (mx - mn + 1e-9)
            return 1 - norm if invert else norm

        all_norm = pd.DataFrame({
            "Monthly Profit":  normalize_series(top_df["Net_Profit_Cash"]),
            "ROI %":           normalize_series(top_df["Annual_ROI_Pct"]),
            "Breakeven Speed": normalize_series(top_df["Breakeven_Months"], invert=True),
            "Low Risk":        normalize_series(top_df["Risk_Score"], invert=True),
            "Low Fuel Cost":   normalize_series(top_df["Fuel_L100km"], invert=True),
        })

        fig_radar   = go.Figure()
        colors_r    = ["#4ecdc4", "#e94560", "#f5a623"]
        base_colors = ["78, 205, 196", "233, 69, 96", "245, 166, 35"]

        for i, (_, row) in enumerate(top3.iterrows()):
            vals = all_norm.iloc[i].tolist()
            vals.append(vals[0])
            fig_radar.add_trace(go.Scatterpolar(
                r=vals,
                theta=categories_radar + [categories_radar[0]],
                fill='toself',
                name=row["Model"].split()[0] + " " + row["Model"].split()[1],
                line_color=colors_r[i],
                fillcolor=f"rgba({base_colors[i]}, 0.12)"
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Normalized Performance: Top 3 Vehicles"
        )
        chart_layout(fig_radar, height=400)
        st.plotly_chart(fig_radar, use_container_width=True, config=PLOTLY_CONFIG)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 · ML PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">ML Profit Predictor · Enter Any Vehicle</p>', unsafe_allow_html=True)
    col_fi, col_pred = st.columns([1, 1])

    with col_fi:
        st.markdown("**Feature Importance (Random Forest)**")
        fi_df  = feature_importance_df(models, df_full)
        fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                        color="Importance",
                        color_continuous_scale=["#1a1a2e", "#e94560", "#f5a623"],
                        title="What Drives Monthly Profit?")
        chart_layout(fig_fi, height=360, showlegend=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_fi, use_container_width=True, config=PLOTLY_CONFIG)
        st.markdown(f"""
        <div class="insight-box">
        🤖 <strong>Model Performance</strong><br>
        Profit Predictor R² = <strong>{models['profit_r2']:.3f}</strong><br>
        ROI Predictor R² = <strong>{models['roi_r2']:.3f}</strong><br>
        Algorithm: Random Forest + Gradient Boosting Ensemble
        </div>
        """, unsafe_allow_html=True)

    with col_pred:
        st.markdown("**Predict for a Custom Vehicle**")
        c1, c2 = st.columns(2)
        with c1:
            p_price      = st.number_input("Vehicle Price (R)", 100000, 800000, 189900, 5000)
            p_fuel       = st.number_input("Fuel (L/100km)", 4.0, 15.0, 6.5, 0.1)
            p_insurance  = st.number_input("Insurance (R/month)", 500, 3000, 1000, 50)
        with c2:
            p_maintenance = st.number_input("Maintenance (R/month)", 300, 2000, 700, 50)
            p_seats       = st.selectbox("Seats", [2, 5, 7, 14], index=1)
            p_tier        = st.selectbox("Uber Tier", ["UberX", "UberComfort", "UberXL"])
        p_category = st.selectbox("Category", ["Hatchback", "Sedan", "SUV", "MPV", "Bakkie"])
        p_demand   = city_config.get(selected_city, 1.0)

        if st.button("🔮 Predict Profitability", type="primary"):
            result = predict_custom_car(
                models, p_price, p_fuel, p_insurance, p_maintenance,
                p_seats, p_category, p_tier, p_demand
            )
            st.markdown("---")
            r1, r2, r3 = st.columns(3)
            r1.metric("Monthly Profit", f"R {result['predicted_monthly_profit']:,}")
            r2.metric("Annual ROI",     f"{result['predicted_annual_roi']}%")
            r3.metric("Breakeven",      f"{result['predicted_breakeven_months']:.0f} months")
            verdict = "✅ PROFITABLE" if result["predicted_monthly_profit"] > 0 else "❌ NOT VIABLE"
            color   = "#4ecdc4" if result["predicted_monthly_profit"] > 0 else "#e94560"
            st.markdown(
                f'<div class="insight-box" style="border-color:{color}; font-size:1.1rem; text-align:center;">'
                f'<strong>{verdict}</strong> — Estimated R {result["predicted_monthly_profit"]:,}/month in {selected_city}</div>',
                unsafe_allow_html=True
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 · FLEET GROWTH SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-title">Fleet Growth Simulator · Reinvest & Scale</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-box">
    💡 This simulator models a <strong>real wealth-building strategy</strong>: start with 1 car, reinvest profits,
    and grow to a full fleet. It answers: <em>"If I buy one car today, how long until I own 5?"</em>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        sim_profit       = st.number_input("Monthly Profit per Car (R)", 2000, 15000, 6000, 500)
        # FIX: integer slider (50-100) avoids the broken float format="%.0f%%" bug
        sim_reinvest_pct = st.slider("Reinvestment Rate (%)", 50, 100, 80, 5,
                                     help="% of monthly profit set aside to buy the next car")
        sim_reinvest     = sim_reinvest_pct / 100
    with c2:
        sim_target = st.slider("Target Fleet Size", 2, 10, 5)
        sim_price  = st.number_input("Price per Car (R)", 100000, 500000, 200000, 10000)
    with c3:
        st.markdown("")
        st.markdown("")
        if not df_budget.empty:
            rec_car = df_budget.loc[df_budget["Net_Profit_Cash"].idxmax()]
            st.markdown(f"""
            <div class="insight-box">
            📌 <strong>Recommended starting car:</strong><br>
            {rec_car['Model']}<br>
            R {rec_car['Net_Profit_Cash']:,.0f}/month profit<br>
            {rec_car['Uber_Tier']} · {rec_car['Category']}
            </div>
            """, unsafe_allow_html=True)

    sim_df, total_months = fleet_growth_simulator(sim_profit, sim_reinvest, sim_target, sim_price)

    fig_sim = make_subplots(specs=[[{"secondary_y": True}]])
    fig_sim.add_trace(go.Scatter(
        x=sim_df["month"], y=sim_df["cars"], name="Cars Owned",
        mode="lines+markers", line=dict(color="#4ecdc4", width=3), marker=dict(size=8)
    ), secondary_y=False)
    fig_sim.add_trace(go.Bar(
        x=sim_df["month"], y=sim_df["total_monthly_income"],
        name="Monthly Income (R)", marker_color="#e94560", opacity=0.6
    ), secondary_y=True)
    chart_layout(fig_sim, height=400, title=f"Fleet Growth to {sim_target} Cars · {total_months} months total")
    fig_sim.update_yaxes(title_text="Cars Owned",         secondary_y=False)
    fig_sim.update_yaxes(title_text="Monthly Income (R)", secondary_y=True)
    st.plotly_chart(fig_sim, use_container_width=True, config=PLOTLY_CONFIG)

    years      = total_months // 12
    months_rem = total_months % 12
    final_income = sim_df.iloc[-1]["total_monthly_income"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Time to Fleet Goal",     f"{years}y {months_rem}mo")
    m2.metric("Final Monthly Income",   f"R {final_income:,}")
    m3.metric("Total Investment Value", f"R {sim_target * sim_price:,}")

    milestones = sim_df[sim_df["cars"] != sim_df["cars"].shift(1)].reset_index(drop=True)
    milestones["Year/Month"] = milestones["month"].apply(lambda m: f"Year {m//12}, Month {m%12}")
    milestones.columns = ["Month", "Cars", "Savings (R)", "Monthly Income (R)", "Timeline"]
    st.dataframe(milestones[["Timeline", "Cars", "Monthly Income (R)", "Savings (R)"]],
                 use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 · RISK DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-title">Risk Dashboard · Know Before You Buy</p>', unsafe_allow_html=True)

    risk_df = df_full.copy()
    risk_df["Risk_Category"] = pd.cut(risk_df["Risk_Score"],
        bins=[0, 3, 6, 10], labels=["Low Risk", "Medium Risk", "High Risk"])

    col1, col2 = st.columns(2)
    with col1:
        fig_risk = px.scatter(risk_df, x="Fuel_Cost_Monthly", y="Maintenance_Monthly",
                              color="Risk_Category",
                              color_discrete_map={"Low Risk": "#4ecdc4", "Medium Risk": "#f5a623", "High Risk": "#e94560"},
                              size="Price_R", hover_name="Model",
                              title="Risk Map: Fuel Cost vs Maintenance Cost",
                              labels={"Fuel_Cost_Monthly": "Monthly Fuel Cost (R)",
                                      "Maintenance_Monthly": "Monthly Maintenance (R)"})
        chart_layout(fig_risk)
        st.plotly_chart(fig_risk, use_container_width=True, config=PLOTLY_CONFIG)

    with col2:
        fig_resale = px.bar(risk_df.nlargest(12, "Resale_Value_R"),
                            x="Resale_Value_R", y="Model", orientation="h",
                            color="Category",
                            color_discrete_sequence=["#4ecdc4", "#e94560", "#f5a623", "#a8dadc", "#e0aaff"],
                            title="Top 12 Vehicles by 3-Year Resale Value",
                            labels={"Resale_Value_R": "Estimated Resale Value (R)"})
        chart_layout(fig_resale, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_resale, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown('<p class="section-title">Risk Assessment: All Vehicles</p>', unsafe_allow_html=True)
    risk_filter = st.selectbox("Show", ["All", "Low Risk", "Medium Risk", "High Risk"])
    show_df = risk_df if risk_filter == "All" else risk_df[risk_df["Risk_Category"] == risk_filter]
    risk_display = show_df[[
        "Model", "Category", "Uber_Tier", "Price_R", "Risk_Score", "Risk_Category",
        "Fuel_L100km", "Maintenance_Monthly", "Resale_Value_R", "Net_Profit_Cash"
    ]].sort_values("Risk_Score").reset_index(drop=True)
    st.dataframe(risk_display, use_container_width=True, height=400)

    # FIX: guard against empty subset before .iloc[0]
    low_risk_df = risk_df[risk_df["Risk_Category"] == "Low Risk"]
    if not low_risk_df.empty:
        low_risk_best = low_risk_df.nlargest(1, "Net_Profit_Cash").iloc[0]
        st.markdown(f"""
        <div class="insight-box">
        🛡️ <strong>Safest High-Earner:</strong> {low_risk_best['Model']} —
        Risk Score <span class="risk-low">{low_risk_best['Risk_Score']}</span>,
        Monthly Profit <strong>R {low_risk_best['Net_Profit_Cash']:,.0f}</strong>,
        Resale Value <strong>R {low_risk_best['Resale_Value_R']:,.0f}</strong>.
        A strong first-car choice if capital preservation matters.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="insight-box">ℹ️ No vehicles fall into the Low Risk category with current filters.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 · 2024 vs 2026 COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<p class="section-title">2024 vs 2026 · What the Fuel Crisis Did to Fleet Profitability</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    📅 This tab compares <strong>2024 baseline</strong> (R22.50/l · prime ~13%) against
    <strong>2026 current</strong> (R26.63/l · prime 10.25%) for the same vehicle lineup.
    The fuel price rose <strong>R4.13/litre (+18%)</strong> while finance costs
    <em>dropped</em> ~170bps — revealing which vehicles are resilient and which are now marginal.
    </div>
    """, unsafe_allow_html=True)

    # ── Key macro comparison ─────────────────────────────────────────────────
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Fuel 2024", "R22.50/l")
    mc2.metric("Fuel 2026", "R26.63/l", delta="+R4.13 (+18%)", delta_color="inverse")
    mc3.metric("Finance Rate 2024", "15.00%")
    mc4.metric("Finance Rate 2026", "13.25%", delta="-1.75%", delta_color="normal")

    st.markdown("---")

    # ── Profit delta waterfall ───────────────────────────────────────────────
    st.markdown('<p class="section-title">Profit Change Per Vehicle (2024 → 2026)</p>', unsafe_allow_html=True)

    delta_plot = df_delta.copy()
    delta_plot["Colour"] = delta_plot["Profit_Delta"].apply(
        lambda x: "#4ecdc4" if x >= 0 else "#e94560"
    )
    delta_plot["Label"] = delta_plot["Profit_Delta"].apply(
        lambda x: f"+R{x:,.0f}" if x >= 0 else f"R{x:,.0f}"
    )
    delta_sorted = delta_plot.sort_values("Profit_Delta", ascending=True)

    fig_delta = px.bar(
        delta_sorted, x="Profit_Delta", y="Model", orientation="h",
        color="Colour", color_discrete_map="identity",
        text="Label",
        title="Monthly Profit Change: 2024 → 2026 (same vehicle, new market conditions)",
        labels={"Profit_Delta": "Profit Change (R/month)", "Model": ""}
    )
    fig_delta.update_traces(textposition="outside")
    chart_layout(fig_delta, height=900, yaxis={"categoryorder": "total ascending"}, showlegend=False)
    fig_delta.add_vline(x=0, line_dash="dash", line_color="rgba(128,128,128,0.5)")
    st.plotly_chart(fig_delta, use_container_width=True, config=PLOTLY_CONFIG)

    # ── Side-by-side profit scatter ──────────────────────────────────────────
    st.markdown('<p class="section-title">Profit Distribution Shift</p>', unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        fig_box = px.box(
            df_compare, x="Uber_Tier", y="Net_Profit_Cash",
            color="Year", color_discrete_map={2024: "#f5a623", 2026: "#4ecdc4"},
            title="Monthly Profit by Tier: 2024 vs 2026",
            labels={"Net_Profit_Cash": "Monthly Net Profit (R)", "Year": "Year"}
        )
        chart_layout(fig_box)
        st.plotly_chart(fig_box, use_container_width=True, config=PLOTLY_CONFIG)

    with col_r:
        # Fuel efficiency scatter coloured by year
        fig_fuel = px.scatter(
            df_compare, x="Fuel_L100km", y="Net_Profit_Cash",
            color="Year", color_discrete_map={2024: "#f5a623", 2026: "#4ecdc4"},
            hover_name="Model",
            title="Fuel Efficiency vs Profit: 2024 vs 2026",
            labels={"Fuel_L100km": "Fuel Consumption (L/100km)",
                    "Net_Profit_Cash": "Monthly Net Profit (R)"}
        )
        chart_layout(fig_fuel)
        st.plotly_chart(fig_fuel, use_container_width=True, config=PLOTLY_CONFIG)

    # ── Vehicles that crossed to loss ────────────────────────────────────────
    st.markdown('<p class="section-title">Vehicles That Became Unprofitable in 2026</p>', unsafe_allow_html=True)
    crossed = df_delta[df_delta["Crossed_To_Loss"] == True]
    if crossed.empty:
        st.markdown('<div class="insight-box">✅ No vehicles crossed into loss — all remained cash-profitable despite the fuel increase. However, financed vehicles face tighter margins.</div>', unsafe_allow_html=True)
    else:
        st.dataframe(
            crossed[["Model", "Profit_2024", "Profit_2026", "Profit_Delta", "Delta_Pct"]]
            .rename(columns={"Profit_2024": "Profit 2024 (R)", "Profit_2026": "Profit 2026 (R)",
                             "Profit_Delta": "Change (R)", "Delta_Pct": "Change %"}),
            use_container_width=True
        )

    # ── Most resilient vehicles ──────────────────────────────────────────────
    st.markdown('<p class="section-title">Most Resilient Vehicles (Smallest Profit Drop)</p>', unsafe_allow_html=True)
    resilient = df_delta.nlargest(8, "Profit_Delta")[
        ["Model", "Profit_2024", "Profit_2026", "Profit_Delta", "Delta_Pct"]
    ].rename(columns={"Profit_2024": "Profit 2024 (R)", "Profit_2026": "Profit 2026 (R)",
                      "Profit_Delta": "Change (R)", "Delta_Pct": "Change %"})
    st.dataframe(resilient, use_container_width=True)

    fig_resil = px.bar(
        df_delta.nlargest(10, "Profit_Delta"),
        x="Model", y=["Profit_2024", "Profit_2026"],
        barmode="group",
        color_discrete_map={"Profit_2024": "#f5a623", "Profit_2026": "#4ecdc4"},
        title="Top 10 Most Resilient Vehicles: 2024 vs 2026 Monthly Profit",
        labels={"value": "Monthly Profit (R)", "variable": "Year"}
    )
    chart_layout(fig_resil, xaxis_tickangle=-30)
    st.plotly_chart(fig_resil, use_container_width=True, config=PLOTLY_CONFIG)

    # ── Key finding callout ──────────────────────────────────────────────────
    most_resilient = df_delta.nlargest(1, "Profit_Delta").iloc[0]
    hardest_hit    = df_delta.nsmallest(1, "Profit_Delta").iloc[0]
    st.markdown(f"""
    <div class="insight-box">
    🔍 <strong>Key Finding:</strong> Despite an 18% fuel price increase,
    fuel-efficient vehicles absorbed the shock far better.
    <strong>{most_resilient["Model"]}</strong> saw the smallest profit erosion
    (change: R {most_resilient["Profit_Delta"]:+,.0f}/month), while
    <strong>{hardest_hit["Model"]}</strong> was hit hardest
    (R {hardest_hit["Profit_Delta"]:+,.0f}/month).
    The simultaneous drop in finance rates partially offset the fuel cost increase —
    making financed purchases in 2026 more competitive than the fuel headlines suggest.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 · FULL DATASET
# ══════════════════════════════════════════════════════════════════════════════
with tab7:
    fuel_note = f"R{year_cfg['fuel_price']}/l"
    rate_note = f"{year_cfg['financing_rate']*100:.2f}%"
    st.markdown(f'<p class="section-title">Full Dataset · {selected_year} · {len(df_full)} South African Vehicles</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="insight-box">
    📦 Showing <strong>{selected_year}</strong> data.
    Fuel: <strong>{fuel_note}/litre</strong> (Gauteng 95).
    Finance rate: <strong>{rate_note} p.a.</strong>
    Prices from AutoTrader SA. Insurance from SAIA benchmarks.
    22 working days/month · 180km/day operating assumption.
    Switch the year in the sidebar to compare datasets.
    </div>
    """, unsafe_allow_html=True)

    search  = st.text_input("Search by model name")
    show_df = df_full[df_full["Model"].str.contains(search, case=False)] if search else df_full
    st.dataframe(show_df.reset_index(drop=True), use_container_width=True, height=500)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csv = df_full.to_csv(index=False).encode("utf-8")
        st.download_button(f"⬇️ Download {selected_year} Dataset (CSV)", csv,
                          f"fleetwise_sa_{selected_year}.csv", "text/csv")
    with col_dl2:
        both = build_comparison(selected_city)
        csv_both = both.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Both Years (CSV)", csv_both,
                          "fleetwise_sa_2024_2026.csv", "text/csv")

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="fw-footer">
    FleetWise SA · Built with Python, Streamlit, Scikit-learn &amp; Plotly ·
    ALX Africa Data Science Portfolio ·
    Data sourced from AutoTrader SA, Uber estimates, DMPR fuel prices &amp; SAIA benchmarks
</div>
""", unsafe_allow_html=True)
