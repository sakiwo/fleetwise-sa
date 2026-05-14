# 🚗 FleetWise SA — Should I Build an Uber Fleet?

**A Data Science Portfolio Project | ALX Africa Data Science Certification 2024–2026**

> *"I didn't just predict house prices. I built a system to answer a real business question I actually had to make a decision on."*

---

## The Problem

Ride-hailing vehicle ownership is one of the most accessible income-generating investments available to working South Africans. But buying the wrong car, at the wrong price, in the wrong city — can turn a business idea into a money pit.

**FleetWise SA answers the question most people answer with guesswork:**

> Which vehicle should I buy first? How long until I break even? Can I grow to a fleet — and if so, how fast?

This project models 46 South African vehicles across 8 cities, across two market years (2024 baseline and 2026 current), using real fuel prices, real finance rates, and real Uber fare structures to produce actionable investment intelligence.

---

## Why This Project Exists

This isn't a classroom exercise. The analysis was built to inform a real decision — whether to purchase a vehicle for Uber ride-hailing as an ownership-based income source in Kimberley, Northern Cape.

The fuel crisis of 2026 (petrol rising to R26.63/litre, the highest on record) made this analysis urgent. Understanding which vehicles survive a cost shock — and which don't — is the difference between a profitable fleet and a sunk investment.

---

## Project Structure

```
fleetwise_sa/
│
├── app.py              # Main Streamlit application (7 tabs)
│
├── data/
│   └── vehicles.py           # Dataset generator — 46 vehicles × 2 years × 8 cities
│
└── models/
    └── fleet_models.py       # ML models, fleet simulator, prediction functions
```

---

## Data Sources

All figures are sourced from publicly available South African market data:

| Data Point | Source | Value Used |
|---|---|---|
| Petrol 95 price (2026) | Dept Mineral & Petroleum Resources, 6 May 2026 | R26.63/litre (Gauteng inland) |
| Petrol 95 price (2024) | DMPR historical average | R22.50/litre |
| Prime lending rate (2026) | SARB / CEIC, March 2026 | 10.25% (repo 6.75%) |
| Vehicle finance rate (2026) | Standard bank margin: prime + 3% | 13.25% p.a. |
| Vehicle prices | AutoTrader SA listings, May 2026 | Per vehicle |
| Insurance estimates | SAIA benchmarks by vehicle category | Per vehicle |
| Uber fare structure | UberX / UberComfort / UberXL SA rates | Per tier |
| Used car market data | AutoTrader / IOL, February 2026 | Market context |

---

## Dataset

**46 vehicles** across 6 categories (Hatchback, Sedan, SUV, MPV, Bakkie, Minibus), 3 Uber tiers (UberX, UberComfort, UberXL), 8 cities, and 2 market years.

**23 engineered features per vehicle per city/year**, including:

- Monthly revenue (based on trip volume, fares, Uber commission)
- Monthly fuel cost (based on manufacturer fuel figures × fuel price × operating km)
- Net profit (cash purchase and financed)
- Annual ROI %
- Breakeven months
- 3-year resale value estimate
- Composite risk score (fuel risk + maintenance risk + demand risk)

**Operating assumptions:**
- 22 working days/month
- 180 km/day (consistent with SA Uber driver patterns)
- 25% Uber platform commission
- R10,000 deposit (10%) on financed vehicles
- 60-month loan term

---

## Machine Learning

Three models are trained on the full multi-city dataset:

| Model | Algorithm | Target | Notes |
|---|---|---|---|
| Profit Predictor | Random Forest (200 trees) | Monthly net profit (R) | Trained on all cities combined |
| ROI Predictor | Gradient Boosting | Annual ROI % | Best cross-validation performance |
| Breakeven Predictor | Ridge Regression | Breakeven months | Regularised for small-sample stability |

**Key finding from feature importance:** Uber tier and vehicle price dominate profit prediction. Fuel consumption becomes the critical differentiator under 2026 pricing conditions.

**Honest ML caveat:** With 46 vehicles × 8 cities = 368 training samples, cross-validation scores are moderate (R² ~0.58 for RF on profit). The models demonstrate the methodology — feature engineering, ensemble selection, and hyperparameter choices — rather than production-grade prediction accuracy. A real deployment would require live trip data from Uber's driver API.

---

## Application Features

### Tab 1 — EDA & Market Overview
Exploratory analysis of profit distributions, cost breakdowns, and ROI by city — establishing the business context before any recommendations.

### Tab 2 — Vehicle Rankings
Filter by budget, city, tier, and ownership model. Rank by profit, ROI, breakeven speed, or risk. Radar chart head-to-head comparison for the top 3.

### Tab 3 — ML Profit Predictor
Enter any vehicle's specs and receive a profit/ROI/breakeven prediction. Feature importance chart explains what the model learned.

### Tab 4 — Fleet Growth Simulator
Models a real wealth-building scenario: start with 1 car, reinvest a chosen % of monthly profit, grow to a target fleet. Shows time-to-fleet, income trajectory, and milestone table.

### Tab 5 — Risk Dashboard
Fuel cost vs maintenance scatter plot by risk category. Resale value rankings. Full filterable risk assessment table.

### Tab 6 — 2024 vs 2026 Comparison
The project's centrepiece analytical section. Shows what the 2026 fuel crisis did to fleet profitability across every vehicle:
- Profit change waterfall (per vehicle)
- Distribution shift by tier
- Vehicles that crossed into loss territory
- Most resilient vehicles (those that held up despite the fuel increase)
- Plain-language key finding callout

### Tab 7 — Full Dataset
Searchable, downloadable dataset. Download current year, or both years combined.

---

## Key Findings

1. **The fuel increase did not collapse profitability** — Uber fare increases (+8%) partially offset the R4.13/litre fuel rise. Average monthly cash profit in Johannesburg actually *increased* slightly from 2024 to 2026.

2. **The spread widened significantly** — Fuel-efficient vehicles (Suzuki S-Presso, Swift, Starlet) gained relative to thirsty alternatives. The risk of choosing a high-consumption vehicle more than doubled in cost terms.

3. **Financing became more attractive in 2026** — The SARB rate-cutting cycle (150bps since Sept 2024) reduced vehicle finance costs. The financed vs cash gap narrowed, making bank-financed entry more viable than it was in 2024.

4. **City matters more than vehicle choice at the margin** — Johannesburg demand multiplier (1.20×) vs Kimberley (0.75×) has a larger profit impact than switching from a Polo Vivo to a Swift.

5. **UberComfort vehicles offer the best ROI per rand** — Higher fare rates more than compensate for their higher purchase price, producing better ROI % than equivalent UberX vehicles.

---

## How to Run

### Requirements

```bash
pip install streamlit pandas numpy plotly scikit-learn
```

### Launch

```bash
cd fleetwise_sa
streamlit run app_fixed.py
```

The app will open at `http://localhost:8501`

---

## Tech Stack

| Tool | Use |
|---|---|
| Python 3.10+ | Core language |
| Pandas / NumPy | Data engineering |
| Scikit-learn | Random Forest, Gradient Boosting, Ridge Regression |
| Plotly | Interactive charts (box, scatter, bar, radar, dual-axis) |
| Streamlit | Web application framework |

---

## Author

**Sakiwo Sibango (NGOBE)**

*This project was built to solve a real problem — and to demonstrate that data science is most valuable when it's applied to decisions that actually matter.*
