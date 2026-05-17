# 🚗 FleetWise SA — Should I Build an Uber Fleet?

**A Data Science Portfolio Project | ALX Africa Data Science Certification**

> *"I didn't just predict house prices. I built a system to answer a real business question I actually had to make a decision on."*

---

## The Problem

Ride-hailing vehicle ownership is one of the most accessible income-generating investments available to working South Africans. But buying the wrong car, at the wrong price, in the wrong city — can turn a business idea into a money pit.

**FleetWise SA answers the question most people answer with guesswork:**

> Which vehicle should I buy first? How long until I break even? Can I grow to a fleet — and if so, how fast?

This project models 46 South African vehicles across 8 cities and two market years (2024 baseline and 2026 current), using real fuel prices, real finance rates, and real Uber fare structures to produce actionable investment intelligence.

---

## Why This Project Exists

This isn't a classroom exercise. The analysis was built to inform a real decision — whether to purchase a vehicle for Uber ride-hailing as an ownership-based income source in Kimberley, Northern Cape.

The fuel crisis of 2026 (petrol at R26.63/litre, up 18% from 2024) made this analysis urgent. Understanding which vehicles survive a cost shock — and which don't — is the difference between a profitable fleet and a sunk investment.

---

## What's New in v1.1

This version moves well beyond a standard data science notebook. Several significant additions were made after the initial build based on real user feedback and deployment learnings.

**Role-Based Entry & Personalised Navigation**
The app opens on a landing screen that asks *"Who are you?"* — Driver, Fleet Manager, or Researcher. Each role sees a curated subset of the 8 tabs relevant to their use case. A Driver doesn't need the market comparison tab. A Researcher doesn't need the fleet simulator. This single change eliminates the "I don't know where to start" problem without rebuilding the underlying app, and makes the portfolio story cleaner: it demonstrates multi-stakeholder UX thinking alongside the data work.

**Step-by-Step Onboarding Tour**
First-time visitors are offered a guided tour after selecting their role. Each step targets a specific tab with a short explanation of what it does and what to look for — with a progress bar, Back/Next controls, and a Skip option available at every step. Users who skip (or skip by accident) can restart the tour from the sidebar at any time via a *Restart tour* button. The tour is role-aware: Drivers see different steps to Fleet Managers.

**2024 vs 2026 Market Comparison**
The fuel price rose R4.13/litre (+18%) between 2024 and 2026 while the prime lending rate fell 150bps — two forces pulling in opposite directions. A dedicated comparison tab shows exactly what that did to every vehicle's monthly profit: waterfall chart per vehicle, distribution shift by tier, a resilience ranking, and a plain-language key finding. This is the analytical centrepiece of the project and the section that most clearly demonstrates applied economic reasoning.

**Honest ML Metrics**
The original model performance display showed a cross-validation R² of −3.866. The root cause: the dataset is engineered (profit is a deterministic function of known inputs), so CV on 46 vehicles with 9 test samples per fold produces wildly unstable scores. The fix trains on all cities × both years (736 rows), switches the displayed metric to MAE (Mean Absolute Error — R591/month for profit, 2.3% for ROI), and includes an explanation of why MAE is more appropriate. In-sample R² of 0.98 is also shown to confirm the models learned the relationships correctly. Being transparent about this in the app, rather than hiding it, is itself a demonstration of statistical maturity.

**Live Feedback Collection**
A feedback tab collects responses from real users via Google Sheets. Each submission captures role, usefulness, what the user was trying to figure out, what was missing, and an overall score (1–5). This data feeds directly into the feature brief for version 2 — the app is functioning as its own discovery phase.

**Light/Dark Mode & Mobile Compatibility**
Section titles and insight boxes now use Streamlit's native theme variables, rendering correctly in both modes. Charts use transparent backgrounds rather than hardcoded dark colours. On mobile (≤640px), columns stack vertically, chart titles are left-aligned to avoid the Plotly modebar overlap, and tab labels scroll horizontally.

---

## Project Structure

```
fleetwise_sa/
│
├── app.py                    # Main application — 8 tabs, role routing, onboarding tour
│
├── data/
│   └── vehicles.py           # Dataset generator — 46 vehicles × 2 years × 8 cities
│
└── models/
    └── fleet_models.py       # ML models, fleet simulator, prediction functions
```

---

## Data Sources

| Data Point | Source | Value Used |
|---|---|---|
| Petrol 95 price (2026) | Dept Mineral & Petroleum Resources, 6 May 2026 | R26.63/litre (Gauteng inland) |
| Petrol 95 price (2024) | DMPR historical average | R22.50/litre |
| Prime lending rate (2026) | SARB / CEIC, March 2026 | 10.25% (repo 6.75%) |
| Vehicle finance rate (2026) | Prime + 3% bank margin | 13.25% p.a. |
| Vehicle prices | AutoTrader SA listings, May 2026 | Per vehicle |
| Insurance estimates | SAIA benchmarks by vehicle category | Per vehicle |
| Uber fare structure | UberX / UberComfort / UberXL SA rates | Per tier |
| Used car market data | AutoTrader / IOL, February 2026 | Avg used car R428,562 |

---

## Dataset

**46 vehicles** across 6 categories (Hatchback, Sedan, SUV, MPV, Bakkie, Minibus), 3 Uber tiers (UberX, UberComfort, UberXL), 8 cities, and 2 market years — **736 total data points** used for ML training.

**23 engineered features per vehicle per city/year:**

- Monthly revenue (trip volume × fare structure × Uber commission)
- Monthly fuel cost (manufacturer figures × fuel price × operating km)
- Net profit — cash purchase and bank-financed
- Annual ROI %
- Breakeven months
- 3-year resale value estimate
- Composite risk score (fuel risk + maintenance risk + demand risk)

**Operating assumptions:** 22 working days/month · 180 km/day · 25% Uber commission · 10% deposit · 60-month loan term

---

## Machine Learning

Three models trained on the full multi-city, multi-year dataset:

| Model | Algorithm | Target | MAE |
|---|---|---|---|
| Profit Predictor | Random Forest (300 trees, depth 5) | Monthly net profit (R) | R591/month |
| ROI Predictor | Gradient Boosting (lr=0.1, depth 2) | Annual ROI % | 2.3% |
| Breakeven Predictor | Ridge Regression (α=10) | Breakeven months | 3.7 months |

Both Random Forest and Gradient Boosting in-sample R² exceed 0.97.

**Why MAE instead of CV R²:**
The dataset is engineered — profit is a deterministic function of known inputs. CV on this structure produces unstable scores because any fold may contain vehicles whose city demand range wasn't seen in training, causing wildly negative R² values. MAE answers the practical question: *how wrong are the predictions in rand terms?* A production deployment would retrain on live Uber trip data, at which point CV R² becomes the appropriate generalisation metric.

**Key feature importance finding:** Uber tier and vehicle price dominate profit prediction. Fuel consumption's weight increased measurably between 2024 and 2026 models, reflecting the real-world impact of the fuel price shock.

---

## Application — 8 Tabs

Tabs are surfaced based on the user's selected role. All tabs remain accessible but each role sees a curated default view.

| Tab | Driver | Fleet Manager | Researcher |
|---|:---:|:---:|:---:|
| 📊 EDA & Market Overview | — | ✅ | ✅ |
| 🏆 Vehicle Rankings | ✅ | — | — |
| 🤖 ML Profit Predictor | ✅ | — | — |
| 📈 Fleet Growth Simulator | — | ✅ | — |
| ⚠️ Risk Dashboard | ✅ | ✅ | — |
| 📅 2024 vs 2026 Comparison | — | ✅ | ✅ |
| 📋 Full Dataset | — | ✅ | ✅ |
| 💬 Feedback | ✅ | ✅ | ✅ |

**📊 EDA & Market Overview** — Profit distributions, cost breakdowns by vehicle type, city-level ROI comparisons. The market context before any recommendations.

**🏆 Vehicle Rankings** — Filter by budget, city, tier, and ownership model. Rank by profit, ROI, breakeven speed, or risk. Radar chart head-to-head for the top 3.

**🤖 ML Profit Predictor** — Enter any vehicle's specs for a profit/ROI/breakeven prediction. Built for vehicles found on AutoTrader that aren't in the dataset. Feature importance chart explains the model's reasoning.

**📈 Fleet Growth Simulator** — Compounding reinvestment model: start with 1 car, reinvest a chosen % of profits, grow to a target fleet. Shows time-to-fleet, income trajectory, and milestone table.

**⚠️ Risk Dashboard** — Fuel cost vs maintenance scatter by risk category. Resale value rankings. Filterable risk assessment table.

**📅 2024 vs 2026 Comparison** — Per-vehicle profit change waterfall, distribution shift by tier, resilience ranking, and key finding callout. The most analytically dense section.

**📋 Full Dataset** — Searchable, downloadable dataset. Current year or both years combined.

**💬 Feedback** — Live form saving to Google Sheets. Captures role, usefulness, unmet needs, and overall rating (1–5).

---

## Key Findings

1. **The fuel increase did not collapse profitability** — Uber fare increases (+8%) partially offset the R4.13/litre fuel rise. Average monthly cash profit in Johannesburg increased slightly from 2024 to 2026.

2. **The spread widened significantly** — Fuel-efficient vehicles (Suzuki S-Presso, Swift, Starlet) gained relative to thirsty alternatives. The cost penalty of choosing a high-consumption vehicle roughly doubled in 2026.

3. **Financing became more attractive in 2026** — The SARB rate-cutting cycle (150bps since Sept 2024) reduced vehicle finance rates from ~15% to 13.25%, narrowing the cash vs financed gap.

4. **City matters more than vehicle choice at the margin** — The Johannesburg demand multiplier (1.20×) vs Kimberley (0.75×) has a larger monthly profit impact than switching between equivalent vehicle models.

5. **UberComfort vehicles offer the best ROI per rand** — Higher fare rates more than compensate for the higher purchase price, producing better ROI % than equivalent UberX vehicles.

---

## How to Run

### Requirements

```bash
pip install streamlit pandas numpy plotly scikit-learn gspread google-auth
```

### Launch

```bash
cd fleetwise_sa
streamlit run app.py
```

Opens at `http://localhost:8501`

### Google Sheets feedback (optional)

Add `.streamlit/secrets.toml` with your service account credentials to enable live feedback collection. The app runs without it — the feedback tab will show an error message on submission but won't crash.

```toml
[gcp_service_account]
type = "service_account"
project_id = "..."
private_key_id = "..."
private_key = "..."
client_email = "..."

[sheets]
spreadsheet_id = "your_spreadsheet_id_here"
```

---

## Tech Stack

| Tool | Use |
|---|---|
| Python 3.10+ | Core language |
| Pandas / NumPy | Data engineering and feature construction |
| Scikit-learn | Random Forest, Gradient Boosting, Ridge Regression |
| Plotly | Interactive charts — box, scatter, bar, radar, dual-axis |
| Streamlit | Web application framework |
| gspread / google-auth | Google Sheets feedback integration |

---

## Author

**Sakiwo Sibango (NGOBE)**
ALX Africa Data Science Certification, 2024–2026
Higher Certificate in Mathematics and Statistics, UNISA

*This project was built to solve a real problem — and to demonstrate that data science is most valuable when it's applied to decisions that actually matter.*
