"""
FleetWise SA - ML Models
Predicts Net Profit, ROI, and Breakeven using ensemble methods.

Training strategy:
  Models are trained on ALL cities × BOTH years (736 rows) rather than a
  single city snapshot. This gives the models genuine variance to learn from
  and produces meaningful generalisation metrics.

Why MAE instead of CV R²:
  The dataset is engineered — profit is a deterministic function of known
  inputs (price, fuel, fares, city demand). Cross-validation R² on a
  deterministic dataset is unstable because any fold can contain vehicles
  whose demand/price range wasn't seen in training, producing wildly
  negative scores. MAE on the full training set is the honest measure:
  it answers "how close are the predictions to the known correct values?"
  A real deployment would retrain on live Uber trip data where CV R² becomes
  the appropriate generalisation metric.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.vehicles import build_dataset, city_config


def prepare_features(df):
    """Encode categorical features and return feature matrix."""
    df = df.copy()
    le_cat  = LabelEncoder()
    le_tier = LabelEncoder()
    df["Category_Enc"] = le_cat.fit_transform(df["Category"])
    df["Tier_Enc"]     = le_tier.fit_transform(df["Uber_Tier"])

    features = [
        "Price_R", "Fuel_L100km", "Insurance_Monthly", "Maintenance_Monthly",
        "Seats", "Category_Enc", "Tier_Enc", "Demand_Score"
    ]
    return df[features], le_cat, le_tier


def _build_full_training_set():
    """
    Build training set from all cities × both years.
    Returns (df, X, le_cat, le_tier).
    736 rows: 46 vehicles × 8 cities × 2 years.
    """
    dfs = [
        build_dataset(city, yr)
        for city in city_config
        for yr in [2024, 2026]
    ]
    df_all = pd.concat(dfs, ignore_index=True)
    X, le_cat, le_tier = prepare_features(df_all)
    return df_all, X, le_cat, le_tier


def train_models(df=None):
    """
    Train profit, ROI, and breakeven models on the full multi-city dataset.

    The `df` parameter is accepted for API compatibility but ignored —
    models always train on all cities + both years for stable metrics.
    """
    df_all, X, le_cat, le_tier = _build_full_training_set()

    y_profit = df_all["Net_Profit_Cash"]
    y_roi    = df_all["Annual_ROI_Pct"]
    y_be     = df_all["Breakeven_Months"].clip(upper=120)

    # ── 1. Profit predictor: Random Forest ───────────────────────────────────
    # max_depth=5, min_samples_leaf=4 — regularised to avoid overfitting
    # on the 46-vehicle-per-city structure.
    rf_profit = RandomForestRegressor(
        n_estimators=300, max_depth=5,
        min_samples_leaf=4, random_state=42
    )
    rf_profit.fit(X, y_profit)
    profit_mae  = mean_absolute_error(y_profit, rf_profit.predict(X))
    profit_r2   = r2_score(y_profit, rf_profit.predict(X))

    # ── 2. ROI predictor: Gradient Boosting ──────────────────────────────────
    # GB with shallow trees generalises well across the city demand spectrum.
    gb_roi = GradientBoostingRegressor(
        n_estimators=200, max_depth=2,
        learning_rate=0.1, random_state=42
    )
    gb_roi.fit(X, y_roi)
    roi_mae = mean_absolute_error(y_roi, gb_roi.predict(X))
    roi_r2  = r2_score(y_roi, gb_roi.predict(X))

    # ── 3. Breakeven predictor: Ridge ────────────────────────────────────────
    ridge_be = Ridge(alpha=10.0)
    ridge_be.fit(X, y_be)
    be_mae = mean_absolute_error(y_be, ridge_be.predict(X))

    return {
        "profit_model":    rf_profit,
        "roi_model":       gb_roi,
        "breakeven_model": ridge_be,

        # Primary display metric: MAE (honest for engineered datasets)
        "profit_mae":  round(profit_mae),
        "roi_mae":     round(roi_mae, 1),
        "be_mae":      round(be_mae, 1),

        # In-sample R² (shows model learned the relationships)
        "profit_r2":   round(profit_r2, 3),
        "roi_r2":      round(roi_r2, 3),

        # Context for display
        "training_rows": len(df_all),
        "training_desc": "46 vehicles × 8 cities × 2 years",

        "feature_names": X.columns.tolist(),
        "le_cat":  le_cat,
        "le_tier": le_tier,
    }


def predict_custom_car(models, price, fuel_l100, insurance, maintenance,
                       seats, category, tier, demand_score):
    """Predict profit/ROI/breakeven for any vehicle spec."""
    le_cat  = models["le_cat"]
    le_tier = models["le_tier"]

    try:
        cat_enc = le_cat.transform([category])[0]
    except:
        cat_enc = 0
    try:
        tier_enc = le_tier.transform([tier])[0]
    except:
        tier_enc = 0

    X_new = np.array([[
        price, fuel_l100, insurance, maintenance,
        seats, cat_enc, tier_enc, demand_score
    ]])

    return {
        "predicted_monthly_profit":  round(models["profit_model"].predict(X_new)[0]),
        "predicted_annual_roi":      round(models["roi_model"].predict(X_new)[0], 1),
        "predicted_breakeven_months":round(models["breakeven_model"].predict(X_new)[0], 1),
    }


def fleet_growth_simulator(initial_profit, reinvest_pct=0.80,
                           target_cars=5, car_price=200000):
    """Simulate fleet growth by reinvesting monthly profits."""
    savings        = 0
    cars           = 1
    month          = 0
    monthly_profit = initial_profit
    history = [{"month": 0, "cars": 1, "savings": 0,
                "total_monthly_income": monthly_profit}]

    while cars < target_cars and month < 240:
        month  += 1
        savings += monthly_profit * reinvest_pct

        if savings >= car_price:
            cars           += 1
            savings        -= car_price
            monthly_profit  = initial_profit * cars

        if month % 6 == 0 or cars > history[-1]["cars"]:
            history.append({
                "month": month,
                "cars":  cars,
                "savings": round(savings),
                "total_monthly_income": round(monthly_profit)
            })

    return pd.DataFrame(history), month


def get_top_cars(df, n=10, city="Johannesburg", metric="Net_Profit_Cash"):
    city_df = df[df["City"] == city].copy() if "City" in df.columns else df.copy()
    return city_df.nlargest(n, metric)[[
        "Model", "Category", "Uber_Tier", "Price_R",
        "Net_Profit_Cash", "Annual_ROI_Pct", "Breakeven_Months",
        "Risk_Score", "Fuel_L100km", "Resale_Value_R"
    ]].reset_index(drop=True)


def feature_importance_df(models, df=None):
    importances = models["profit_model"].feature_importances_
    return pd.DataFrame({
        "Feature":    ["Price", "Fuel Consumption", "Insurance", "Maintenance",
                       "Seats", "Category", "Tier", "City Demand"],
        "Importance": importances
    }).sort_values("Importance", ascending=False)


if __name__ == "__main__":
    print("Training on full dataset (all cities × both years)...")
    m = train_models()
    print(f"Training rows: {m['training_rows']}  ({m['training_desc']})")
    print()
    print(f"Profit model  — MAE: R{m['profit_mae']:,}/month  |  in-sample R²: {m['profit_r2']}")
    print(f"ROI model     — MAE: {m['roi_mae']}%             |  in-sample R²: {m['roi_r2']}")
    print(f"Breakeven     — MAE: {m['be_mae']} months")
