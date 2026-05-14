"""
FleetWise SA - ML Models
Predicts Net Profit, ROI, and Risk Score using ensemble methods
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Add parent dir for imports when run directly
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.vehicles import build_dataset, city_config


def prepare_features(df):
    """Encode categorical features and select predictors"""
    df = df.copy()
    le_cat = LabelEncoder()
    le_tier = LabelEncoder()
    df["Category_Enc"] = le_cat.fit_transform(df["Category"])
    df["Tier_Enc"] = le_tier.fit_transform(df["Uber_Tier"])
    
    features = [
        "Price_R", "Fuel_L100km", "Insurance_Monthly", "Maintenance_Monthly",
        "Seats", "Category_Enc", "Tier_Enc", "Demand_Score"
    ]
    return df[features], le_cat, le_tier


def train_models(df):
    """Train profit prediction, ROI prediction, and risk scoring models"""
    X, le_cat, le_tier = prepare_features(df)
    
    models = {}
    
    # 1. Net Profit Predictor
    y_profit = df["Net_Profit_Cash"]
    rf_profit = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    rf_profit.fit(X, y_profit)
    profit_cv = cross_val_score(rf_profit, X, y_profit, cv=5, scoring='r2').mean()
    models["profit_model"] = rf_profit
    models["profit_r2"] = profit_cv

    # 2. ROI Predictor
    y_roi = df["Annual_ROI_Pct"]
    gb_roi = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42)
    gb_roi.fit(X, y_roi)
    roi_cv = cross_val_score(gb_roi, X, y_roi, cv=5, scoring='r2').mean()
    models["roi_model"] = gb_roi
    models["roi_r2"] = roi_cv

    # 3. Breakeven Predictor
    y_be = df["Breakeven_Months"].clip(upper=120)
    ridge_be = Ridge(alpha=1.0)
    ridge_be.fit(X, y_be)
    models["breakeven_model"] = ridge_be

    models["feature_names"] = X.columns.tolist()
    models["le_cat"] = le_cat
    models["le_tier"] = le_tier

    return models


def predict_custom_car(models, price, fuel_l100, insurance, maintenance, seats, category, tier, demand_score):
    """Predict profit/ROI for a car not in the dataset"""
    le_cat = models["le_cat"]
    le_tier = models["le_tier"]
    
    try:
        cat_enc = le_cat.transform([category])[0]
    except:
        cat_enc = 0
    try:
        tier_enc = le_tier.transform([tier])[0]
    except:
        tier_enc = 0
    
    X_new = np.array([[price, fuel_l100, insurance, maintenance, seats, cat_enc, tier_enc, demand_score]])
    
    return {
        "predicted_monthly_profit": round(models["profit_model"].predict(X_new)[0]),
        "predicted_annual_roi": round(models["roi_model"].predict(X_new)[0], 1),
        "predicted_breakeven_months": round(models["breakeven_model"].predict(X_new)[0], 1),
    }


def fleet_growth_simulator(initial_profit, reinvest_pct=0.80, target_cars=5, car_price=200000):
    """
    Simulate fleet growth by reinvesting profits
    Returns months to own N cars
    """
    savings = 0
    cars = 1
    month = 0
    monthly_profit = initial_profit
    history = [{"month": 0, "cars": 1, "savings": 0, "total_monthly_income": monthly_profit}]
    
    while cars < target_cars and month < 240:
        month += 1
        savings += monthly_profit * reinvest_pct
        
        if savings >= car_price:
            cars += 1
            savings -= car_price
            monthly_profit = initial_profit * cars  # simplified: each car earns same
        
        if month % 6 == 0 or cars > history[-1]["cars"]:
            history.append({
                "month": month,
                "cars": cars,
                "savings": round(savings),
                "total_monthly_income": round(monthly_profit)
            })
    
    return pd.DataFrame(history), month


def get_top_cars(df, n=10, city="Johannesburg", metric="Net_Profit_Cash"):
    """Return top N cars ranked by metric"""
    city_df = df[df["City"] == city].copy() if "City" in df.columns else df.copy()
    return city_df.nlargest(n, metric)[[
        "Model", "Category", "Uber_Tier", "Price_R",
        "Net_Profit_Cash", "Annual_ROI_Pct", "Breakeven_Months",
        "Risk_Score", "Fuel_L100km", "Resale_Value_R"
    ]].reset_index(drop=True)


def feature_importance_df(models, df):
    """Return feature importance from Random Forest model"""
    importances = models["profit_model"].feature_importances_
    return pd.DataFrame({
        "Feature": ["Price", "Fuel Consumption", "Insurance", "Maintenance", "Seats", "Category", "Tier", "Demand"],
        "Importance": importances
    }).sort_values("Importance", ascending=False)


if __name__ == "__main__":
    from data.vehicles import build_dataset
    df = build_dataset("Johannesburg")
    models = train_models(df)
    print(f"Profit Model R²: {models['profit_r2']:.3f}")
    print(f"ROI Model R²: {models['roi_r2']:.3f}")
    print("\nTop 5 cars by profit:")
    print(get_top_cars(df, 5))