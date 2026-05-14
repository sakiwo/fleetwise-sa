"""
FleetWise SA - Vehicle Dataset Generator
Supports both 2024 (baseline) and 2026 (current) market conditions.

Usage:
    build_dataset(city="Johannesburg", year=2024)
    build_dataset(city="Johannesburg", year=2026)

Year constants:
    2024: Petrol R22.50/l · Prime ~13% · vehicle finance ~15%
    2026: Petrol R26.63/l · Prime 10.25% · vehicle finance 13.25%
          (DMPR confirmed 6 May 2026; SARB/CEIC March 2026)

The same vehicle list is used for both years.
2026 prices reflect ~15% inflation on 2024 prices (CPI + import costs).
2026 fares reflect ~8% increase (fuel surcharge + CPI pass-through).
2026 insurance/maintenance reflect +10%/+9% YoY respectively.
"""

import pandas as pd
import numpy as np

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# SHARED VEHICLE LIST
# [Model, Category, Price2024(R), Price2026(R), Fuel(L/100km),
#  Insurance2024, Insurance2026, Maintenance2024, Maintenance2026,
#  Seats, UberTier]
# ─────────────────────────────────────────────────────────────────────────────
vehicles = [
    # Model                          Cat        P2024   P2026   Fuel  Ins24  Ins26  Mnt24  Mnt26  Seats  Tier
    ("Toyota Starlet 1.4 Xi",       "Hatchback",174900,199900, 5.8,  950,  1080,  670,   730,   5, "UberX"),
    ("Toyota Starlet 1.5 XS",       "Hatchback",199900,229900, 6.0,  980,  1120,  700,   760,   5, "UberX"),
    ("Suzuki Swift 1.2 GL",         "Hatchback",195900,224900, 5.1, 1050,  1150,  580,   630,   5, "UberX"),
    ("Suzuki Swift 1.2 GLX",        "Hatchback",215900,249900, 5.3, 1100,  1200,  600,   650,   5, "UberX"),
    ("Suzuki S-Presso 1.0 GL",      "Hatchback",149900,169900, 4.9,  900,   980,  520,   560,   5, "UberX"),
    ("Kia Picanto 1.0",             "Hatchback",174900,199900, 5.2,  960,  1050,  560,   610,   5, "UberX"),
    ("Kia Picanto 1.2 Style",       "Hatchback",195900,224900, 5.5, 1000,  1090,  585,   635,   5, "UberX"),
    ("Hyundai Grand i10 1.0",       "Hatchback",185900,214900, 5.5,  975,  1070,  610,   660,   5, "UberX"),
    ("Hyundai Grand i10 1.2 Motion","Hatchback",204900,234900, 5.8, 1015,  1110,  635,   685,   5, "UberX"),
    ("Renault Kwid 1.0 Techno",     "Hatchback",164900,189900, 5.6,  910,   990,  545,   590,   5, "UberX"),
    ("Datsun Go 1.2 Lux",           "Hatchback",152900,174900, 6.2,  895,   970,  530,   575,   5, "UberX"),
    ("Chery QQ 1.0",                "Hatchback",139900,159900, 5.8,  855,   930,  535,   580,   5, "UberX"),
    ("Toyota Etios 1.5 Xs Sedan",   "Sedan",    182900,209900, 6.8, 1005,  1090,  665,   720,   5, "UberX"),
    ("Suzuki Dzire 1.2 GL",         "Sedan",    225900,259900, 5.7, 1050,  1140,  615,   665,   5, "UberX"),
    ("Nissan Almera 1.5 Acenta",    "Sedan",    230900,264900, 6.6, 1160,  1260,  720,   780,   5, "UberX"),
    ("Volkswagen Polo Vivo 1.4",    "Hatchback",230900,264900, 6.9, 1160,  1260,  865,   940,   5, "UberX"),
    ("Volkswagen Polo Vivo Sedan",  "Sedan",    243900,279900, 7.0, 1205,  1310,  900,   980,   5, "UberX"),
    ("Ford Figo 1.5 Ambiente",      "Hatchback",221900,254900, 6.4, 1105,  1200,  735,   800,   5, "UberX"),
    ("Ford Figo 1.5 Titanium Sedan","Sedan",    243900,279900, 6.6, 1150,  1250,  760,   825,   5, "UberX"),
    ("Hyundai i20 1.4",             "Hatchback",278900,319900, 6.3, 1295,  1410,  800,   870,   5, "UberX"),
    ("JAC J4 1.5 Sedan",            "Sedan",    173900,199900, 7.0,  893,   970,  675,   735,   5, "UberX"),
    ("BAIC D20 1.3 Comfort",        "Sedan",    147900,169900, 6.8,  845,   920,  660,   715,   5, "UberX"),
    ("Datsun Go+ 1.2 MPV",          "MPV",      182900,209900, 6.5,  975,  1060,  570,   620,   7, "UberX"),
    ("Nissan NP200 1.6 SE",         "Bakkie",   226900,259900, 7.8, 1095,  1190,  760,   825,   2, "UberX"),
    ("Toyota Corolla Quest 1.8",    "Sedan",    295900,339900, 7.2, 1350,  1470,  810,   880,   5, "UberComfort"),
    ("Toyota Corolla Quest Plus",   "Sedan",    317900,364900, 7.2, 1395,  1520,  840,   910,   5, "UberComfort"),
    ("Toyota Corolla 1.8 Xi",       "Sedan",    365900,419900, 7.5, 1545,  1680,  898,   975,   5, "UberComfort"),
    ("Toyota Corolla Cross 1.8 XS", "SUV",      430900,494900, 5.3, 1820,  1980,  968,  1050,   5, "UberComfort"),
    ("Volkswagen Polo 1.0 TSI",     "Hatchback",347900,399900, 5.6, 1590,  1730, 1040,  1130,   5, "UberComfort"),
    ("Volkswagen Polo 1.6 Sedan",   "Sedan",    387900,444900, 7.0, 1690,  1840, 1085,  1180,   5, "UberComfort"),
    ("Honda Ballade 1.5",           "Sedan",    334900,384900, 6.8, 1500,  1630,  898,   975,   5, "UberComfort"),
    ("Mazda 2 1.5 Dynamic Sedan",   "Sedan",    326900,374900, 6.5, 1490,  1620,  948,  1030,   5, "UberComfort"),
    ("Kia Sonet 1.5",               "SUV",      321900,369900, 6.5, 1445,  1570,  838,   910,   5, "UberComfort"),
    ("Haval Jolion 1.5T",           "SUV",      391900,449900, 8.5, 1635,  1780,  948,  1030,   5, "UberComfort"),
    ("Chery Tiggo 4 Pro 1.5T",      "SUV",      287900,329900, 8.2, 1295,  1410,  810,   880,   5, "UberComfort"),
    ("Omoda C5 1.6T",               "SUV",      374900,429900, 8.0, 1545,  1680,  902,   980,   5, "UberComfort"),
    ("Jetour Dashing 1.5T",         "SUV",      313900,359900, 8.3, 1408,  1530,  828,   900,   5, "UberComfort"),
    ("Toyota Avanza 1.5 SX",        "MPV",      322900,369900, 8.5, 1395,  1520,  865,   940,   7, "UberXL"),
    ("Toyota Rush 1.5 S",           "SUV",      400900,459900, 8.8, 1590,  1730,  902,   980,   7, "UberXL"),
    ("Suzuki Ertiga 1.5 GL",        "MPV",      330900,379900, 7.2, 1340,  1460,  782,   850,   7, "UberXL"),
    ("Suzuki Ertiga 1.5 GX",        "MPV",      357900,409900, 7.2, 1388,  1510,  810,   880,   7, "UberXL"),
    ("Chery Tiggo 8 Pro 1.6T",      "SUV",      461900,529900, 9.2, 1978,  2150, 1058,  1150,   7, "UberXL"),
    ("Toyota Fortuner 2.4 GD-6",    "SUV",      704900,809900, 9.5, 2780,  3020, 1335,  1450,   7, "UberXL"),
    ("GWM P Series 2.0T",           "Bakkie",   539900,619900,10.5, 2190,  2380, 1140,  1240,   5, "UberXL"),
    ("Isuzu D-Max 1.9 DDi LS",      "Bakkie",   478900,549900,10.2, 2070,  2250, 1085,  1180,   5, "UberXL"),
    ("Toyota Hiace 2.8 GL",         "Minibus",  652900,749900,12.5, 2390,  2600, 1435,  1560,  14, "UberXL"),
]

# ─────────────────────────────────────────────────────────────────────────────
# YEAR-SPECIFIC CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
YEAR_CONFIG = {
    2024: {
        "fuel_price":      22.50,   # R/l Gauteng 95 unleaded, 2024 avg
        "financing_rate":  0.1500,  # prime ~13% + 2% = 15%
        "uber_fares": {
            "UberX":       {"base_fare": 12.0, "per_km": 8.5,  "per_min": 0.95,
                            "avg_trip_km": 9,  "trips_per_day": 14, "demand_score": 0.90},
            "UberComfort": {"base_fare": 18.0, "per_km": 11.5, "per_min": 1.20,
                            "avg_trip_km": 11, "trips_per_day": 12, "demand_score": 0.80},
            "UberXL":      {"base_fare": 22.0, "per_km": 13.5, "per_min": 1.45,
                            "avg_trip_km": 14, "trips_per_day": 10, "demand_score": 0.70},
        },
        "label": "2024 Baseline  (R22.50/l · Prime 13%)",
    },
    2026: {
        "fuel_price":      26.63,   # R/l Gauteng 95, DMPR confirmed 6 May 2026
        "financing_rate":  0.1325,  # prime 10.25% + 3% = 13.25%
        "uber_fares": {
            "UberX":       {"base_fare": 13.0, "per_km": 9.2,  "per_min": 1.02,
                            "avg_trip_km": 9,  "trips_per_day": 14, "demand_score": 0.90},
            "UberComfort": {"base_fare": 19.5, "per_km": 12.4, "per_min": 1.30,
                            "avg_trip_km": 11, "trips_per_day": 12, "demand_score": 0.80},
            "UberXL":      {"base_fare": 24.0, "per_km": 14.6, "per_min": 1.56,
                            "avg_trip_km": 14, "trips_per_day": 10, "demand_score": 0.70},
        },
        "label": "2026 Current  (R26.63/l · Prime 10.25%)",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# SHARED CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
city_config = {
    "Johannesburg":   1.20,
    "Cape Town":      1.15,
    "Pretoria":       1.10,
    "Durban":         1.08,
    "Port Elizabeth": 0.90,
    "Kimberley":      0.75,
    "Bloemfontein":   0.80,
    "East London":    0.82,
}

UBER_COMMISSION  = 0.25
WORKING_DAYS     = 22
KM_PER_DAY       = 180
LOAN_TERM_MONTHS = 60
DEPOSIT_PERCENT  = 0.10

DEPRECIATION = {
    "Hatchback": 0.43, "Sedan": 0.41, "MPV": 0.48,
    "SUV": 0.49, "Bakkie": 0.34, "Minibus": 0.55,
}

# Price index offset: column indices in `vehicles` tuple
_PRICE_IDX  = {2024: 2, 2026: 3}
_INS_IDX    = {2024: 4, 2026: 5}
_MAINT_IDX  = {2024: 6, 2026: 7}


def _monthly_payment(price, rate, term=LOAN_TERM_MONTHS, deposit=DEPOSIT_PERCENT):
    principal = price * (1 - deposit)
    r = rate / 12
    return principal * (r * (1 + r) ** term) / ((1 + r) ** term - 1)


def build_dataset(city="Johannesburg", year=2026):
    if year not in YEAR_CONFIG:
        raise ValueError(f"year must be 2024 or 2026, got {year}")

    cfg         = YEAR_CONFIG[year]
    fuel_price  = cfg["fuel_price"]
    fin_rate    = cfg["financing_rate"]
    tier_config = cfg["uber_fares"]
    demand_mult = city_config.get(city, 1.0)

    p_idx = _PRICE_IDX[year]
    i_idx = _INS_IDX[year]
    m_idx = _MAINT_IDX[year]

    rows = []
    for car in vehicles:
        model     = car[0]
        category  = car[1]
        price     = car[p_idx]
        fuel_l100 = car[4]   # always index 4 — shared across years
        insurance = car[i_idx]
        maint     = car[m_idx]
        seats     = car[9]
        tier      = car[10]

        tc = tier_config[tier]

        # Revenue
        trips_day       = tc["trips_per_day"] * demand_mult
        trip_rev        = tc["base_fare"] + tc["avg_trip_km"] * tc["per_km"] + (tc["avg_trip_km"] / 30 * tc["per_min"])
        monthly_revenue = trips_day * WORKING_DAYS * trip_rev * (1 - UBER_COMMISSION)

        # Costs
        fuel_cost   = KM_PER_DAY * WORKING_DAYS * (fuel_l100 / 100) * fuel_price
        total_costs = fuel_cost + insurance + maint

        # Profit
        net_cash     = monthly_revenue - total_costs
        loan_pmt     = _monthly_payment(price, fin_rate)
        net_financed = net_cash - loan_pmt

        # ROI & breakeven
        roi_cash         = (net_cash * 12) / price * 100
        breakeven_months = price / net_cash if net_cash > 0 else 999

        # Risk (fuel weight higher in 2026)
        fuel_w  = 0.45 if year == 2026 else 0.40
        maint_w = 1.0 - fuel_w - 0.25
        risk_score = round(
            (fuel_l100 / 12 * fuel_w + maint / 1600 * maint_w + (1 - tc["demand_score"]) * 0.25) * 10, 2
        )

        resale = price * (1 - DEPRECIATION.get(category, 0.44))

        rows.append({
            "Model":                model,
            "Category":             category,
            "Price_R":              price,
            "Uber_Tier":            tier,
            "Seats":                seats,
            "Fuel_L100km":          fuel_l100,
            "Insurance_Monthly":    insurance,
            "Maintenance_Monthly":  maint,
            "Fuel_Cost_Monthly":    round(fuel_cost),
            "Monthly_Revenue":      round(monthly_revenue),
            "Total_Costs_Monthly":  round(total_costs),
            "Net_Profit_Cash":      round(net_cash),
            "Loan_Payment_Monthly": round(loan_pmt),
            "Net_Profit_Financed":  round(net_financed),
            "Annual_ROI_Pct":       round(roi_cash, 1),
            "Breakeven_Months":     round(breakeven_months, 1) if breakeven_months < 999 else 999,
            "Resale_Value_R":       round(resale),
            "Risk_Score":           risk_score,
            "Demand_Score":         round(tc["demand_score"] * demand_mult, 2),
            "City":                 city,
            "Year":                 year,
            "Fuel_Price_Used":      fuel_price,
            "Finance_Rate_Pct":     round(fin_rate * 100, 2),
        })

    return pd.DataFrame(rows)


def build_comparison(city="Johannesburg"):
    """Return both years merged with a 'Year' column — useful for cross-year charts."""
    df24 = build_dataset(city, 2024)
    df26 = build_dataset(city, 2026)
    return pd.concat([df24, df26], ignore_index=True)


def profit_delta(city="Johannesburg"):
    """
    Per-vehicle profit change between 2024 and 2026.
    Returns a DataFrame with columns: Model, Profit_2024, Profit_2026,
    Profit_Delta, Delta_Pct, Crossed_To_Loss.
    """
    df24 = build_dataset(city, 2024)[["Model", "Net_Profit_Cash"]].rename(
        columns={"Net_Profit_Cash": "Profit_2024"}
    )
    df26 = build_dataset(city, 2026)[["Model", "Net_Profit_Cash"]].rename(
        columns={"Net_Profit_Cash": "Profit_2026"}
    )
    merged = df24.merge(df26, on="Model")
    merged["Profit_Delta"]    = merged["Profit_2026"] - merged["Profit_2024"]
    merged["Delta_Pct"]       = (merged["Profit_Delta"] / merged["Profit_2024"].abs() * 100).round(1)
    merged["Crossed_To_Loss"] = (merged["Profit_2024"] > 0) & (merged["Profit_2026"] <= 0)
    return merged.sort_values("Profit_Delta")


if __name__ == "__main__":
    for yr in [2024, 2026]:
        df = build_dataset("Johannesburg", yr)
        cfg = YEAR_CONFIG[yr]
        print(f"\n{'='*60}")
        print(f"  {cfg['label']}")
        print(f"{'='*60}")
        print(f"  Fuel: R{cfg['fuel_price']}/l  |  Finance: {cfg['financing_rate']*100:.2f}%")
        print(f"  Avg fuel cost/month: R{df['Fuel_Cost_Monthly'].mean():,.0f}")
        print(f"  Profitable (cash): {(df['Net_Profit_Cash'] > 0).sum()}/{len(df)}")
        print(f"  Profitable (financed): {(df['Net_Profit_Financed'] > 0).sum()}/{len(df)}")

    print(f"\n{'='*60}")
    print("  Vehicles that crossed into LOSS (2024 profitable → 2026 loss):")
    delta = profit_delta("Johannesburg")
    losses = delta[delta["Crossed_To_Loss"]]
    if losses.empty:
        print("  None — all vehicles remained profitable.")
    else:
        print(losses[["Model", "Profit_2024", "Profit_2026", "Profit_Delta"]].to_string(index=False))

    print(f"\n  Top 5 most resilient (smallest profit drop):")
    print(delta.tail(5)[["Model", "Profit_2024", "Profit_2026", "Profit_Delta"]].to_string(index=False))