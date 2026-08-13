# Household Energy Consumption Forecasting
 
Forecasts hourly household energy consumption (kWh) using XGBoost, with time-based lag and rolling-window features per household.
 
## Overview
 
- **Model:** XGBoost regressor (`XGBRegressor`)
- **Target:** `consumption_kwh`
- **Train/test split:** time-based — trains on Jan–Oct 2023, tests on Nov–Dec 2023
- **Key features:** lag features (1h, 24h, 168h), 24h rolling mean, cyclical hour encoding, temperature, calendar features — all computed per household to avoid data leakage across households
