import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

#load_data
FILE_PATH = 'data/energy_consumption.csv'

df = pd.read_csv(FILE_PATH, parse_dates=['timestamp'])

df = df.sort_values(['timestamp', 'household_id']).reset_index(drop=True)
df = df.set_index('timestamp')

#plot
sample_household = df['household_id'].iloc[0]
df[df['household_id'] == sample_household]['consumption_kwh'].plot(
    style='.', figsize=(15, 5), color=color_pal[0],
    title=f'Consumption (Household: {sample_household})'
)
plt.savefig('sample_household_plot.png')
plt.show()


columns_to_drop = [c for c in ['level_0', 'index'] if c in df.columns]
if columns_to_drop:
    df = df.drop(columns=columns_to_drop)

if isinstance(df.index, pd.DatetimeIndex) and df.index.name == 'timestamp':
    df = df.reset_index()

df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(['household_id', 'timestamp']).reset_index(drop=True)


#feature_engineering
def create_features(data):
    data = data.copy()

    # Calendar features
    data['month'] = data['timestamp'].dt.month
    data['dayofmonth'] = data['timestamp'].dt.day
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)

    # Cyclical encoding for hour
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24.0)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24.0)

    # Temperature non-linear signal
    data['temp_squared'] = data['temperature'] ** 2

    # Grouped lag features (per household, avoids cross-household leakage)
    for lag in [1, 24, 168]:
        data[f'lag_{lag}'] = data.groupby('household_id')['consumption_kwh'].shift(lag)

    # Grouped rolling mean (24 hours)
    data['rolling_mean_24'] = (
        data.groupby('household_id')['consumption_kwh']
        .transform(lambda x: x.shift(1).rolling(24).mean())
    )

    return data


df_feat = create_features(df)

df_clean = df_feat.dropna().reset_index(drop=True)

df_clean['household_id'] = df_clean['household_id'].astype('category')

#data_split
split_date = pd.to_datetime('2023-11-01 00:00:00')

train = df_clean[df_clean['timestamp'] < split_date]
test = df_clean[df_clean['timestamp'] >= split_date]

FEATURES = [
    'household_id', 'temperature', 'temp_squared', 'hour', 'day_of_week',
    'month', 'is_weekend', 'hour_sin', 'hour_cos',
    'lag_1', 'lag_24', 'lag_168', 'rolling_mean_24'
]
TARGET = 'consumption_kwh'

X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

#model_train
reg = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist',
    enable_categorical=True,
    early_stopping_rounds=50,
    random_state=42
)

reg.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=100
)

#evaluation
preds = reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)

print(f"\n--- Model Performance ---")
print(f"Validation RMSE: {rmse:.4f} kWh")
print(f"Validation MAE:  {mae:.4f} kWh")
