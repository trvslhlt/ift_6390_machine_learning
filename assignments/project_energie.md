# Project Energie

## Notes

Ranking features by coefficient weights

```python
###!!! rank importance of features
ridge_coeffs = model_ridge.coef_.copy()

# standardize coefficients by multiplying by feature std
feature_stds = np.std(X_train_reg, axis=0)
ridge_coeffs_std = ridge_coeffs * feature_stds
feature_importance = pd.DataFrame({
    'characteristic': features_disponibles,
    'standardized_coefficient': ridge_coeffs_std
}).sort_values('standardized_coefficient', key=lambda x: np.abs(x), ascending=False)
feature_importance['rank'] = range(1, len(feature_importance) + 1)
print("\nCharacteristic importance (Ridge):")
print(feature_importance.to_string(index=False))
```

Searching for the sources of error in the model

```python
# make preditions on trining data
y_train_pred_final = model_final.predict(X_train_final)

# find the N data points with the largest absolute residuals
train_residuals = y_train_final - y_train_pred_final
train_residuals_abs = np.abs(train_residuals)

# add residuals to the dataframe
analysis_df = train_eng.copy()
analysis_df.insert(loc=0, column='residual', value=train_residuals)
analysis_df.insert(loc=1, column='residual_abs', value=train_residuals_abs)

# add the residuals to the training data for analysis
indices_sorted = np.argsort(analysis_df['residual_abs'].values) 
error_indices = indices_sorted[-N:]
remaining_indices = indices_sorted[:-N]
analysis_df_errors = analysis_df.iloc[error_indices]
analysis_df_remaining = analysis_df.iloc[remaining_indices]

# sort by largest residual
analysis_df = analysis_df.sort_values(by='residual_abs', ascending=False)

# Are errors over-represented in certain categories
# look for patterns in the features of these high-residual points
# For each numeric feature, compare means
for col in analysis_df.select_dtypes(include='number').columns:
    error_mean = analysis_df_errors[col].mean()
    remaining_mean = analysis_df_remaining[col].mean()
    diff_pct = 100 * (error_mean - remaining_mean) / (remaining_mean + 1e-8)
    if abs(diff_pct) > 20:  # flag large differences
        print(f"{col}: {diff_pct:+.1f}% difference")
print("\n")

for col in analysis_df.select_dtypes(exclude='number').columns:
    print(f"\n{col}:")
    print("Errors:", analysis_df_errors[col].value_counts(normalize=True))
    print("Remaining:", analysis_df[col].value_counts(normalize=True))
    print("\n")
print("\n")

# for col in ['poste', 'mois', 'heure', 'est_weekend', 'evenement_pointe']:
#     print(f"\n{col}:")
#     print("Errors:", analysis_df_errors[col].value_counts(normalize=True))
#     print("Remaining:", analysis_df[col].value_counts(normalize=True))
#     print("\n")


# print(type(df))
# display(df, rows=min(N, 100))

```

Feature tests

```python
###!!!
class CustomFeatures:
    energy_capped = 'energy_capped'
    energy_lag_1 = 'energy_lag_1'
    energy_rolling_6h = 'energy_rolling_6h'
    energy_rolling_12h = 'energy_rolling_12h'
    energy_rolling_24h = 'energy_rolling_24h'
    temp_lag_1 = 'temp_lag_1'
    temp_rolling_6h = 'temp_rolling_6h'
    temp_rolling_12h = 'temp_rolling_12h'
    temp_rolling_24h = 'temp_rolling_24h'
    irradiance_lag_1 = 'irradiance_lag_1'
    irradiance_rolling_6h = 'irradiance_rolling_6h'
    irradiance_rolling_12h = 'irradiance_rolling_12h'
    irradiance_rolling_24h = 'irradiance_rolling_24h'
    humidity_lag_1 = 'humidity_lag_1'
    humidity_rolling_6h = 'humidity_rolling_6h'
    humidity_rolling_12h = 'humidity_rolling_12h'
    humidity_rolling_24h = 'humidity_rolling_24h'
    wind_lag_1 = 'wind_lag_1'
    wind_rolling_6h = 'wind_rolling_6h'
    wind_rolling_12h = 'wind_rolling_12h'
    wind_rolling_24h = 'wind_rolling_24h'

    energy_client_connetion_interaction = 'energy_client_connection_interaction'
    temp_hour_interaction = 'temp_hour_interaction'
    temp_hour_sin_interaction = 'temp_hour_sin_interaction'
    temp_hour_cos_interaction = 'temp_hour_cos_interaction'
    temp_humidity_interaction = 'temp_humidity_interaction'
    temp_wind_interaction = 'temp_wind_interaction'
    temp_irradiance_interaction = 'temp_irradiance_interaction'

    degrees_heating = 'degrees_heating'
    degrees_heating_rolling_6h = 'degrees_heating_rolling_6h'
    degrees_heating_rolling_12h = 'degrees_heating_rolling_12h'
    degrees_heating_rolling_24h = 'degrees_heating_rolling_24h'
    degrees_cooling = 'degrees_cooling'
    degrees_cooling_rolling_6h = 'degrees_cooling_rolling_6h'
    degrees_cooling_rolling_12h = 'degrees_cooling_rolling_12h'
    degrees_cooling_rolling_24h = 'degrees_cooling_rolling_24h'

    temp_ext_st = 'temp_ext_st'


def create_custom_features(df: pd.DataFrame):
    energy_cap_kwh = 800
    heating_temp = 15
    cooling_temp = 22

    energy = df['energie_kwh']
    temp = df['temperature_ext']
    irradiance = df['irradiance_solaire']
    humidity = df['humidite']
    wind = df['vitesse_vent']
    client_connect = df['clients_connectes']

    energy_capped = energy.where(np.abs(energy) < energy_cap_kwh, energy_cap_kwh)
    energy_lag_1 = energy.shift(1).fillna(method='bfill')
    client_connect_lag_1 = client_connect.shift(1).fillna(method='bfill')
    temp_rolling_6h = temp.rolling(window=6).mean().fillna(method='bfill')
    temp_rolling_12h = temp.rolling(window=12).mean().fillna(method='bfill')
    temp_rolling_24h = temp.rolling(window=24).mean().fillna(method='bfill')

    return {
        CustomFeatures.energy_capped: energy_capped,
        CustomFeatures.energy_lag_1: energy_lag_1 / client_connect_lag_1,
        CustomFeatures.energy_rolling_6h: energy_capped.rolling(window=6).mean().fillna(method='bfill'),
        CustomFeatures.energy_rolling_12h: energy_capped.rolling(window=12).mean().fillna(method='bfill'),
        CustomFeatures.energy_rolling_24h: energy_capped.rolling(window=24).mean().fillna(method='bfill'),
        CustomFeatures.temp_lag_1: temp.shift(1).fillna(method='bfill'),
        CustomFeatures.temp_rolling_6h: temp.rolling(window=6).mean().fillna(method='bfill'),
        CustomFeatures.temp_rolling_12h: temp_rolling_12h,
        CustomFeatures.temp_rolling_24h: temp.rolling(window=24).mean().fillna(method='bfill'),
        CustomFeatures.irradiance_lag_1: irradiance.shift(1).fillna(method='bfill'),
        CustomFeatures.irradiance_rolling_6h: irradiance.rolling(window=6).mean().fillna(method='bfill'),
        CustomFeatures.irradiance_rolling_12h: irradiance.rolling(window=12).mean().fillna(method='bfill'),
        CustomFeatures.irradiance_rolling_24h: irradiance.rolling(window=24).mean().fillna(method='bfill'),
        CustomFeatures.humidity_lag_1: humidity.shift(1).fillna(method='bfill'),
        CustomFeatures.humidity_rolling_6h: humidity.rolling(window=6).mean().fillna(method='bfill'), 
        CustomFeatures.humidity_rolling_12h: humidity.rolling(window=12).mean().fillna(method='bfill'),
        CustomFeatures.humidity_rolling_24h: humidity.rolling(window=24).mean().fillna(method='bfill'),
        CustomFeatures.wind_lag_1: wind.shift(1).fillna(method='bfill'),
        CustomFeatures.wind_rolling_6h: wind.rolling(window=6).mean().fillna(method='bfill'),
        CustomFeatures.wind_rolling_12h: wind.rolling(window=12).mean().fillna(method='bfill'),
        CustomFeatures.wind_rolling_24h: wind.rolling(window=24).mean().fillna(method='bfill'),

        CustomFeatures.energy_client_connetion_interaction: df['energie_kwh'] / df['clients_connectes'],
        CustomFeatures.temp_hour_interaction: temp * df['heure'],
        CustomFeatures.temp_hour_sin_interaction: temp * df['heure_sin'] + 1,
        CustomFeatures.temp_hour_cos_interaction: temp * df['heure_cos'] + 1,
        CustomFeatures.temp_humidity_interaction: np.abs(temp - 18) * humidity,
        CustomFeatures.temp_wind_interaction: temp * wind,
        CustomFeatures.temp_irradiance_interaction: temp * irradiance,

        CustomFeatures.degrees_heating: np.maximum(heating_temp - temp, 0),
        CustomFeatures.degrees_heating_rolling_6h: np.maximum(heating_temp - temp_rolling_6h, 0),
        CustomFeatures.degrees_heating_rolling_12h: np.maximum(heating_temp - temp_rolling_12h, 0),
        CustomFeatures.degrees_heating_rolling_24h: np.maximum(heating_temp - temp_rolling_24h, 0),
        CustomFeatures.degrees_cooling: np.maximum(temp - cooling_temp, 0),
        CustomFeatures.degrees_cooling_rolling_6h: np.maximum(temp_rolling_6h - cooling_temp, 0),
        CustomFeatures.degrees_cooling_rolling_12h: np.maximum(temp_rolling_12h - cooling_temp, 0),
        CustomFeatures.degrees_cooling_rolling_24h: np.maximum(temp_rolling_24h - cooling_temp, 0),
    }

active_custom_features: list[str] = [
    # CustomFeatures.energy_capped,
    CustomFeatures.energy_lag_1,
    CustomFeatures.energy_rolling_6h,
    CustomFeatures.energy_rolling_12h,
    CustomFeatures.energy_rolling_24h,
    # CustomFeatures.humidity_lag_1,
    # CustomFeatures.humidity_rolling_6h,
    # CustomFeatures.humidity_rolling_12h,
    # CustomFeatures.humidity_rolling_24h,
    CustomFeatures.temp_lag_1,
    # CustomFeatures.temp_rolling_6h,
    CustomFeatures.temp_rolling_12h,
    # CustomFeatures.temp_rolling_24h,
    # CustomFeatures.irradiance_lag_1,
    # CustomFeatures.irradiance_rolling_6h,
    # CustomFeatures.irradiance_rolling_12h,
    CustomFeatures.irradiance_rolling_24h,

    # CustomFeatures.energy_client_connetion_interaction,
    CustomFeatures.temp_humidity_interaction,
    CustomFeatures.temp_wind_interaction,
    CustomFeatures.temp_irradiance_interaction,

    CustomFeatures.degrees_heating,
    # CustomFeatures.degrees_heating_rolling_6h,
    # CustomFeatures.degrees_heating_rolling_12h,
    # CustomFeatures.degrees_heating_rolling_24h,
    # CustomFeatures.degrees_cooling,
    CustomFeatures.degrees_cooling_rolling_6h,
    # CustomFeatures.degrees_cooling_rolling_12h,
    # CustomFeatures.degrees_cooling_rolling_24h,
]
```

Correclation matrix

```python
###!!!
# drop features
# print(train_eng['poste'].unique())
# corr_df = train_eng.drop(columns=[
#     'horodatage_local',
#     'jour_semaine_cos',
#     'jour_semaine_sin',
#     'heure_sin',
#     'heure_cos',
#     'est_weekend',
#     'est_ferie',
# ])
# corr_df['poste_num'] = corr_df['poste'].map({
#     'A': 0,
#     'B': 1,
#     'C': 2,
# })
# corr_df.head()


# Create correlation matrix for training data
corr_df = analysis_df.copy()
corr_df['poste_num'] = corr_df['poste'].map({
    'A': 0,
    'B': 1,
    'C': 2,
})
corr_df = corr_df.select_dtypes(include='number')
corr_matrix = corr_df.corr()

plt.figure(figsize=(12, 10))
plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar(label='Correlation')
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title('Correlation Matrix')
plt.tight_layout()

# corr_df.head(n=50)
# train_eng[['clients_connectes', 'poste']].head(n=100)
# train_eng[train_eng['poste'] == 'B'][['clients_connectes', 'poste']].head(n=500)
```

Postes

```python
###!!!
fig, axes = plt.subplots(1, 1, figsize=(12, 10))

# Consommation vs température

def _get_y(df):
    # return df['energie_kwh'] / df['clients_connectes']
    return df['energie_kwh'] / df['tstats_intelligents_connectes']

a = train[train['poste'] == 'A'][train['energie_kwh'] < 1000]
b = train[train['poste'] == 'B'][train['energie_kwh'] < 1000]
c = train[train['poste'] == 'C'][train['energie_kwh'] < 1000]


axes.scatter(c['temperature_ext'], _get_y(c), alpha=0.3, s=5, color='lightgreen')
axes.scatter(a['temperature_ext'], _get_y(a), alpha=0.3, s=5, color='red')
axes.scatter(b['temperature_ext'], _get_y(b), alpha=0.3, s=5, color='blue')

```

Plotting features against energy consumption

```python
import math

analysis_df = train_eng.copy()
analysis_df = analysis_df.drop(columns=[
    'heure',
    'jour',
])

analysis_df = analysis_df[analysis_df['energie_kwh'] < 1000]
energy_feat = analysis_df['energie_kwh']
numeric_features = analysis_df.select_dtypes(include='number').columns.to_list()
features = numeric_features
plt_column_n = 3
plt_row_n = math.ceil(len(features) / plt_column_n)
fig, axes = plt.subplots(plt_row_n, plt_column_n, figsize=(12, 30))

for idx, feature in enumerate(numeric_features):
    r = idx // plt_column_n
    c = idx % plt_column_n
    ax = axes[r, c]
    ax.scatter(analysis_df[feature], energy_feat, alpha=0.2, s=5)
    ax.set_title(feature)

plt.tight_layout()
```