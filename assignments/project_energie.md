# Project Energie

## Notes

### Baseline: no custom features

```txt
OLS (baseline):
  R² train: 0.4161
  R² test:  -0.1887
  RMSE test: 77.1386

Ridge avec scaling (λ=100.0):
  R² train: 0.4159
  R² test:  -0.1825
  RMSE test: 76.9382

Comparaison des coefficients (triés par réduction):
       Caractéristique        OLS      Ridge  Réduction (%)
          is_peak_hour  21.231722  10.416325   5.093980e+01
              mois_sin  19.079766  13.144773   3.110622e+01
      jour_semaine_cos  -1.660517  -1.161708   3.003936e+01
             heure_cos -45.300469 -31.739168   2.993634e+01
             heure_sin -37.446491 -26.792259   2.845188e+01
      jour_semaine_sin  -0.296764  -0.212370   2.843785e+01
              mois_cos   5.687246   4.836384   1.496088e+01
          vitesse_vent  -0.641622  -0.915100  -4.262294e+01
       poste_c_cooling   1.153395   3.083910  -1.673768e+02
       poste_a_heating   1.732259   8.128558  -3.692462e+02
       poste_b_heating  -1.922117 -12.548746  -5.528608e+02
       poste_c_heating   4.219950  38.632671  -8.154770e+02
          temp_heating   4.030092  37.193649  -8.228982e+02
       temperature_ext   6.712729  69.355874  -9.331994e+02
      temp_rolling_24h  -3.690085 -43.502033  -1.078890e+03
              humidite   0.320823   5.706802  -1.678798e+03
        conns_trend_1h  -0.119000  -2.932391  -2.364191e+03
          conns_lag_1h   1.164783  34.534251  -2.864865e+03
     clients_connectes   1.045783  32.432631  -3.001277e+03
             poste_enc   0.139873   9.382602  -6.607959e+03
irradiance_rolling_24h   0.021709   2.333023  -1.064672e+04
       temp_heating_sq  -0.004578   0.781625  -1.697204e+04
    irradiance_solaire  -0.092834 -20.695154  -2.219265e+04
     conn_heating_load   0.002537  62.073264  -2.446154e+06
```

## Code Snippets

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

Visualize residuals
```python
import math

RES_VIS_T = 5000
# RES_ERROR_T = 100

# make preditions on trining data
y_train_pred_final = model_final.predict(X_train_final)

# find the N data points with the largest absolute residuals
train_residuals = y_train_final - y_train_pred_final
train_residuals_abs = np.abs(train_residuals)

# add residuals to the dataframe
analysis_df = train_eng.copy()
analysis_df.insert(loc=0, column='residual', value=train_residuals)
analysis_df.insert(loc=1, column='residual_abs', value=train_residuals_abs)
analysis_df = analysis_df[analysis_df['residual'].abs() < RES_VIS_T]

# sort by largest residual
analysis_df = analysis_df.sort_values(by='residual_abs', ascending=False)

# plot features aginst energy and residual
numeric_features = analysis_df.select_dtypes(include='number').columns.to_list()
non_numeric_features = []
features = numeric_features + non_numeric_features
residual_feature = 'residual'
plt_column_n = 3
plt_row_n = math.ceil(len(features) / plt_column_n)
fig, axes = plt.subplots(plt_row_n, plt_column_n, figsize=(12, 50))

for idx, feature in enumerate(numeric_features):
    r = idx // plt_column_n
    c = idx % plt_column_n
    ax = axes[r, c]
    ax.scatter(analysis_df[feature], analysis_df[CustomFeatures.energy_capped], alpha=0.3, s=5)
    ax2 = ax.twinx() 
    ax2.scatter(analysis_df[feature], analysis_df[residual_feature], alpha=0.3, s=5, color='red')
    ax2.axhline(0, color='darkred', linestyle='-', linewidth=1)
    
    
    ax.set_title(feature)

plt.tight_layout()
```

Error Analysis (Part 7: Option C)

```python
# Option C: In-depth Error Analysis
# Goal: Identify when the model makes the most errors and propose improvements

print("=" * 60)
print("OPTION C: IN-DEPTH ERROR ANALYSIS")
print("=" * 60)

# 1. Calculate residuals on test data
residuals = y_test_final - y_pred_final
residuals_abs = np.abs(residuals)

# Add residuals to test dataframe for analysis
error_analysis = test_eng.copy()
error_analysis['residual'] = residuals
error_analysis['residual_abs'] = residuals_abs
error_analysis['y_pred'] = y_pred_final

# 2. Basic error statistics
print("\n📊 ERROR STATISTICS")
print("-" * 40)
print(f"RMSE: {np.sqrt(np.mean(residuals**2)):.2f} kWh")
print(f"MAE:  {np.mean(residuals_abs):.2f} kWh")
print(f"Median error: {np.median(residuals_abs):.2f} kWh")
print(f"Max under-prediction: {residuals.min():.2f} kWh")
print(f"Max over-prediction: {residuals.max():.2f} kWh")

# 3. Identify high-error observations (top 10%)
error_threshold = np.percentile(residuals_abs, 90)
high_error = error_analysis[error_analysis['residual_abs'] > error_threshold]
low_error = error_analysis[error_analysis['residual_abs'] <= error_threshold]

print(f"\n🔍 HIGH-ERROR OBSERVATIONS (top 10%)")
print("-" * 40)
print(f"Error threshold (90th percentile): {error_threshold:.2f} kWh")
print(f"Number of observations: {len(high_error)}")





# 4. Analyze patterns in high-error observations
print("\n📈 PATTERNS IN HIGH-ERROR PREDICTIONS")
print("-" * 40)

# Compare categorical distributions
categorical_cols = ['poste', 'heure', 'mois', 'est_weekend', 'evenement_pointe']

for col in categorical_cols:
    high_dist = high_error[col].value_counts(normalize=True)
    all_dist = error_analysis[col].value_counts(normalize=True)
    
    print(f"\n{col.upper()}:")
    for val in high_dist.index[:5]:  # Top 5 values
        high_pct = high_dist.get(val, 0) * 100
        all_pct = all_dist.get(val, 0) * 100
        diff = high_pct - all_pct
        if abs(diff) > 5:  # Only show significant differences
            arrow = "↑" if diff > 0 else "↓"
            print(f"  {val}: {high_pct:.1f}% (vs {all_pct:.1f}% overall) {arrow}")



# 5. Numeric feature comparison: high-error vs low-error
print("\n📉 NUMERIC FEATURE COMPARISON")
print("-" * 40)

numeric_cols = ['temperature_ext', 'humidite', 'vitesse_vent', 'irradiance_solaire', 
                'clients_connectes', 'energie_kwh', 'P_pointe']

comparison_data = []
for col in numeric_cols:
    if col in error_analysis.columns:
        high_mean = high_error[col].mean()
        low_mean = low_error[col].mean()
        diff_pct = 100 * (high_mean - low_mean) / (low_mean + 1e-8)
        comparison_data.append({
            'Feature': col,
            'Mean (high error)': high_mean,
            'Mean (low error)': low_mean,
            'Difference (%)': diff_pct
        })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('Difference (%)', key=lambda x: np.abs(x), ascending=False)
print(comparison_df.to_string(index=False))


# 6. Visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 6a. Error by hour
hourly_error = error_analysis.groupby('heure')['residual_abs'].mean()
axes[0, 0].bar(hourly_error.index, hourly_error.values, color='steelblue')
axes[0, 0].set_xlabel('Hour')
axes[0, 0].set_ylabel('Mean Absolute Error (kWh)')
axes[0, 0].set_title('Error by Hour')
axes[0, 0].axhline(residuals_abs.mean(), color='red', linestyle='--', label='Mean')
axes[0, 0].legend()

# 6b. Error by poste
poste_error = error_analysis.groupby('poste')['residual_abs'].mean()
axes[0, 1].bar(poste_error.index, poste_error.values, color='coral')
axes[0, 1].set_xlabel('Poste')
axes[0, 1].set_ylabel('Mean Absolute Error (kWh)')
axes[0, 1].set_title('Error by Poste')

# 6c. Error vs Temperature
axes[0, 2].scatter(error_analysis['temperature_ext'], error_analysis['residual_abs'], 
                   alpha=0.3, s=10)
axes[0, 2].set_xlabel('Temperature (°C)')
axes[0, 2].set_ylabel('Absolute Error (kWh)')
axes[0, 2].set_title('Error vs Temperature')

# 6d. Error vs Actual energy
axes[1, 0].scatter(error_analysis['energie_kwh'], error_analysis['residual_abs'], 
                   alpha=0.3, s=10)
axes[1, 0].set_xlabel('Actual Energy (kWh)')
axes[1, 0].set_ylabel('Absolute Error (kWh)')
axes[1, 0].set_title('Error vs Actual Consumption')

# 6e. Residual distribution by poste
for poste in ['A', 'B', 'C']:
    poste_residuals = error_analysis[error_analysis['poste'] == poste]['residual']
    axes[1, 1].hist(poste_residuals, bins=30, alpha=0.5, label=f'Poste {poste}')
axes[1, 1].set_xlabel('Residual (kWh)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Residual Distribution by Poste')
axes[1, 1].legend()

# 6f. Error by month
monthly_error = error_analysis.groupby('mois')['residual_abs'].mean()
axes[1, 2].bar(monthly_error.index, monthly_error.values, color='green')
axes[1, 2].set_xlabel('Month')
axes[1, 2].set_ylabel('Mean Absolute Error (kWh)')
axes[1, 2].set_title('Error by Month')

plt.tight_layout()


# 7. Conclusions and improvement proposals
print("\n" + "=" * 60)
print("📋 CONCLUSIONS AND IMPROVEMENT PROPOSALS")
print("=" * 60)

# Calculate key insights
poste_c_error = error_analysis[error_analysis['poste'] == 'C']['residual_abs'].mean()
poste_a_error = error_analysis[error_analysis['poste'] == 'A']['residual_abs'].mean()
peak_hour_error = error_analysis[error_analysis['is_peak_hour'] == 1]['residual_abs'].mean()
off_peak_error = error_analysis[error_analysis['is_peak_hour'] == 0]['residual_abs'].mean()

print("\n🔍 KEY OBSERVATIONS:")
print(f"  1. Poste C has {poste_c_error/poste_a_error:.1f}x more error than Poste A")
print(f"  2. Peak hour error: {peak_hour_error:.1f} kWh vs {off_peak_error:.1f} kWh (off-peak)")
print(f"  3. High-consumption observations have the largest errors")

print("\n💡 IMPROVEMENT PROPOSALS:")
print("""
  1. MODEL PER POSTE: Train a separate Ridge model for each poste
     → The postes have very different behaviors
     
  2. NON-LINEAR FEATURES: Add polynomial terms
     → The temperature-consumption relationship is not linear
     
  3. INTERACTIONS: Create poste × temperature features
     → Each poste reacts differently to weather conditions
     
  4. TARGET TRANSFORMATION: Use log(energie_kwh)
     → The distribution is skewed, log transform can help
     
  5. WIDER TEMPORAL WINDOWS: 24-hour or 7-day averages
     → Capture seasonal trends
""")

# Justification for Option C
print("\n" + "=" * 60)
print("Option chosen: C - In-depth Error Analysis")
print("Justification: This analysis allows understanding the model's")
print("weaknesses and proposes concrete improvements based on data")
print("rather than assumptions.")
print("=" * 60)

```

Overfitting/underfitting

```python
# Learning curves to diagnose overfitting/underfitting
from sklearn.model_selection import learning_curve

# Use return_times=False for compatibility across sklearn versions
train_sizes, train_scores, val_scores = learning_curve(
    Ridge(alpha=model_ridge.alpha_),
    X_train_scaled, y_train_reg,
    cv=TimeSeriesSplit(n_splits=5),
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='r2',
    return_times=False
)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Train R²')
plt.plot(train_sizes, val_scores.mean(axis=1), 'o-', label='Validation R²')
plt.fill_between(train_sizes,
                 train_scores.mean(axis=1) - train_scores.std(axis=1),
                 train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1)
plt.fill_between(train_sizes,
                 val_scores.mean(axis=1) - val_scores.std(axis=1),
                 val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1)
plt.xlabel('Training Set Size')
plt.ylabel('R² Score')
plt.legend()
plt.title('Learning Curves - Ridge Regression')
plt.grid(True, alpha=0.3)
plt.tight_layout()
```

Late stage feature graveyard

```python
from enum import StrEnum, auto

class EngFeature(StrEnum):
    # energy
    energy_lag = auto()
    energy_rolling = auto()
    # TODO try capping based on pct increase from previous

    # energy proxies
    # conns_lag_1h = auto()
    conns_scaled = auto()
    # conns_trend_1h = auto()
    
    # # temperature
    # temp_rolling_avg = auto()
    # temp_heating = auto()
    temp_heating_pow = auto()
    # temp_cooling = auto()
    
    # # time
    # is_peak_hour = auto()

    # # sunlight
    # irradiance_rolling_24h = auto()
    
    # # "poste" encoding
    # poste_a = auto()
    # poste_b = auto()
    poste_c = auto()
    # poste_enc = auto()

    # # interactions
    # energy_per_conn_lag_1h = auto()
    # conn_heating_load = auto()
    # poste_a_heating = auto()
    # poste_b_heating = auto()
    # poste_c_heating = auto()
    # poste_c_cooling = auto()
    # temp_rolling_avg_poste = auto()
    # TODO wind * heating
    temp_heating_wind = auto()
    # temp_rolling_avg_cos = auto()

    @classmethod
    def values(cls) -> list[str]:
        return [member.value for member in cls]


def create_custom_features(df: pd.DataFrame):
    # # Poste-specific energy means (from training data, target encoding)
    # poste_energy_means = {
    #     'A': 82.727205,
    #     'B': 129.809350,
    #     'C': 259.096250,
    # }
    conns = df['clients_connectes']
    # conns_lag_1h = conns.shift(1).bfill()
    # conns_trend_1h = conns - conns_lag_1h
    # conns_scaled = conns.max() / conns
    conns_scaled = conns * (conns.max() / conns.rolling(window=168).max().bfill())  # 168 hours = 7 days
    # conns_scaled = conns / (conns.max() / conns.rolling(window=168).max().bfill())  # 168 hours = 7 days

    energy_per_conn = df['energie_kwh'] / conns_scaled
    energy_lag = energy_per_conn.shift(1).bfill()
    energy_rolling = energy_lag.rolling(window=6).mean().bfill()
    # clip
    energy_lag = energy_lag.clip(upper=energy_rolling * 2)  

    # energy_per_conn_lag_1h = energy_lag / conns_lag_1h

    temp = df['temperature_ext']
    # irradiance = df['irradiance_solaire']

    temp_rolling_avg = temp.rolling(window=24).mean().bfill()

    # Heating degree-days (cold weather)
    temp_heating = np.maximum(18 - temp, 0)
    temp_heating_pow = temp_heating ** 2
    temp_heating_wind = temp_heating * df['vitesse_vent']
    
    # Cooling degree-days (hot weather) - important for summer test data
    # temp_cooling = np.maximum(temp - 22, 0)

    # Irradiance features
    # irradiance_rolling_24h = irradiance.rolling(window=24).mean().bfill()

    # Target encoding for poste
    # poste_enc = df['poste'].map(poste_energy_means)
    poste_one_hot = pd.get_dummies(df['poste'], prefix='poste')
    # poste_a = poste_one_hot['poste_A']
    # poste_b = poste_one_hot['poste_B']
    poste_c = poste_one_hot['poste_C']

    # Poste × heating interactions (cold weather)
    # poste_a_heating = np.where(df['poste'] == 'A', temp_heating, 0)
    # poste_b_heating = np.where(df['poste'] == 'B', temp_heating, 0)
    # poste_c_heating = np.where(df['poste'] == 'C', temp_heating, 0)
    
    # Poste × cooling interactions (hot weather - for summer test data)
    # poste_c_cooling = np.where(df['poste'] == 'C', temp_cooling, 0)

    # Peak hour features
    # is_peak_hour_cond = ((df['heure'] >= 6) & (df['heure'] <= 9)) | ((df['heure'] >= 16) & (df['heure'] <= 20))
    # is_peak_hour = np.where(is_peak_hour_cond, 1.0, 0.0)

    # conn_heating_load = conns * temp_heating_sq
    temp_rolling_avg_poste = temp_rolling_avg * poste_c
    temp_rolling_avg_cos = temp_rolling_avg * df['heure_cos']

    return {
        EngFeature.energy_lag: energy_lag,
        EngFeature.energy_rolling: energy_rolling,
        # EngFeature.conns_lag_1h: conns_lag_1h,
        EngFeature.conns_scaled: conns_scaled,
        # EngFeature.conns_trend_1h: conns_trend_1h,
        # EngFeature.temp_rolling_avg: temp_rolling_avg,
        # EngFeature.temp_heating: temp_heating,
        EngFeature.temp_heating_pow: temp_heating_pow,
        # EngFeature.temp_cooling: temp_cooling,
        # EngFeature.is_peak_hour: is_peak_hour,
        # EngFeature.irradiance_rolling_24h: irradiance_rolling_24h,
        # EngFeature.poste_enc: poste_enc,
        # EngFeature.poste_b: poste_b,
        EngFeature.poste_c: poste_c,
        # EngFeature.energy_per_conn_lag_1h: energy_per_conn_lag_1h,
        # EngFeature.poste_a_heating: poste_a_heating,
        # EngFeature.poste_b_heating: poste_b_heating,
        # EngFeature.poste_c_heating: poste_c_heating,
        # EngFeature.poste_c_cooling: poste_c_cooling,
        
        # EngFeature.conn_heating_load: conn_heating_load,
        # EngFeature.temp_rolling_avg_poste: temp_rolling_avg_poste
        # EngFeature.temp_rolling_avg_cos: temp_rolling_avg_cos,
        EngFeature.temp_heating_wind: temp_heating_wind,
    }
```