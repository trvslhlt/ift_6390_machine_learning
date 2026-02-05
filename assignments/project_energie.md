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