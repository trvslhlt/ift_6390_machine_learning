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