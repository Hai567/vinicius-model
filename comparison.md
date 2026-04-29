# Notebooks vs forecast.py — Comparison

## Architecture Overview

| Aspect | Your Notebooks (1→2→3) | forecast.py |
|--------|------------------------|-------------|
| Structure | 3 notebooks + CSV intermediates | Single self-contained script |
| Model | LightGBM only | LightGBM + XGBoost ensemble |
| Features | 25 | 82 |
| Prediction | One-shot batch predict | Recursive day-by-day |
| Val R² (Rev) | **0.7735** | **0.8287** (XGB) |
| Val R² (COGS) | **0.7692** | **0.8140** (XGB) |
| Test R² (Rev) | unknown | **0.9369** |

---

## Detailed Comparison

### 1. Preprocessing (Notebook 1)

**What you do well:**
- Proper Train/Test split labeling (`Split` column)
- Lag-1 and lag-7 for transaction features (orders, returns, discounts)
- Correctly dropping raw columns to avoid leakage
- Inventory features with 1-month lag
- Web traffic + promo count merging

**Issues found:**

> [!WARNING]
> **Data leakage via `ffill`** — In your final merge (line 330), you do `ffill()` on ALL feature columns. This forward-fills the *last known* training values into the test period. For example, `total_orders_lag1` in test gets the Dec 31, 2022 value carried forward for all 548 test days. This makes the model think test-period order counts are constant.

> [!WARNING]
> **No Revenue/COGS lag features** — Your lag features only cover auxiliary data (orders, returns, discounts). You're missing the most important features: lagged values of the target itself (Revenue_lag1, Revenue_lag7, Revenue_lag365, rolling means, etc.).

### 2. Feature Engineering Comparison

| Feature Category | Your Notebooks | forecast.py |
|-----------------|---------------|-------------|
| Calendar basics | year, month, day, dow, is_weekend, quarter | Same + doy, woy, day_frac, cyclical sin/cos |
| **Seasonal baseline** | ❌ Missing | ✅ (month,day) profile × YoY trend |
| **Seasonal by DOW** | ❌ Missing | ✅ (month,dow) profile |
| **Vietnamese holidays** | ❌ Missing | ✅ Tet, New Year, Reunification, Labour, National Day |
| **Tet features** | ❌ Missing | ✅ days_to_tet, is_tet_season, pre/post_tet |
| **EOM sale cycle** | ❌ Missing | ✅ end-of-month spike detection |
| **Revenue/COGS lags** | ❌ Missing | ✅ lag 1,2,3,7,14,28,30,365,366,730 |
| **Rolling statistics** | ❌ Missing | ✅ 7/14/30/90-day rolling mean & std |
| **COGS/Rev ratio** | ❌ Missing | ✅ ratio lags + rolling means |
| **Trend** | ❌ Missing | ✅ days_idx + YoY growth factors |
| Transaction lags | ✅ lag1, lag7 for orders/returns/items | ❌ Not used (data doesn't exist in test) |
| Web traffic | ✅ raw values (NaN in test) | ❌ Not used (data doesn't exist in test) |
| Inventory | ✅ monthly lag | ❌ Not used |
| Promotions | ✅ active count | ✅ active count |

### 3. Model Training (Notebook 2)

**Your approach:**
```python
model_rev = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42
)
```

**forecast.py approach:**
```python
# LightGBM
lgb_p = dict(objective='regression', metric='mae', learning_rate=0.02,
             num_leaves=127, min_child_samples=15, subsample=0.8,
             colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=0.5,
             n_estimators=5000, random_state=42)
# + XGBoost with similar params
# + Optimized ensemble weights via grid search
```

| Parameter | Your Notebooks | forecast.py |
|-----------|---------------|-------------|
| n_estimators | 500 | 5000 (with early stopping) |
| learning_rate | 0.05 | 0.02 (lower = more trees = better) |
| max_depth | 6 | unlimited (num_leaves=127) |
| Regularization | None | alpha=0.1, lambda=0.5, subsample=0.8 |
| num_leaves | default (31) | 127 |
| Second model | ❌ | ✅ XGBoost |
| Ensemble | ❌ | ✅ Optimized weights |

> [!IMPORTANT]
> The massive "No further splits with positive gain" warnings in your output indicate the model is **underfitting** — it exhausts the 500 trees trying to find splits and can't improve. This is because: (a) too few features (25), (b) features become constant in parts of data, (c) max_depth=6 is restrictive.

### 4. Test-Time Prediction (Notebook 3)

> [!CAUTION]
> **Critical flaw: one-shot prediction without recursive lags.** Your notebook simply calls `model.predict(X_test)` on all 548 test rows at once. Since lag features are forward-filled constants from Dec 2022, the model sees the same stale lag values for every test day. This makes predictions flat and unable to capture the monthly cycles.

**forecast.py uses recursive prediction:**
```
For each test day i:
  1. Recompute lag features using previously predicted values
  2. Predict Revenue[i] and COGS[i]
  3. Store predictions for use as future lags
```
This captures the monthly spike pattern (end-of-month → drop → gradual buildup).

### 5. Explainability (Notebook 2)

Your SHAP analysis is a **strong point** not present in forecast.py:
- ✅ SHAP TreeExplainer 
- ✅ Feature importance bar plot
- ✅ Beeswarm summary plot
- These directly satisfy the contest's "Explainability" requirement

forecast.py only has basic feature importance (no SHAP).

---

## Key Recommendations

### Must-Fix (High Impact)
1. **Add Revenue/COGS lag features** — These are the #1 most important features (see forecast.py importances)
2. **Implement recursive prediction** — Without this, test predictions will be flat
3. **Add seasonal baseline** — The `(month, day) × trend` profile alone gives reasonable predictions
4. **Add Vietnamese calendar features** — Tet and holidays cause massive revenue swings

### Should-Fix (Medium Impact)
5. **Increase model capacity** — More trees (5000), lower learning rate (0.02), more leaves (127)
6. **Add XGBoost ensemble** — Reduces variance
7. **Add regularization** — subsample, colsample_bytree, reg_alpha/lambda
8. **Fix ffill leakage** — Use proper NaN handling instead of blind forward-fill

### Nice-to-Have
9. **Add cyclical encoding** for month/dow — Helps with periodicity
10. **Keep your SHAP analysis** — It's required by the contest and well-implemented

---

## Suggested Merge Strategy

The ideal solution combines the best of both:

```
Your Notebooks (keep):          forecast.py (adopt):
├── SHAP explainability         ├── 82 features (seasonal, Tet, lags, trend)
├── Inventory lag features      ├── Recursive prediction loop
├── Clean notebook structure    ├── LGB + XGB ensemble
└── Vietnamese comments         ├── Optimized hyperparameters
                                └── Ensemble weight optimization
```
