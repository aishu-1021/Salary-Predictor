import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# ── 1. Load dataset ──────────────────────────────────────────
df = pd.read_csv('ds_salaries.csv')

# ── 2. Keep US only & remove outliers ───────────────────────
df = df[df['company_location'] == 'US'].copy()
low  = df['salary_in_usd'].quantile(0.05)
high = df['salary_in_usd'].quantile(0.95)
df = df[(df['salary_in_usd'] >= low) & (df['salary_in_usd'] <= high)]

print(f"Rows after cleaning : {len(df)}")
print(f"Salary range        : ${df['salary_in_usd'].min():,} - ${df['salary_in_usd'].max():,}")

# ── 3. Drop unnecessary columns ──────────────────────────────
drop_cols = [c for c in ['salary', 'salary_currency',
             'employee_residence', 'company_location'] if c in df.columns]
df = df.drop(columns=drop_cols)

# ── 4. Encode categorical columns ────────────────────────────
encoders = {}
for col in ['experience_level', 'employment_type', 'job_title', 'company_size']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# ── 5. Features & target ─────────────────────────────────────
features = ['work_year', 'experience_level', 'employment_type',
            'job_title', 'remote_ratio', 'company_size']
target = 'salary_in_usd'

X = df[features]
y = df[target]

# ── 6. Train/test split ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ── 7. Train THREE models ─────────────────────────────────────
# Low estimate  = 25th percentile
# Mid estimate  = 50th percentile (median)
# High estimate = 75th percentile

params = dict(n_estimators=500, learning_rate=0.05,
              max_depth=6, subsample=0.8,
              colsample_bytree=0.8, verbosity=0)

print("\nTraining low estimate model  (25th percentile)...")
model_low  = XGBRegressor(objective='reg:quantileerror',
                           quantile_alpha=0.25, **params)
model_low.fit(X_train, y_train)

print("Training mid estimate model  (50th percentile)...")
model_mid  = XGBRegressor(objective='reg:quantileerror',
                           quantile_alpha=0.50, **params)
model_mid.fit(X_train, y_train)

print("Training high estimate model (75th percentile)...")
model_high = XGBRegressor(objective='reg:quantileerror',
                           quantile_alpha=0.75, **params)
model_high.fit(X_train, y_train)

print("\n✅ All 3 models trained!")

# ── 8. Evaluate ──────────────────────────────────────────────
pred_low  = model_low.predict(X_test)
pred_mid  = model_mid.predict(X_test)
pred_high = model_high.predict(X_test)

mae = mean_absolute_error(y_test, pred_mid)

# Coverage = how often the true salary falls within predicted range
coverage = np.mean((y_test >= pred_low) & (y_test <= pred_high))
avg_range = np.mean(pred_high - pred_low)

print(f"\n Model Performance:")
print(f"   Median MAE          : ${mae:,.0f}")
print(f"   Range Coverage      : {coverage*100:.1f}% of actual salaries fall within predicted range")
print(f"   Average Range Width : ${avg_range:,.0f}")

# ── 9. Sample predictions ────────────────────────────────────
print(f"\n Sample Predictions vs Actual:")
for i in range(5):
    actual = y_test.iloc[i]
    lo, mid, hi = pred_low[i], pred_mid[i], pred_high[i]
    inside = "✅" if lo <= actual <= hi else "❌"
    print(f"   Actual: ${actual:>9,.0f}  |  "
          f"Range: ${lo:>9,.0f} - ${hi:>9,.0f}  {inside}")

# ── 10. Save everything ──────────────────────────────────────
os.makedirs('model', exist_ok=True)
joblib.dump(model_low,  'model/model_low.pkl')
joblib.dump(model_mid,  'model/model_mid.pkl')
joblib.dump(model_high, 'model/model_high.pkl')
joblib.dump(encoders,   'model/encoders.pkl')
joblib.dump(features,   'model/features.pkl')

print("\n All models saved to /model folder!")