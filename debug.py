import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

df = pd.read_csv('ds_salaries.csv')
df = df[df['company_location'] == 'US'].copy()

low  = df['salary_in_usd'].quantile(0.05)
high = df['salary_in_usd'].quantile(0.95)
df = df[(df['salary_in_usd'] >= low) & (df['salary_in_usd'] <= high)]

# ── Encode ───────────────────────────────────────────────────
for col in ['experience_level', 'employment_type', 'job_title', 'company_size']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

features = ['work_year', 'experience_level', 'employment_type',
            'job_title', 'remote_ratio', 'company_size']
X = df[features]
y = df['salary_in_usd']

# ── Test 1: Train ON FULL DATA (should get very high R²) ─────
model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
model.fit(X, y)
y_pred = model.predict(X)
print(f"R² on FULL data (no split): {r2_score(y, y_pred):.2f}")
# If this is also low → data itself is the problem

# ── Test 2: Normal split ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model2 = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
model2.fit(X_train, y_train)
print(f"R² on test split          : {r2_score(y_test, model2.predict(X_test)):.2f}")

# ── Check salary distribution ────────────────────────────────
print(f"\nSalary std deviation: ${y.std():,.0f}")
print(f"Salary value counts by experience:")
df_check = df.copy()
exp_map_rev = {0:'EN', 1:'MI', 2:'SE', 3:'EX'}
print(pd.DataFrame({'salary': y, 'exp': df['experience_level']})
      .groupby('exp')['salary'].agg(['mean','std','count']))