import pandas as pd

# Load the dataset
df = pd.read_csv('ds_salaries.csv')

# 1. See the first 5 rows
print("First 5 rows:")
print(df.head())

# 2. See all column names
print("\nColumns:")
print(df.columns.tolist())

# 3. See shape (rows x columns)
print("\nShape:", df.shape)

# 4. Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# 5. Basic statistics
print("\nBasic stats:")
print(df.describe())