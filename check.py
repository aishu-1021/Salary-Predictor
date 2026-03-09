import pandas as pd
df = pd.read_csv('ds_salaries.csv')

# Check exact column names
print("Columns:", df.columns.tolist())

# Check what values are in company_location
print("\nUnique company_location values:")
print(df['company_location'].value_counts().head(20))

# Check what values are in experience_level
print("\nUnique experience_level values:")
print(df['experience_level'].value_counts())

# Check salary range
print("\nSalary range:")
print(f"Min: ${df['salary_in_usd'].min():,}")
print(f"Max: ${df['salary_in_usd'].max():,}")