# eda_checklist.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("data/raw_sales.csv", parse_dates=["date"])

# 1. Inspect missing values and duplicates
print("Missing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

# 2. Aggregate to required granularity (daily/weekly)
# If many zeros, consider weekly aggregation
weekly_sales = df.groupby(["region", "product"]).resample("W", on="date")["sales"].sum().reset_index()
print(weekly_sales.head())

# 3. Plot time series per SKU-region
sample = weekly_sales[(weekly_sales["region"]=="UAE") & (weekly_sales["product"]=="SKU-102")]
sample.set_index("date")["sales"].plot(title="Weekly Sales - Prod A (UAE)", figsize=(10,5))
plt.show()

# 4. Check holidays/promotions (if columns exist)
if "holiday_flag" in df.columns:
    holiday_sales = df[df["holiday_flag"]==1].groupby("date")["sales"].sum()
    holiday_sales.plot(kind="bar", title="Sales on Holidays")
    plt.show()

if "promotion_flag" in df.columns:
    promo_sales = df.groupby("promotion_flag")["sales"].mean()
    promo_sales.plot(kind="bar", title="Avg Sales by Promotion Flag")
    plt.show()

# 5. Check outliers
sns.boxplot(x="sales", data=df)
plt.title("Sales Outlier Detection")
plt.show()

