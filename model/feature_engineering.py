# feature_engineering.py

import pandas as pd
from scipy.stats.mstats import winsorize

def prepare_features(df):
    # Step 1: Fill missing dates
    df = df.sort_values(["region","product","date"])
    all_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    filled = []
    for (region, product), group in df.groupby(["region","product"]):
        group = group.set_index("date").reindex(all_dates)
        group["region"] = region
        group["product"] = product
        group["sales"] = group["sales"].fillna(0)
        filled.append(group.reset_index().rename(columns={"index":"date"}))
    df = pd.concat(filled)

    # Step 2: Time features
    df["dayofweek"] = df["date"].dt.dayofweek
    df["weekofyear"] = df["date"].dt.isocalendar().week
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["dayofweek"].isin([5,6]).astype(int)

    # Step 3: Holiday flags (example)
    uae_holidays = pd.to_datetime(["2025-01-01","2025-04-01"])
    df["holiday_flag"] = df["date"].isin(uae_holidays).astype(int)

    # Step 4: Lag features
    df["lag_1"] = df.groupby(["region","product"])["sales"].shift(1)
    df["lag_7"] = df.groupby(["region","product"])["sales"].shift(7)
    df["lag_30"] = df.groupby(["region","product"])["sales"].shift(30)
    df["rmean_7"] = df.groupby(["region","product"])["sales"].shift(1).rolling(7).mean().reset_index(level=[0,1], drop=True)
    df["rmean_30"] = df.groupby(["region","product"])["sales"].shift(1).rolling(30).mean().reset_index(level=[0,1], drop=True)

    # Step 5: Weekly aggregation (optional)
    weekly = df.groupby(["region","product"]).resample("W", on="date")["sales"].sum().reset_index()

    # Step 6: Handle NaNs
    df = df.dropna()

    return df, weekly

