# forecast_xgb.py
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
import joblib

def create_lag_features(df, lags=[1,7,30]):
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby(['region','product'])['sales'].shift(lag)
    df['rmean_7'] = df.groupby(['region','product'])['sales'].shift(1).rolling(7).mean().reset_index(level=[0,1], drop=True)
    return df

if __name__=='__main__':
    df = pd.read_csv('../data/cleaned_sales.csv', parse_dates=['date'])
    df = create_lag_features(df)
    df = df.dropna(subset=['lag_1','lag_7','rmean_7'])
    models = {}
    preds = []
    for (reg,prod), group in df.groupby(['region','product']):
        g = group.copy().sort_values('date')
        X = g[['lag_1','lag_7','rmean_7']].fillna(0)
        y = g['sales']
        # last row(s) to forecast using last features
        model = XGBRegressor(n_estimators=200, learning_rate=0.05)
        model.fit(X, y)
        models[(reg,prod)] = model
        # generate n-step forecast by iteratively predicting and updating lags
        last = g.iloc[-1:].copy()
        steps = 90
        for i in range(steps):
            X_last = last[['lag_1','lag_7','rmean_7']].values
            pred = model.predict(X_last)[0]
            next_date = last['date'].iloc[0] + pd.Timedelta(days=1)
            preds.append({'date': next_date, 'region':reg, 'product':prod, 'forecast':pred})
            # update last for next iter (shift lags)
            new = last.copy()
            new['date'] = next_date
            new['lag_1'] = pred
            new['lag_7'] = new['lag_7']  # you can implement rolling shift logic
            last = new
    pd.DataFrame(preds).to_csv('../data/forecast_output_xgb.csv', index=False)
    joblib.dump(models, '../model/xgb_models.jobli/b')
