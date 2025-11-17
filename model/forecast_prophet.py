import pandas as pd
from prophet import Prophet

def prepare(df, region, product):
    sub = df[(df.region==region) & (df.product==product)][['date','sales']].rename(columns={'date':'ds','sales':'y'})
    return sub

def fit_forecast(df, periods=90, holidays=None):
    m = Prophet(weekly_seasonality=True, daily_seasonality=False, yearly_seasonality=True)
    if holidays is not None:
        m = Prophet(holidays=holidays)
    m.fit(df)
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    return forecast[['ds','yhat','yhat_lower','yhat_upper']]

if __name__=='__main__':
    df = pd.read_csv('../data/cleaned_sales.csv', parse_dates=['date'])
    out_frames = []
    for (reg,prod), group in df.groupby(['region','product']):
        sub = prepare(group, reg, prod)
        fc = fit_forecast(sub, periods=90)
        fc['region'] = reg
        fc['product'] = prod
        out_frames.append(fc)
    out = pd.concat(out_frames)
    out.rename(columns={'ds':'date','yhat':'forecast'}, inplace=True)
    out.to_csv('../data/forecast_output.csv', index=False)
    print('Forecasts written to ../data/forecast_output.csv')

