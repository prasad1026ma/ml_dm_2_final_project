import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings('ignore')

def load_and_prepare_zori(filepath):
    df = pd.read_csv(filepath)
    date_cols = [col for col in df.columns if isinstance(col, str) and col.count('-') == 2]
    metadata_cols = [col for col in df.columns if col not in date_cols]
    zori_long = df[metadata_cols + date_cols].melt(
        id_vars=metadata_cols,
        value_vars=date_cols,
        var_name='Date',
        value_name='ZORI'
    )

    zori_long['Date'] = pd.to_datetime(zori_long['Date'])
    zori_long['ZORI'] = pd.to_numeric(zori_long['ZORI'], errors='coerce')
    zori_long = zori_long.sort_values(['ZipCode', 'Date']).reset_index(drop=True)

    return zori_long

def check_stationarity(timeseries, name=''):
    result = adfuller(timeseries.dropna(), autolag='AIC')

    is_stationary = result[1] < 0.05

    return {
        'name': name,
        'adf_statistic': result[0],
        'p_value': result[1],
        'is_stationary': is_stationary,
        'critical_values': result[4]
    }
def find_optimal_arima_order(timeseries, p_range=(0, 3), d_range=(0, 2), q_range=(0, 3)):
    best_aic = np.inf
    best_order = (1, 1, 1)

    for p in range(p_range[0], p_range[1] + 1):
        for d in range(d_range[0], d_range[1] + 1):
            for q in range(q_range[0], q_range[1] + 1):
                try:
                    model = ARIMA(timeseries, order=(p, d, q))
                    fitted = model.fit()

                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                except:
                    continue

    return best_order
def forecast_zip_code_zori(zip_code_data, zip_code, months_ahead=[1, 3]):
    ts = zip_code_data.sort_values('Date').set_index('Date')['ZORI']
    ts = ts[ts.notna()]

    if len(ts) < 12:
        return {
            'ZipCode': zip_code,
            'ZORI_forecast_1m': np.nan,
            'ZORI_forecast_3m': np.nan,
        }

    try:
        best_order = find_optimal_arima_order(ts)
        model = ARIMA(ts, order=best_order)
        fitted_model = model.fit()

        forecast_result = fitted_model.get_forecast(steps=max(months_ahead))
        forecasts = forecast_result.predicted_mean

        results = {
            'ZipCode': zip_code,
            'Data_Points': len(ts),
        }

        for month in months_ahead:
            if month <= len(forecasts):
                forecast_val = forecasts.iloc[month - 1]
                results[f'ZORI_forecast_{month}m'] = round(forecast_val, 2)
            else:
                results[f'ZORI_forecast_{month}m'] = np.nan

        return results

    except Exception as e:
        return {
            'ZipCode': zip_code,
            'ZORI_forecast_1m': np.nan,
            'ZORI_forecast_3m': np.nan,
        }

def generate_zori_forecasts(input_filepath, output_filepath=None):
    zori_data = load_and_prepare_zori(input_filepath)
    zip_codes = zori_data['ZipCode'].unique()

    all_forecasts = []

    for i, zip_code in enumerate(zip_codes):
        zip_data = zori_data[zori_data['ZipCode'] == zip_code]
        forecast = forecast_zip_code_zori(zip_data, zip_code, months_ahead=[1, 3])
        all_forecasts.append(forecast)

    forecast_df = pd.DataFrame(all_forecasts)
    metadata = zori_data.groupby('ZipCode').first()[['RegionID', 'City', 'Metro', 'State']].reset_index()
    forecast_df = forecast_df.merge(metadata, on='ZipCode', how='left')
    key_cols = ['ZipCode', 'ZORI_forecast_1m', 'ZORI_forecast_3m']
    other_cols = [col for col in forecast_df.columns if col not in key_cols]
    forecast_df = forecast_df[key_cols + other_cols]

    if output_filepath is None:
        output_filepath = 'zori_forecasts.csv'

    forecast_df.to_csv(output_filepath, index=False)
    return forecast_df
