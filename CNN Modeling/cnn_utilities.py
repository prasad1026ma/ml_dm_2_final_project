import numpy as np
import pandas as pd
from datetime import datetime


class MinMaxScaler:
    """
    Performs sklearns min-max scalar manually
    """
    def fit(self, data):
        """
        Finds the max and min of the data
        :param data: array of inputted values
        :return:
        """
        self.min = np.min(data)
        self.max = np.max(data)
        return self
    def transform(self, data):
        """

        :param data:
        :return:
        """
        if self.max == self.min:
            return np.zeros_like(data, dtype=float)
        return (data - self.min) / (self.max - self.min)

    def fit_transform(self, data):
        return self.fit(data).transform(data)


def load_data(filepath):
    df = pd.read_csv(filepath, dtype={'ZipCode': str})
    return df

def get_date_columns(df):
    meta_cols = ['RegionID', 'SizeRank', 'ZipCode', 'RegionType',
                 'StateName', 'State', 'City', 'Metro', 'CountyName',
                 'Neighborhood', 'HasUniversity']
    date_cols = [c for c in df.columns if c not in meta_cols]
    return date_cols

def scale_prices(prices):
    scaler = MinMaxScaler()
    return scaler.fit_transform(prices), scaler

def calculate_volatility(prices, window=6):
    volatility = np.zeros_like(prices, dtype=float)
    for i in range(window, len(prices)):
        volatility[i] = np.std(prices[i - window:i])

    return volatility
def calculate_rate_of_change(prices):
    roc = np.zeros_like(prices, dtype=float)
    for i in range(1, len(prices)):
        if prices[i - 1] != 0:
            roc[i] = (prices[i] - prices[i - 1]) / prices[i - 1]
        else:
            roc[i] = 0
    return roc

def extract_month_from_date_string(date_str):
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m')
        except ValueError:
            return None
    return date_obj.month


def create_seasonal_features(date_cols):
    seasonal_features = []
    for date_col in date_cols:
        month = extract_month_from_date_string(date_col)
        if month is None:
            month_sin = 0.0
            month_cos = 0.0
            season_flag = 0.0
        else:
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)

            if month in [6, 7, 8]:
                season_flag = 1.0
            elif month in [12, 1, 2]:
                season_flag = -1.0
            else:
                season_flag = 0.0

        seasonal_features.append([month_sin, month_cos, season_flag])

    return np.array(seasonal_features)


def create_september_tracker(date_cols, sigma=1.5):
    intensities = []
    for date_col in date_cols:
        month = extract_month_from_date_string(date_col)
        if month is None:
            intensities.append(0.0)
            continue
        val = np.exp(-0.5 * ((month - 9) / sigma) ** 2)
        intensities.append(val)

    return np.array(intensities)

def integrated_dfs(df):
    prediction_df = pd.read_csv('zori_forecasts.csv')
    prediction_df = prediction_df[['ZipCode', 'ZORI_forecast_1m', 'ZORI_forecast_3m']]
    prediction_df['ZipCode'] = prediction_df['ZipCode'].astype(str).str.zfill(5)
    df = df.merge(prediction_df, on="ZipCode", how="left")
    return df
