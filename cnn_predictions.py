import pandas as pd
import numpy as np
from manual_neural_network import ManualNN

SEQ_LEN = 12

class MinMaxScaler:
    def fit(self, data):
        self.min = np.min(data)
        self.max = np.max(data)
        return self
    def transform(self, data):
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
                 'StateName', 'State', 'City', 'Metro', 'CountyName']
    date_cols = [c for c in df.columns if c not in meta_cols]
    return date_cols

def scale_prices(prices):
    scaler = MinMaxScaler()
    return scaler.fit_transform(prices), scaler


def create_sequences(prices, seq_len=12):
    X, y = [], []

    for i in range(seq_len, len(prices)):
        seq = prices[i - seq_len:i]
        rolling_avg = np.mean(prices[i - 6:i])
        label = 1 if prices[i] < rolling_avg else 0
        X.append(seq)
        y.append(label)

    return np.array(X), np.array(y).reshape(-1, 1)


def process_data(prices, date_cols, train_ratio=0.8):
    prices = prices[~np.isnan(prices)]

    if len(prices) < SEQ_LEN + 6:
        return None

    split = int(len(prices) * train_ratio)
    train_prices = prices[:split]
    test_prices = prices[split:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_prices)
    test_scaled = scaler.transform(test_prices)

    X_train, y_train = create_sequences(train_scaled, SEQ_LEN)

    combined = np.concatenate([train_scaled[-SEQ_LEN:], test_scaled])
    X_test, y_test = create_sequences(combined, SEQ_LEN)

    return X_train, y_train, X_test, y_test


def model_eval(model, X_train, y_train, X_test, y_test, epochs, batch_size):
    """Create a new ManualNN model"""
    model.train(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    loss, acc = model.evaluate(X_test, y_test)
    return loss, acc



def prepare_training_data(df, date_cols):
    X_train_all, y_train_all = [], []
    X_test_all, y_test_all = [], []

    for idx, row in df.iterrows():
        prices = row[date_cols].values.astype(float)
        result = process_data(prices, date_cols)
        if result is None:
            continue

        X_train, y_train, X_test, y_test = result

        X_train_all.append(X_train)
        y_train_all.append(y_train)
        X_test_all.append(X_test)
        y_test_all.append(y_test)

        print(f"Processed region {idx + 1}/{len(df)}")

    X_train = np.concatenate(X_train_all, axis=0)
    y_train = np.concatenate(y_train_all, axis=0)
    X_test = np.concatenate(X_test_all, axis=0)
    y_test = np.concatenate(y_test_all, axis=0)

    return X_train, y_train, X_test, y_test


def predict_zip(df, date_cols, model, zip_code):
    match = df[df['ZipCode'] == str(zip_code).strip()]
    if match.empty:
        print(f"Zip code {zip_code} not found.")
        return

    prices = match.iloc[0][date_cols].values.astype(float)
    prices = prices[~np.isnan(prices)]

    if len(prices) < SEQ_LEN:
        print("Not enough data for prediction.")
        return

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)
    seq = scaled[-SEQ_LEN:].reshape(1, SEQ_LEN)

    prob = model.predict(seq)[0][0]

    print(f"Zip Code: {zip_code}")
    print(f"Good to rent confidence: {prob:.2%}")
    if prob >= 0.6:
        print("Good time to rent")
    elif prob <= 0.4:
        print("Bad time to rent")
    else:
        print("No clear signal")


def main():
    print("Loading data...")
    df = load_data('boston_cleaned_data.csv')
    date_cols = get_date_columns(df)
    print(f"Loaded {len(df)} regions\n")

    print("Preparing training data...")
    X_train, y_train, X_test, y_test = prepare_training_data(df, date_cols)

    learning_rate = 0.01
    model = ManualNN(input_size=SEQ_LEN, learning_rate=learning_rate)
    model_eval(model, X_train, y_train, X_test, y_test, epochs=20, batch_size=32)

    predict_zip(df, date_cols, model, '02115')
    predict_zip(df, date_cols, model, '02114')


if __name__ == "__main__":
    main()
