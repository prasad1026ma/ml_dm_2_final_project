import pandas as pd
import numpy as np
from manual_neural_network import ManualNN
from cnn_utilities import(load_data, get_date_columns, MinMaxScaler, create_september_tracker,
                    create_seasonal_features, calculate_volatility, calculate_rate_of_change)
''
SEQ_LEN = 12

def create_sequences(data, seq_len=12):
    X, y = [], []
    for i in range(seq_len, len(data)):
        seq = data[i - seq_len:i]
        price_series = data[:, 0]
        rolling_avg = np.mean(price_series[i - 6:i])
        label = 1 if price_series[i] < rolling_avg else 0

        X.append(seq)
        y.append(label)

    return np.array(X), np.array(y).reshape(-1, 1)


def process_data(prices, date_cols, has_university, train_ratio=0.8):
    valid_mask = ~np.isnan(prices)
    prices = prices[valid_mask]
    if len(prices) < SEQ_LEN + 6:
        return None
    features = build_features(prices, date_cols, has_university)

    split = int(len(features) * train_ratio)
    train_feat = features[:split]
    test_feat = features[split:]

    scaler = MinMaxScaler()
    train_feat[:, 0] = scaler.fit_transform(train_feat[:, 0])
    test_feat[:, 0] = scaler.transform(test_feat[:, 0])

    X_train, y_train = create_sequences(train_feat, SEQ_LEN)

    combined = np.vstack([train_feat[-SEQ_LEN:], test_feat])
    X_test, y_test = create_sequences(combined, SEQ_LEN)

    return X_train, y_train, X_test, y_test


def model_eval(model, X_train, y_train, X_test, y_test, epochs, batch_size):
    model.train(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    loss, acc = model.evaluate(X_test, y_test)
    return loss, acc

def prepare_training_data(df, date_cols):
    X_train_all, y_train_all = [], []
    X_test_all, y_test_all = [], []

    for idx, row in df.iterrows():
        prices = row[date_cols].values.astype(float)
        has_university = row['HasUniversity']
        result = process_data(prices, date_cols, has_university)
        if result is None:
            continue

        X_train, y_train, X_test, y_test = result

        X_train_all.append(X_train)
        y_train_all.append(y_train)
        X_test_all.append(X_test)
        y_test_all.append(y_test)

        print(f"Processed region {idx + 1}/{len(df)}")

        if len(X_train_all) == 0:
            print("No valid training data found.")
            return None

    X_train = np.concatenate(X_train_all, axis=0)
    y_train = np.concatenate(y_train_all, axis=0)
    X_test = np.concatenate(X_test_all, axis=0)
    y_test = np.concatenate(y_test_all, axis=0)

    return X_train, y_train, X_test, y_test

def build_features(prices, date_cols, has_university):
    volatility = calculate_volatility(prices)
    roc = calculate_rate_of_change(prices)

    seasonal = create_seasonal_features(date_cols)[:len(prices)]
    sept = create_september_tracker(date_cols)[:len(prices)]

    uni = np.full(len(prices), has_university)

    return np.column_stack([
        prices,
        volatility,
        roc,
        seasonal[:, 0],
        seasonal[:, 1],
        sept,
        uni
    ])

def predict_zip(df, date_cols, model, zip_code):
    match = df[df['ZipCode'] == str(zip_code).strip()]
    if match.empty:
        print(f"Zip code {zip_code} not found.")
        return

    prices = match.iloc[0][date_cols].values.astype(float)
    prices = prices[~np.isnan(prices)]
    has_university = match['HasUniversity']

    if len(prices) < SEQ_LEN:
        print("Not enough data for prediction.")
        return

    features = build_features(prices, date_cols, has_university)

    scaler = MinMaxScaler()
    features[:, 0] = scaler.fit_transform(features[:, 0])

    seq = features[-SEQ_LEN:]
    seq = seq.reshape(1, -1)
    prob = model.predict(seq)[0][0]

    print(f"Zip Code: {zip_code}")
    print(f"Good to rent confidence: {prob:.2%}")


def main():
    print("Loading data...")
    df = load_data('../boston_cleaned_data.csv')
    date_cols = get_date_columns(df)
    print(f"Loaded {len(df)} regions\n")

    print("Preparing training data...")
    X_train, y_train, X_test, y_test = prepare_training_data(df, date_cols)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    input_size = X_train.shape[1]

    learning_rate = 0.01
    model = ManualNN(input_size=input_size, learning_rate=learning_rate)
    model_eval(model, X_train, y_train, X_test, y_test, epochs=20, batch_size=32)

    predict_zip(df, date_cols, model, '02115')
    predict_zip(df, date_cols, model, '02114')


if __name__ == "__main__":
    main()