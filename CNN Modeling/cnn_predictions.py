import pandas as pd
import numpy as np
from manual_neural_network import ManualCNN, binary_crossentropy
from cnn_utilities import(load_data, get_date_columns, MinMaxScaler, create_september_tracker,
                    create_seasonal_features, calculate_volatility, calculate_rate_of_change, integrated_dfs)

SEQ_LEN = 12
def create_sequences(data, seq_len=12):
    X, y = [], []

    price_series = data[:, 0]

    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i, :])
        y.append(
            1 if price_series[i] < np.mean(price_series[i-6:i]) else 0
        )

    return np.array(X), np.array(y).reshape(-1, 1)


def process_data(prices, date_cols, has_university, forecast_1m, forecast_3m, train_ratio=0.8):
    valid_mask = ~np.isnan(prices)
    prices = prices[valid_mask]
    date_cols = np.array(date_cols)[valid_mask]

    if len(prices) < SEQ_LEN + 6:
        return None
    features = build_features(prices, date_cols, has_university, forecast_1m, forecast_3m)

    split = int(len(features) * train_ratio)
    train_feat = features[:split]
    test_feat = features[split:]

    scaler = MinMaxScaler()
    train_feat = scaler.fit_transform(train_feat)
    test_feat = scaler.transform(test_feat)

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
        forecast_1m = row['ZORI_forecast_1m']
        forecast_3m = row['ZORI_forecast_3m']

        result = process_data(prices, date_cols, has_university, forecast_1m, forecast_3m)
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

def build_features(prices, date_cols, has_university, forecast_1m, forecast_3m):
    volatility = calculate_volatility(prices)
    roc = calculate_rate_of_change(prices)

    seasonal = create_seasonal_features(date_cols)[:len(prices)]
    sept = create_september_tracker(date_cols)[:len(prices)]

    uni = np.full(len(prices), has_university)
    forecast_1m_arr = np.full(len(prices), forecast_1m)
    forecast_3m_arr = np.full(len(prices), forecast_3m)

    return np.column_stack([
        prices,
        volatility,
        uni,
        roc,
        seasonal[:, 0],
        seasonal[:, 1],
        sept,
        forecast_1m_arr,
        forecast_3m_arr
    ])

def predict_zip(df, date_cols, model, zip_code):
    match = df[df['ZipCode'] == str(zip_code).strip()]
    forecast_1m = match['ZORI_forecast_1m'].values[0]
    forecast_3m = match['ZORI_forecast_3m'].values[0]

    if match.empty:
        print(f"Zip code {zip_code} not found.")
        return

    prices = match.iloc[0][date_cols].values.astype(float)
    prices = prices[~np.isnan(prices)]
    has_university = match['HasUniversity']

    if len(prices) < SEQ_LEN:
        print("Not enough data for prediction.")
        return

    features = build_features(prices, date_cols, has_university, forecast_1m, forecast_3m)

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
    df = integrated_dfs(df)
    date_cols = get_date_columns(df)
    print(f"Loaded {len(df)} regions\n")

    print("Preparing training data...")
    X_train, y_train, X_test, y_test = prepare_training_data(df, date_cols)

    pos_weight = np.sum(y_train == 0) / (np.sum(y_train == 1) + 1e-8)


    timesteps = X_train.shape[1]
    features = X_train.shape[2]

    model = ManualCNN(input_size=timesteps, num_features= features)

    epochs = 20
    batch_size = 64

    print("\nTraining CNN from scratch...\n")

    for epoch in range(epochs):

        # shuffle samples
        idx = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[idx]
        y_train_shuffled = y_train[idx]

        num_batches = len(X_train) // batch_size
        epoch_loss = 0

        for b in range(num_batches):
            start = b * batch_size
            end = start + batch_size

            X_batch = X_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]

            y_pred = model.forward(X_batch)
            loss = binary_crossentropy(y_batch, y_pred)
            epoch_loss += loss

            model.backward(y_batch)

        # validation
        val_pred = model.forward(X_test)
        val_loss = binary_crossentropy(y_test, val_pred)
        val_acc = np.mean((val_pred > 0.5) == y_test)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"loss={epoch_loss/num_batches:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

    print("\nRunning predictions...\n")

    predict_zip(df, date_cols, model, '02115')
    predict_zip(df, date_cols, model, '02114')


if __name__ == "__main__":
    main()