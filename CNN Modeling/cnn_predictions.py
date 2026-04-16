import numpy as np
import pandas as pd
import pickle

from manual_neural_network import ManualCNN, binary_crossentropy, weighted_binary_crossentropy
from sklearn.preprocessing import StandardScaler
from cnn_utilities import (load_data, get_date_columns, create_september_tracker,
                           create_seasonal_features, calculate_volatility, calculate_rate_of_change, integrated_dfs,
                           plot_training_history, save_model, load_model)

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)

import argparse


SEQ_LEN = 12

def create_sequences(data, seq_len=12):
    """Create sequences for time series prediction"""
    X, y = [], []
    price_series = data[:, 0]

    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i, :])
        y.append(
            1 if price_series[i] < np.mean(price_series[i - 6:i]) else 0
        )

    return np.array(X), np.array(y).reshape(-1, 1)


def flatten_sequences(X):
    """
    Converts (batch, timesteps, features) to (batch, timesteps*features)
    """
    batch_size = X.shape[0]
    return X.reshape(batch_size, -1)


def process_data(prices, date_cols, has_university, forecast_1m, forecast_3m,
                 scaler=None, fit_scaler=False, train_ratio=0.8):
    """Process data with proper scaling"""
    valid_mask = ~np.isnan(prices)
    prices = prices[valid_mask]
    date_cols = np.array(date_cols)[valid_mask]

    if len(prices) < SEQ_LEN + 6:
        return None

    features = build_features(prices, date_cols, has_university, forecast_1m, forecast_3m)

    split = int(len(features) * train_ratio)
    train_feat = features[:split]
    test_feat = features[split:]

    if scaler is None:
        scaler = StandardScaler()

    if fit_scaler:
        train_feat = scaler.fit_transform(train_feat)
    else:
        train_feat = scaler.transform(train_feat)

    test_feat = scaler.transform(test_feat)

    X_train, y_train = create_sequences(train_feat, SEQ_LEN)

    combined = np.vstack([train_feat[-SEQ_LEN:], test_feat])
    X_test, y_test = create_sequences(combined, SEQ_LEN)

    X_train = flatten_sequences(X_train)
    X_test = flatten_sequences(X_test)

    return X_train, y_train, X_test, y_test, scaler


def prepare_training_data(df, date_cols):
    """Prepare all training data"""
    X_train_all, y_train_all = [], []
    X_test_all, y_test_all = [], []
    scalers = {}

    scaler = StandardScaler()
    for idx, row in df.iterrows():
        prices = row[date_cols].values.astype(float)
        has_university = row['HasUniversity']
        forecast_1m = row['ZORI_forecast_1m']
        forecast_3m = row['ZORI_forecast_3m']

        result = process_data(prices, date_cols, has_university, forecast_1m, forecast_3m,
                              scaler=scaler, fit_scaler=True)

        if result is None:
            continue

        X_train, y_train, X_test, y_test, scaler = result

        X_train_all.append(X_train)
        y_train_all.append(y_train)
        X_test_all.append(X_test)
        y_test_all.append(y_test)
        scalers[idx] = scaler

        print(f"Processed region {idx + 1}/{len(df)}")

    if len(X_train_all) == 0:
        print("No valid training data found.")
        return None

    X_train = np.concatenate(X_train_all, axis=0)
    y_train = np.concatenate(y_train_all, axis=0)
    X_test = np.concatenate(X_test_all, axis=0)
    y_test = np.concatenate(y_test_all, axis=0)

    return X_train, y_train, X_test, y_test, scalers


def build_features(prices, date_cols, has_university, forecast_1m, forecast_3m):
    """Build feature matrix"""
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

def model_training():
    print("Loading data...")
    df = load_data('../boston_cleaned_data.csv')
    df = integrated_dfs(df)
    date_cols = get_date_columns(df)

    result = prepare_training_data(df, date_cols)

    if result is None:
        print("Failed to prepare data")
        return

    X_train, y_train, X_test, y_test, scalers = result

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scalers[0], f)

    # Calculate class weights
    class_0 = np.sum(y_train == 0)
    class_1 = np.sum(y_train == 1)
    pos_weight = class_0 / (class_1 + 1e-8)

    # Create model
    input_size = X_train.shape[1]
    print(f"Creating model with input_size={input_size}\n")

    model = ManualCNN(input_size=input_size, num_filters=16, l2_lambda=0.0005)

    epochs = 20
    batch_size = 64

    print("Training CNN:")
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        # Shuffle the data
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
            loss = weighted_binary_crossentropy(y_batch, y_pred)
            epoch_loss += loss

            model.backward(y_batch)

        # Validation
        val_pred = model.forward(X_test)
        val_loss = weighted_binary_crossentropy(y_test, val_pred)
        val_acc = np.mean((val_pred > 0.5) == y_test)

        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(
            f"Epoch {epoch + 1:2d}/{epochs} | "
            f"loss={epoch_loss / num_batches:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )
    plot_training_history(train_losses, val_losses)
    y_pred_prob = model.forward(X_test).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_pred_prob))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    save_model(model)

def predict_zip(df, model, zip_code, scaler):
    """Predict for a specific zip code"""
    match = df[df['ZipCode'] == str(zip_code).strip()]

    if match.empty:
        print(f"Zip code {zip_code} not found.")
        return

    date_cols = get_date_columns(df)
    prices = match.iloc[0][date_cols].values.astype(float)
    prices = prices[~np.isnan(prices)]
    has_university = match['HasUniversity'].values[0]
    forecast_1m = match['ZORI_forecast_1m'].values[0]
    forecast_3m = match['ZORI_forecast_3m'].values[0]

    if len(prices) < SEQ_LEN:
        print("Not enough data for prediction.")
        return

    # Build features
    features = build_features(prices, date_cols, has_university, forecast_1m, forecast_3m)
    features = scaler.transform(features)

    # Extract last SEQ_LEN timesteps and FLATTEN
    seq = features[-SEQ_LEN:].reshape(1, -1)

    pred_val = model.predict(seq)[0]

    print(f"Zip Code: {zip_code}")
    if pred_val < 0.5:
        print("Good Time to Buy")
    else:
        print("Bad Time to Buy")

def main(zip_code):
    df = load_data('../boston_cleaned_data.csv')
    df = integrated_dfs(df)

    print(f"User Zip Code: {zip_code}")
    model = load_model(ManualCNN)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    predict_zip(df, model, zip_code, scaler)


if __name__ == "__main__":
    main("02115")