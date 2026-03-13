import pandas as pd
import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense
from keras.optimizers import SGD
from keras.losses import BinaryCrossentropy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('boston_cleaned_data.csv', dtype={'ZipCode': str})

meta_cols = ['RegionID', 'SizeRank', 'ZipCode', 'RegionType',
             'StateName', 'State', 'City', 'Metro', 'CountyName']
date_cols = [c for c in df.columns if c not in meta_cols]

SEQ_LEN = 12
X, y = [], []

TRAIN_RATIO = 0.8
X_train, y_train, X_test, y_test = [], [], [], []

for _, row in df.iterrows():
    prices = row[date_cols].values.astype(float)
    prices = prices[~np.isnan(prices)]

    if len(prices) < SEQ_LEN + 6:
        continue

    split = int(len(prices) * TRAIN_RATIO)

    train_prices = prices[:split]
    test_prices = prices[split:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_prices.reshape(-1,1)).flatten()
    test_scaled = scaler.transform(test_prices.reshape(-1,1)).flatten()

    # TRAIN sequences
    for i in range(SEQ_LEN, len(train_scaled)):
        seq = train_scaled[i-SEQ_LEN:i]
        rolling_avg = np.mean(train_scaled[i-6:i])
        label = 1 if train_scaled[i] < rolling_avg else 0
        X_train.append(seq)
        y_train.append(label)

    # TEST sequences (use last train window for continuity)
    combined = np.concatenate([train_scaled[-SEQ_LEN:], test_scaled])

    for i in range(SEQ_LEN, len(combined)):
        seq = combined[i-SEQ_LEN:i]
        rolling_avg = np.mean(combined[i-6:i])
        label = 1 if combined[i] < rolling_avg else 0
        X_test.append(seq)
        y_test.append(label)

X_train = np.array(X_train).reshape(-1, SEQ_LEN, 1)
X_test = np.array(X_test).reshape(-1, SEQ_LEN, 1)
y_train = np.array(y_train)
y_test = np.array(y_test)

inpx = Input(shape=(SEQ_LEN, 1))
conv_layer = Conv1D(filters=1, kernel_size=3, strides=1, activation=None, padding='valid')(inpx)
pool_layer = MaxPooling1D(pool_size=3)(conv_layer)

flat_G = Flatten()(pool_layer)

hid_layer = Dense(250, activation='relu')(flat_G)
hid_layer2 = Dense(100, activation='tanh')(hid_layer)
out_layer = Dense(1, activation='sigmoid')(hid_layer2)

model = Model(inputs=[inpx], outputs=out_layer)
model.compile(optimizer=SGD(), loss=BinaryCrossentropy, metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc:.4f}")

def predict_zip(zip_code):
    match = df[df['ZipCode'] == str(zip_code).strip()]
    if match.empty:
        print(f"Zip code {zip_code} not found.")
        return

    prices = match.iloc[0][date_cols].values.astype(float)
    prices = prices[~np.isnan(prices)]

    if len(prices) < SEQ_LEN:
        print("Not enough data for prediction.")
        return

    # IMPORTANT: fit scaler on historical prices ONLY for this ZIP
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices.reshape(-1,1)).flatten()

    seq = scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)

    prob = model.predict(seq, verbose=0)[0][0]

    print(f"Zip Code: {zip_code}")
    print(f"Good to rent confidence: {prob:.2%}")
    if prob >= 0.6:
        print("Good time to rent")
    elif prob <= 0.4:
        print("Bad time to rent")
    else:
        print("No clear signal")

predict_zip('02115')