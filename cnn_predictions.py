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

scaler = MinMaxScaler()
scaled_prices = pd.DataFrame(
    scaler.fit_transform(df[date_cols].T).T,
    columns=date_cols
)

for idx, row in scaled_prices.iterrows():
    prices = row.values.astype(float)
    for i in range(SEQ_LEN, len(prices)):
        seq = prices[i - SEQ_LEN:i]
        rolling_avg = np.mean(prices[i - 6:i])
        label = 1 if prices[i] < rolling_avg else 0
        X.append(seq)
        y.append(label)

X = np.array(X).reshape(-1, SEQ_LEN, 1)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
        print(f"Zip code {zip_code} not found. Available: {sorted(df['ZipCode'].unique())}")
        return

    prices = scaled_prices.loc[match.index[0]].values.astype(float)
    seq = prices[~np.isnan(prices)][-SEQ_LEN:].reshape(1, SEQ_LEN, 1)
    prob = model.predict(seq, verbose=0)[0][0]

    print(f"Zip Code: {zip_code}")
    print(f"Good to rent confidence: {prob:.2%}")
    if prob >= 0.6:
        print("Good time to rent")
    elif prob <= 0.4:
        print("Bad time to rent")
    else:
        print("No signal")

predict_zip(input("Enter a Boston-area zip code: "))