import yfinance as yf
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import joblib

os.makedirs("models", exist_ok=True)

def train_lstm(stock):
    print(f"Training LSTM for {stock}...")

    df = yf.download(stock, start="2020-01-01")
    data = df[['Close']]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    window = 10

    for i in range(window, len(scaled_data)):
        X.append(scaled_data[i-window:i])
        y.append(scaled_data[i])

    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    model.save(f"models/{stock}_lstm.h5")
    joblib.dump(scaler, f"models/{stock}_scaler.pkl")

    print(f"LSTM saved for {stock}")

if __name__ == "__main__":
    stocks = ["AAPL", "TSLA", "GOOG", "MSFT", "AMZN"]
    for s in stocks:
        train_lstm(s)

    print("✅ All LSTM models trained!")