import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

os.makedirs("models", exist_ok=True)

def train_model(stock):
    print(f"Training model for {stock}...")
    
    df = yf.download(stock, start="2020-01-01")
    
    df['Prediction'] = df['Close'].shift(-1)
    df = df.dropna()
    
    X = df[['Close']]
    y = df['Prediction']
    
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    
    joblib.dump(model, f"models/{stock}_model.pkl")
    print(f"Saved: models/{stock}_model.pkl")

if __name__ == "__main__":
    stocks = ["AAPL", "TSLA", "GOOG", "MSFT", "AMZN"]
    
    for stock in stocks:
        train_model(stock)

    print("✅ All models trained!")