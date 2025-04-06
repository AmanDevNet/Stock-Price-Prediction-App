import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib
import os

# Parameters
stock_symbol = "TSLA"
start_date = "2020-01-01"
end_date = "2025-04-01"
lookback = 60

# Fetch stock data
data = yf.download(stock_symbol, start=start_date, end=end_date)
if data.empty or 'Close' not in data:
    raise ValueError(f"No data fetched for symbol '{stock_symbol}'")

df = data[['Close']].copy()

# Normalize closing prices
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Prepare sequences
X, y = [], []
for i in range(lookback, len(scaled_data)):
    X.append(scaled_data[i - lookback:i, 0])
    y.append(scaled_data[i, 0])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Train/test split
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# Save model and scaler
model.save('stock_model.h5')
joblib.dump(scaler, 'scaler_stock.pkl')

print("[✓] Model and scaler saved successfully!")
