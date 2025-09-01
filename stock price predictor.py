import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential #In Pythob cmd terminal paste this command: py -m pip install tensorflow
from tensorflow.keras.layers import LSTM, Dense, Dropout

# -------------------------
# 1. Download Stock Data
# -------------------------
ticker = '' #Enter the ticker symbol of any stock
data = yf.download(ticker, start="2018-01-01", end="2025-08-01")

if data.empty:
    print("No data returned. Check ticker or date range.")
else:
    # Use Adjusted Close price (or Close if not available)
    adj_close = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']

    # Reset index so Date is a column
    df = adj_close.reset_index()
    df.columns = ['Date', 'Price']

    # -------------------------
    # 2. Preprocess Data
    # -------------------------
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Price']].values)

    seq_len = 60  # how many past days to use for each prediction
    X, y = [], []

    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i - seq_len:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # reshape for LSTM

    # -------------------------
    # 3. Build LSTM Model
    # -------------------------
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # prediction output

    model.compile(optimizer='adam', loss='mean_squared_error')

    # -------------------------
    # 4. Train the Model
    # -------------------------
    model.fit(X, y, epochs=20, batch_size=32)

    # -------------------------
    # 5. Predict Future Prices
    # -------------------------
    future_days = 180
    last_seq = scaled_data[-seq_len:]
    future_preds = []

    current_seq = last_seq.reshape(1, seq_len, 1)

    for _ in range(future_days):
        pred = model.predict(current_seq, verbose=0)
        future_preds.append(pred[0, 0])
        # Update sequence with the new prediction
        current_seq = np.append(current_seq[:, 1:, :], [[[pred[0, 0]]]], axis=1)


    # Inverse transform predictions back to USD prices
    future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

    # -------------------------
    # 6. Plot Results
    # -------------------------
    future_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=future_days)

    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Price'], label="Historical Price")
    plt.plot(future_dates, future_preds, label="Predicted Price (LSTM)", linestyle="--")
    plt.title(f"{ticker} Stock Price Prediction with LSTM")
    plt.xlabel("Date")
    plt.ylabel("Price ($USD)")
    plt.legend()
    plt.grid(True)
    plt.show()

