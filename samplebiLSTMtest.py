import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense

# Function to fetch historical stock prices using yfinance
def get_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

# Function to preprocess the data
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    return scaled_data, scaler

# Function to create sequences for training the model
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i+sequence_length]
        label = data[i+sequence_length]
        sequences.append((seq, label))
    return np.array([s[0] for s in sequences]), np.array([s[1] for s in sequences])

# Fetch historical stock prices
symbol = 'SBIN.NS'
start_date = '2020-01-01'
end_date = '2024-01-22'
stock_data = get_stock_data(symbol, start_date, end_date)

# Preprocess the data
scaled_data, scaler = preprocess_data(stock_data)

# Create sequences for training
sequence_length = 10
X, y = create_sequences(scaled_data, sequence_length)

# Build the Bidirectional LSTM model
model = Sequential()
model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=(X.shape[1], 1)))
model.add(Bidirectional(LSTM(50)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=100, batch_size=32)

# Make predictions
last_sequence = scaled_data[-sequence_length:]
last_sequence = last_sequence.reshape((1, sequence_length, 1))
predicted_value = model.predict(last_sequence)

# Inverse transform the predicted value to get the actual stock price
predicted_price = scaler.inverse_transform(np.array([[predicted_value[0][0]]]))

print(f"Predicted Closing Price for the next day: {predicted_price[0][0]:.2f}")
