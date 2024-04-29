import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


def fetch_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data


def preprocess_data(stock_data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))
    return scaled_data, scaler


def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

def build_and_train_model(X_train, y_train, sequence_length):
    model = Sequential()
    model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(sequence_length, 1)))
    model.add(LSTM(100, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=100, batch_size=32)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse


def predict_stock_price(stock_symbol, start_date, end_date, sequence_length):
    stock_data = fetch_stock_data(stock_symbol, start_date, end_date)
    scaled_data, scaler = preprocess_data(stock_data)
    X, y = create_sequences(scaled_data, sequence_length)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_and_train_model(X_train, y_train, sequence_length)
    
    mse = evaluate_model(model, X_test, y_test)
    
    last_10_days = scaled_data[-sequence_length:]
    next_day_scaled_price = model.predict(last_10_days.reshape(1, sequence_length, 1))
    next_day_price = scaler.inverse_transform(next_day_scaled_price)[0][0]

    y_pred = model.predict(X)
    y_pred = scaler.inverse_transform(y_pred)
    y_actual = scaler.inverse_transform(y)
    
    # Prepare data for plotting
    dates = stock_data.index[-len(y):].strftime('%Y-%m-%d')
    actual_prices = y_actual
    predicted_prices = y_pred
    
    # Create a dictionary with the results
    result = {
    'mse': mse,
    'stock_symbol': stock_symbol,
    'next_day_price': float(next_day_price),  # Convert to float
    'dates': dates.tolist(),  # Convert to a Python list
    'actual_prices': actual_prices.tolist(),  # Convert to a Python list
    'predicted_prices': predicted_prices.tolist(),  # Convert to a Python list
}
    return result


if __name__ == "__main__":
    print("Select a stock to predict or type 'exit' to quit:")
    stocks_to_predict = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'ICICIBANK.NS', 'SBIN.NS', 'WIPRO.NS', 'HCLTECH.NS', 'KOTAKBANK.NS', 'LT.NS']

    for i, stock_symbol in enumerate(stocks_to_predict):
        print(f"{i + 1}. {stock_symbol}")
    
    user_input = input("Enter the number of the stock: ")
    
    if user_input.lower() == 'exit':
        print("Goodbye!")
        
    
    try:
        choice = int(user_input)
        if choice >= 1 and choice <= len(stocks_to_predict):
            selected_stock = stocks_to_predict[choice - 1]
            predict_stock_price(selected_stock, datetime(2020, 1, 1).date(), datetime(2023, 10, 27).date(), 10)
        else:
            print("Invalid choice. Please select a valid stock.")
    except ValueError:
        print("Invalid input. Please enter a valid number.")
