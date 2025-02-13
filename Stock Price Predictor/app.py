import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go

# Load stock data
def load_stock_data(stock_ticker):
    stock_data = yf.download(stock_ticker, start="2010-01-01", end="2023-01-01")
    stock_data['Date'] = stock_data.index
    return stock_data

# Prepare data for machine learning models
def prepare_data(stock_data):
    stock_data['Close'] = stock_data['Close'].shift(-1) # Predicting next day's close price
    stock_data = stock_data.dropna()

    X = stock_data[['Open', 'High', 'Low', 'Volume']].values
    y = stock_data['Close'].values
    return X, y

# Train and evaluate a Linear Regression model
def linear_regression_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    return model, mse, y_pred, y_test

# Prepare LSTM model for time-series prediction
def prepare_lstm_data(stock_data, time_step=60):
    stock_data = stock_data[['Close']].values
    data_X, data_y = [], []
    for i in range(time_step, len(stock_data)):
        data_X.append(stock_data[i-time_step:i, 0])
        data_y.append(stock_data[i, 0])
    
    data_X, data_y = np.array(data_X), np.array(data_y)
    data_X = np.reshape(data_X, (data_X.shape[0], data_X.shape[1], 1))
    
    return data_X, data_y

def build_lstm_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Streamlit app layout
def main():
    st.title("Stock Price Prediction App")

    st.sidebar.header("Stock Selection")
    stock_ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL")
    
    # Load and display stock data
    stock_data = load_stock_data(stock_ticker)
    
    st.header(f"Stock Data for {stock_ticker}")
    st.write(stock_data.tail())
    
    st.subheader("Stock Price Prediction Using Linear Regression")
    X, y = prepare_data(stock_data)
    model, mse, y_pred, y_test = linear_regression_model(X, y)
    
    st.write(f"Mean Squared Error: {mse}")
    st.line_chart(np.concatenate([y_test.reshape(-1, 1), y_pred.reshape(-1, 1)], axis=1), width=600, height=300, use_container_width=True)

    st.subheader("Stock Price Prediction Using LSTM")
    X_lstm, y_lstm = prepare_lstm_data(stock_data)
    model_lstm = build_lstm_model()
    model_lstm.fit(X_lstm, y_lstm, epochs=5, batch_size=32)
    
    st.write("LSTM Model trained for stock price prediction.")
    
    # Visualize predicted stock prices (for example)
    predictions = model_lstm.predict(X_lstm)
    st.subheader("LSTM Predicted Stock Prices")
    st.line_chart(predictions.flatten(), width=600, height=300, use_container_width=True)

# Run the Streamlit app
if __name__ == "__main__":
    main()
