import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
from alpha_vantage.timeseries import TimeSeries


import streamlit as st

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter stock ticker:', 'AAPL')

api_key = "W7QUAT6V7SSS6K76"

# Initialize Alpha Vantage API
ts = TimeSeries(key=api_key, output_format="pandas")

# Fetch historical daily stock data
df, meta_data = ts.get_daily(symbol=user_input, outputsize="compact")


# Convert index to datetime and ensure sorting
df.index = pd.to_datetime(df.index)
df = df.sort_index()  # Ensure the index is in ascending order

# Define time frame
start_date = "2010-01-01"
end_date = "2024-12-31"

# Filter only available dates
df_filtered = df[(df.index >= start_date) & (df.index <= end_date)]
df = df_filtered

# Save to CSV
csv_filename = "AAPL_stock_data.csv"
df_filtered.to_csv(csv_filename)

print(f"Data saved to {csv_filename}")

st.subheader('Data from 2010 - 2025 ')
st.write(df.describe())


#visualization

# Rename the column properly
df.rename(columns={'4. close': 'Close Price'}, inplace=True)

# Closing Price vs Time Chart
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close Price'])
st.pyplot(fig)

# Closing Price vs Time Chart with 100 Moving Average
st.subheader('Closing Price vs Time Chart with 100 Moving Average')
ma100 = df['Close Price'].rolling(100).mean()  
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)  
plt.plot(df['Close Price'])
st.pyplot(fig)


# Closing Price vs Time Chart with 100 Moving Average
st.subheader('Closing Price vs Time Chart with 100 MA & 200 MA')
ma100 = df['Close Price'].rolling(100).mean()  
ma200 = df['Close Price'].rolling(200).mean() 
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)  
plt.plot(ma200)
plt.plot(df['Close Price'])
st.pyplot(fig)


