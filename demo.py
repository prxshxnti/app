import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
from alpha_vantage.timeseries import TimeSeries
import streamlit as st

st.title('Stock Trend Prediction')

# User input for stock ticker
user_input = st.text_input('Enter stock ticker:', 'AAPL')

# User input for date range
start_date, end_date = st.date_input("Select Date Range:", 
                                     [pd.to_datetime("2010-01-01"), pd.to_datetime("2024-12-31")])

api_key = "W7QUAT6V7SSS6K76"

# Initialize Alpha Vantage API
ts = TimeSeries(key=api_key, output_format="pandas")

# Fetch historical daily stock data
df, meta_data = ts.get_daily(symbol=user_input, outputsize="full")

# Convert index to datetime and sort
df.index = pd.to_datetime(df.index)
df = df.sort_index()

# Filter data based on user-selected date range
df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]

# Rename columns
df.rename(columns={'4. close': 'Close Price'}, inplace=True)

st.subheader(f'Data from {start_date} to {end_date}')
st.write(df.describe())

# Visualization: Closing Price vs Time
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close Price'], label='Closing Price', color='blue')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
st.pyplot(fig)

# Closing Price vs Time with 100 Moving Average
st.subheader('Closing Price vs Time Chart with 100 Moving Average')
ma100 = df['Close Price'].rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close Price'], label='Closing Price', color='blue', alpha=0.5)
plt.plot(ma100, label='100-Day MA', color='green', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
st.pyplot(fig)

# Closing Price vs Time with 100 & 200 Moving Averages
st.subheader('Closing Price vs Time Chart with 100 MA & 200 MA')
ma200 = df['Close Price'].rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close Price'], label='Closing Price', color='blue', alpha=0.5)
plt.plot(ma100, label='100-Day MA', color='green', linewidth=2)
plt.plot(ma200, label='200-Day MA', color='red', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
st.pyplot(fig)


#split data into training and testing

data_training = pd.DataFrame(df['Close Price'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close Price'][int(len(df) * 0.70) : int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

data_training_array = scaler.fit_transform(data_training)

# #split the training data
# x_train = []
# y_train = []

# for i in range(100, data_training_array.shape[0]):
#     x_train.append(data_training_array[i - 100 : i])
#     y_train.append(data_training_array[i, 0])

# x_train, y_train = np.array(x_train), np.array(y_train)

#load the model

model = load_model('keras_model.h5')

#make prediction

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range (100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)


y_predicted = model.predict(x_test)

scaler = scaler.scale_
scaler_factor = 1 / scaler[0]
y_predicted = y_predicted * scaler_factor
y_test = y_test * scaler_factor

#plot the results
st.subheader('Predictions vs Orginal')

fig2 = plt.figure(figsize = (12, 6))
plt.plot(y_test,'b', label = 'Orginal Price')
plt.plot(y_predicted,'r', label = ' Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
