#%% This section is to gather required imports and setting up basic variables
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

raw = pd.read_csv("EOD-MCD.csv").sort_index(ascending=False)
raw['Date'] = pd.to_datetime(raw['Date'], format="%Y-%m-%d")
raw.index = raw['Date']

data = pd.DataFrame(index=range(0, len(raw)), data=raw.loc[:, ['Date', 'Close']].values, columns=['Date', 'Price'])
#%% Plots and lists the data that will be used for training and testing
plt.plot(data['Date'], data['Price'])

data.head() #just verify that we read the data properly

#%% Scaling and dividing the data into test and train
scaler = MinMaxScaler()

train_data = data.loc[:3095, 'Price']
test_data = data.loc[3095:, 'Price']
