#%% This section is to gather required imports and setting up basic variables
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt


data= pd.read_csv("EOD-MCD.csv")
data['Date'] = pd.to_datetime(data['Date'], format="%Y-%m-%d")
data.index = data['Date']

#%% Plots and lists the data that will be used for training and testing
plt.plot(data['Close'])

data.head() #just verify that we read the data properly