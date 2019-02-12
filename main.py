#%% This section is to gather required imports and setting up basic variables
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

rawMCD = pd.read_csv("EOD-MCD.csv").sort_index(ascending=False)
rawMCD.index = rawMCD['Date']

rawDIS = pd.read_csv("EOD-DIS.csv").sort_index(ascending=False)
rawDIS.index = rawDIS['Date']

dataMCD = pd.DataFrame(index=range(0, len(rawMCD)), data=rawMCD.loc[:, ['Date', 'Close']].values, columns=['Date', 'price'])
dataDIS = pd.DataFrame(index=range(0, len(rawDIS)), data=rawDIS.loc[:, ['Date', 'Close']].values, columns=['Date', 'price'])

dataDIS['Date'] = pd.to_datetime(dataDIS['Date'], format="%Y-%m-%d")
dataMCD['Date'] = pd.to_datetime(dataMCD['Date'], format="%Y-%m-%d")
#%% Plots and lists the data that will be used for training and testing
plt.plot(dataMCD['Date'], dataMCD['price'])
# plt.plot(dataDIS['Date'], dataDIS['price'])

plt.xlabel('Time')
plt.ylabel('Stock Price')

plt.savefig('TexFiles/images/graph1.png')

dataMCD.head() #just verify that we read the data properly

#%% Scaling and dividing the data into test and train
scaler = MinMaxScaler()

dataDIS['price'] = scaler.fit_transform(dataDIS.price.values.reshape(-1,1))
dataMCD['price'] = scaler.fit_transform(dataMCD.price.values.reshape(-1,1))

x_train = []
y_train = []
for i in range(60, 4000):
    x_train.append(dataMCD.loc[i-60:i, 'price'])
    y_train.append(dataMCD.loc[i, 'price'])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
#%% This is what we used to train and setup the network 
regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

returns = regressor.fit(x_train, y_train, epochs = 10, batch_size = 32)
#%% Plot the loss over the eopchs

plt.plot(returns.history['loss'])

#%% Partition the test data and test it
x_test = []
for i in range(60, 90):
    x_test.append(dataDIS.loc[i-60:i, 'price'])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted = regressor.predict(x_test)
#%% Plot the predicted
# predicted = scaler.inverse_transform(predicted)da

df = pd.DataFrame(data=predicted, columns=['price'])
df['date'] = dataDIS.loc[:, 'Date']

plt.plot(df['date'], df['price'])
plt.plot(dataDIS.loc[0:30, 'Date'], dataDIS.loc[0:30, 'price'])
