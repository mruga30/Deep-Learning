import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import LSTM
from keras.layers import Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import mean_squared_error
import warnings
from keras.models import Sequential
warnings.filterwarnings('ignore')
import math


def split(data, scaler):
	data.Date = pd.to_datetime(data.Date)
	confirmed = data.set_index('Date')
	train = confirmed[:97]
	test = confirmed[97:]

	# SCALE DATA
	scaler.fit(train)
	train = scaler.transform(train)
	test = scaler.transform(test)

	return train, test


def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)


def load_data():
	np.random.seed(7)
	scaler = MinMaxScaler()
	df = pd.read_csv('USall.csv', delimiter=',', usecols=['Date', 'Confirmed', 'Deaths', 'Recovered'])
	print('Loaded data from csv...')
	df.Date = pd.to_datetime(df.Date)
	df = df.set_index('Date')

	#print df.shape

	data = df.filter(['Recovered'])
	dataset = data.values
	training_data_len = int(math.ceil(len(dataset)*.8))

	scaler = MinMaxScaler(feature_range=(0, 1))
	scaled_data = scaler.fit_transform(dataset)
	#print scaled_data

	train_data = scaled_data[0:training_data_len, :]
	x_train = []
	y_train = []
	for i in range(30, len(train_data)):
		x_train.append(train_data[i-30:i, 0])
		y_train.append(train_data[i, 0])
		'''if i <= 31:
			print x_train
			print y_train
			print ('\n') '''

	x_train, y_train = np.array(x_train), np.array(y_train)

	#print x_train

	x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

	model = Sequential()
	model.add(LSTM(20, return_sequences=True, input_shape=(x_train.shape[1], 1)))
	model.add(LSTM(40, return_sequences=False))
	model.add(Dense(1))

	model.compile(optimizer='adam', loss='mean_squared_error')
	model.fit(x_train, y_train, batch_size=1, epochs=100)

	test_data = scaled_data[training_data_len-31:, :]
	x_test = []
	y_test = dataset[training_data_len-1:, :]

	for i in range(30, len(test_data)):
		x_test.append(test_data[i-30:i, 0])

	x_test = np.array(x_test)
	x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

	predictions = model.predict(x_test)
	predictions = scaler.inverse_transform(predictions)

	rmse = np.sqrt(np.mean(predictions-y_test)**2)
	print(rmse)

	#PLOT
	train = data[:training_data_len]
	valid = data[training_data_len-1:]
	valid['Predictions'] = predictions
	#Visualize the data
	plt.figure(figsize=(16, 8))
	train['Date'] = train.index
	valid['Date'] = valid.index

	plt.title('COVID-19 Model Training Recovered')
	plt.plot(train['Date'], train['Recovered'])
	plt.plot(valid['Date'], valid[['Recovered', 'Predictions']])
	plt.xlabel('Date')
	plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
	plt.show()

if __name__ == "__main__":
	load_data()