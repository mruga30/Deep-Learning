import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import LSTM
from keras.layers import Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import mean_squared_error
import warnings
from keras.models import Sequential
warnings.filterwarnings('ignore')
import math


def load_data():
    np.random.seed(7)
    df = pd.read_csv('USall.csv', delimiter=',',usecols=['Date', 'Confirmed', 'Deaths', 'Recovered'])
    print('Loaded data from csv...')

    df = df[['Date', 'Confirmed']]
    df.Date = pd.to_datetime(df.Date)
    df = df.set_index('Date')
    # print df.head(5)

    train = df
    scaler = MinMaxScaler()
    scaler.fit(train)
    train = scaler.transform(train)

    n_input = 30
    n_features = 1

    generator = TimeseriesGenerator(train, train, length=n_input+1, batch_size=15)
    model = Sequential()
    model.add(LSTM(47, input_shape=(n_input+1, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit_generator(generator, epochs=100)

    preds_list = []
    batch = train[-n_input-1:].reshape((1, n_input+1, n_features))
    for i in range(n_input+1):
        preds_list.append(model.predict(batch)[0])
        batch = np.append(batch[:, 1:, :], [[preds_list[i]]], axis=1)

    add_dates = [df.index[-2] + DateOffset(days=x) for x in range(0, 32)]
    futureDates = pd.DataFrame(index=add_dates[:], columns=df.columns)

    df_predict = pd.DataFrame(scaler.inverse_transform(preds_list), index=futureDates[-n_input-1:].index, columns=['Predictions'])
    df_proj = pd.concat([df, df_predict], axis=1)
    print(df_proj.tail(14))

    plt.figure(figsize=(20, 5))
    plt.plot(df_proj.index, df_proj['Confirmed'])
    plt.plot(df_proj.index, df_proj['Predictions'], color='r')
    plt.title('COVID-Deaths Projection')
    plt.xlabel('Date')
    plt.ylabel('Number of Deaths = y*(10^6)')
    plt.legend(['Train', 'Predictions'], loc='lower right')
    plt.show()

if __name__ == "__main__":
    load_data()