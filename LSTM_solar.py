#!c:\users\guanyu tian\anaconda2\envs\py3

import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from statsmodels.tsa.ar_model import AR
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import time

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

def rectify(lst, k):
    return numpy.vstack((lst[k:,:] , lst[:k,:]))

def train_n_predict(dataset, train_size, test_size, look_back):

    # fix random seed for reproducibility
    numpy.random.seed(7)

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets
    train, test = dataset[0:train_size, :], dataset[train_size:train_size + test_size, :]

    # reshape into X=t and Y=t+1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # fit ARIMA model
    model_ARIMA = ARIMA(testY, order=(7, 0, 7))
    model_fit_ARIMA = model_ARIMA.fit(disp=0)
    forecast_ARIMA = model_fit_ARIMA.forecast(steps=287)[0]

    # fit AR model
    model_AR = AR(trainY)
    model_fit_AR = model_AR.fit(10)
    forecast_AR = model_fit_AR.predict(10,287)

    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=10, batch_size=10, verbose=2)

    # make predictions
    testPredict = rectify(numpy.maximum(model.predict(testX),0),1)


    plt.figure()
    plt.plot(testY,label='Actual value')
    plt.plot(forecast_AR, label='AR')
    plt.plot(forecast_ARIMA, label='ARIMA')
    plt.plot(testPredict,label='LSTM')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Normalzied solar power')
    plt.show()

    # invert predictions
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    # calculate root mean squared error
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    percentile_score=testScore/max(testY[0])
    print('Test Score: %.2f RMSE' % (testScore))
    print('Percentile Score: %.2f ' % (percentile_score))
    print(max(testY[0]))

    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset[-287:])
    testPredictPlot[:, :] = testPredict

    # plot baseline and predictions
    plt.figure()
    plt.plot(scaler.inverse_transform(dataset[-287:-1]),label='Real data')
    plt.plot(testPredictPlot,label='Prediction')
    plt.legend()
    return testScore

def main():
    # load the dataset
    print('Loading data...')
    #'''
    train_start= 288 * 263
    train_size = 288 * 100
    test_size = 288 * 2
    look_back = 288
    RMSE=[]

    dataframe = pandas.read_csv('AllData.csv')
    dataset = dataframe.values[train_start:,1:288]
    dataset = dataset.astype('float32')

    dataset_i = dataset[:, 33] # select the prediction location
    rmse=train_n_predict(dataset_i.reshape(-1, 1), train_size, test_size, look_back)

    #for i in range(0, 287):
    #    print('Location '+str(i+1)+':')
    #    dataset_i = dataset[:,i]
    #    RMSE.append(train_n_predict(dataset_i.reshape(-1, 1), train_size, test_size, look_back))
    #print(RMSE)

    #RMSE_average=numpy.mean(RMSE)
    #RMSE_max_value=max(RMSE)
    #RMSE_max_index=RMSE.index(max(RMSE))
    #RMSE_min_value = min(RMSE)
    #RMSE_min_index = RMSE.index(min(RMSE))
    #print('Average RMSE is: '+str(RMSE_average))
    #print('Max: '+str(RMSE_max_value)+' from location '+str(RMSE_max_index+1))
    #print('Min: '+str(RMSE_min_value)+' from location '+str(RMSE_min_index+1))
    # '''


if __name__ == '__main__':
    start = time.clock()
    main()
    elapsed = (time.clock() - start)
    print("Time used:",elapsed)
    plt.show()






