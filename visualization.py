#!c:\users\guanyu tian\anaconda2\envs\py3

import numpy
import matplotlib.pyplot as plt
import pandas
import math
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

def look(dataset, train_size, test_size, look_back):

    # fix random seed for reproducibility
    numpy.random.seed(7)

    '''
    # split into train and test sets
    train, test = dataset[0:train_size, :], dataset[train_size:train_size + test_size, :]
    plt.figure()
    plt.plot(test)
    # plt.title('Outdoor air temperature reference')
    plt.xlabel('Time step')
    plt.ylabel('Original solar power/kW')
    plt.legend()
    plt.show()
    '''

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets
    train, test = dataset[0:train_size, :], dataset[train_size:train_size + test_size, :]

    plt.figure()
    plt.plot(test)
    #plt.title('Outdoor air temperature reference')
    plt.xlabel('Time step')
    plt.ylabel('Normalized solar power')
    plt.legend()
    plt.show()

    # reshape into X=t and Y=t+1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)



    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))



def main():

    # load the dataset
    print('Loading data...')
    train_start= 288 * 263
    train_size = 288 * 100
    test_size = 288 * 2
    look_back = 288
    RMSE=[]

    dataframe = pandas.read_csv('AllData.csv')
    dataset = dataframe.values[train_start:,1:288]
    dataset = dataset.astype('float32')

    dataset_i = dataset[:, 33] # select the prediction location
    rmse=look(dataset_i.reshape(-1, 1), train_size, test_size, look_back)



if __name__ == '__main__':
    start = time.clock()
    main()
    elapsed = (time.clock() - start)
    print("Time used:",elapsed)
    plt.show()






