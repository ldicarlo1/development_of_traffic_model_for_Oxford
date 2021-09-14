import numpy as np

''' Lists of functions for preprocessing data before modeling.
'''

def train_test_split(data, train_portion):
    '''Function splits data into testing and training using a split ratio.

    Input Args:
    
    data (array-obj): traffic or weather data

    train portion (float): percentage of training data to be split (value between 0 or 1)

    Return Args:

    train data (array-obj): split of training data

    test data (array-obj): split of testing data

    '''
    time_len = data.shape[1]
    train_size = int(time_len * train_portion)
    train_data = np.array(data[:, :train_size])
    test_data = np.array(data[:, train_size:])

    return train_data, test_data

def scale_data(train_data, test_data):
    '''Function normalizes the data between 0 and 1.

    Input Args:
    
    train_data (array-obj): training traffic or weather data to be scaled (value between 0 or 1)

    test_data (array-obj): testing traffic or weather data to be scaled (value between 0 or 1 )

    Return Args:

    train scaled (array-obj): scaled training data

    test scale (array-obj): scaled testing data

    '''

    train_max = train_data.max()
    train_min = train_data.min()
    
    train_scaled = (train_data - train_min) / (train_max - train_min)
    test_scaled = (test_data - train_min) / (train_max - train_min)

    return train_scaled, test_scaled

def sequence_data_preparation(seq_len, pre_len, train_data, test_data):
    '''Function prepares a sequence for the LSTM input by creating a sequence of predictors and a target prediction N steps in advance.
    Input Args:
    
    seq_len (float): number of timesteps into the past to predict the future time series value.

    pre_len (float): number of timesteps into the future that the model should aim to predict.

    train_data (array-obj): training data

    test_data (array-obj): testing data


    Return Args:

    trainX (array-obj): training predictors
    trainY (array-obj): training predictions
    testX (array-obj): testing predictors
    testY (array-obj): training predictions


    '''
    
    trainX, trainY, testX, testY = [], [], [], []

    for i in range(train_data.shape[1] - int(seq_len + pre_len - 1)):
        a = train_data[:, i : i + seq_len + pre_len]
        trainX.append(a[:, :seq_len])
        trainY.append(a[:, -1])

    for i in range(test_data.shape[1] - int(seq_len + pre_len - 1)):
        b = test_data[:, i : i + seq_len + pre_len]
        testX.append(b[:, :seq_len])
        testY.append(b[:, -1])

    trainX = np.array(trainX)
    trainY = np.array(trainY)
    testX = np.array(testX)
    testY = np.array(testY)

    return trainX, trainY, testX, testY