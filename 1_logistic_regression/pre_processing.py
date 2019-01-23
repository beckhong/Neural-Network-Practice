import numpy as np
import pandas as pd


def get_data(path):
    '''Get ecommerce data and do data pre-processing
    '''
    df = pd.read_csv(path)

    # change dataframe to numpy array
    data = df.values

    # shuffle data
    np.random.shuffle(data)

    # split features(user_action) and labels(user_action, number of lebel: 4)
    X = data[:, :-1] # number of feature=5
    Y = data[:, -1].astype(np.int32) # user_action

    # create a new matrix X2 with the correct number of columns
    num_level = len(set(Y))
    n, d = X.shape # (500, 5)
    X2 = np.zeros((n, d+num_level-1)) # (500, 8)

    # pre-processing data(is_mobile, n_products_viewed, visit_duration, and is_returning_visitor) to X2
    X2[:, 0:d-1] = X[:, :d-1]

    # one-hot encode the categorical data
    # treat time_of_day column
    for idx in range(n):
        t = int(X[idx, d-1])
        X2[idx, d+t-1] = 1

    return X2, Y
    

def get_binary(data):
    # return only the data from the first 2 classes
    X, Y = get_data(data)
    X = X[Y <= 1]
    Y = Y[Y <= 1]

    return X, Y
