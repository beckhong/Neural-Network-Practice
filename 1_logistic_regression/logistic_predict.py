import numpy as np
from pre_processing import get_binary


X, Y = get_binary('ecommerce_data.csv') # X shape: (398, 8)

d = X.shape[1] 
W = np.random.randn(d) # (8, )
b = 0

# logistic function
def sigmoid(z):
    return 1/(1+np.exp(-z))

# forward propagation
def forward(X, W, b):
    return sigmoid(X.dot(W)+b)

# accuracy
def classification_rate(P, Y):
    return np.mean(P == Y)

predictions = np.round(forward(X, W, b))
classification_rate(predictions, Y) # 0.32, not good
