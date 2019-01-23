'''
Logistic function testing.
'''
import numpy as np

# create random data
n = 20
d = 2

X = np.random.randn(n, d)
ones = np.ones((n, 1))
Xb = np.concatenate((ones, X), axis=1)

w = np.random.randn(d+1)
z = Xb.dot(w)

def sigmoid(z):
    return 1/(1+np.exp(-z))

print(sigmoid(z))
