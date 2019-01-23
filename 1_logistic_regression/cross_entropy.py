import numpy as np

n = 100 # 100 instances
d = 2 # 2 features

# create fake data
X = np.random.randn(n, d)

# create specific binary data 
X[:50, :] = X[:50, :] - 2*np.ones([50, d]) # class 0
X[50:, :] = X[50:, :] + 2*np.ones([50, d]) # class 1

# labels
Tags = np.array([1]*50 + [2]*50)

# concatenate 1 and X
ones = np.array([[1]*n]).T
Xb = np.concatenate((ones, X), axis=1)

# random initialize the weights
W = np.random.randn(d+1)

# calculate the model output
Z = Xb.dot(W)

# sigmoid
def sigmoid(z):
    return 1/(1+np.exp(-z))

Y = sigmoid(Z)

# cross entropy
def cross_entropy(Tags, Y, num_sample):
    e = 0
    for i in range(num_sample):
        if Tags[i] == 1:
            e -= np.log(Y[i])
        else:
            e -= np.log(1 - Y[i])
    return e

print(cross_entropy(Tags, Y, n))
