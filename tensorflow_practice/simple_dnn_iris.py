"""Simple aritificial neural network for binary clssification"""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


TEST_PROPORTION = 0.3
VARIABLES = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
NUM_CLASS = 2
NUM_NODE = 16
# ACTIVATION = 'sigmoid'
ACTIVATION = 'relu'
EPOCH = 500


def iris_dataset(path="datasets/iris.csv"):
    """Loading iris dataset and split training/testing data"""
    iris_data = pd.read_csv(path)

    iris_x = iris_data[VARIABLES].values
    iris_y = np.array([0 if y=='Iris-setosa' else 1 for y in iris_data['species']])

    return train_test_split(iris_x, iris_y, test_size=TEST_PROPORTION)


def simple_iris_one_layer(train_x, test_x, train_y, test_y):
    """Using 1-layer nn"""
    # define placeholder
    tf_x = tf.placeholder(tf.float32, [None, len(VARIABLES)])
    tf_y = tf.placeholder(tf.int32, [None, ])

    # 1-layer neural network
    if ACTIVATION == 'relu':
        activation_fun = tf.nn.relu
    elif ACTIVATION == 'sigmoid':
        activation_fun = tf.nn.sigmoid
    else:
        # default relu
        activation_fun = tf.nn.relu

    l1 = tf.layers.dense(tf_x, NUM_NODE, activation_fun)
    output = tf.layers.dense(l1, NUM_CLASS)

    # loss
    # https://www.jianshu.com/p/95d0dd92a88a
    loss = tf.losses.sparse_softmax_cross_entropy(labels=tf_y, logits=output)
#     loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf_y,
#                                                                          logits=output))
    
    predict = tf.argmax(output, axis=1)
    accuracy = tf.metrics.accuracy(labels=tf.squeeze(tf_y), predictions=predict)[1]
    opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train = opt.minimize(loss)

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init)
        for i in range(EPOCH):
            sess.run(train, {tf_x: train_x, tf_y: train_y})
            if (i+1) % 50 == 0:
                l, acc = sess.run([loss, accuracy], {tf_x: test_x, tf_y: test_y})
                print('epoch:', (i+1), ',loss:', l, ',accuracy:', acc)


if __name__ == "__main__":
    train_x, test_x, train_y, test_y = iris_dataset()
    simple_iris_one_layer(train_x, test_x, train_y, test_y)
    