"""
Simple aritificial neural network for multi-clssification
https://github.com/MorvanZhou/tutorials/blob/master/tensorflowTUT/tf16_classification/full_code.py

id: Sequential id
huml: Length of Humerus (mm)
humw: Diameter of Humerus (mm)
ulnal: Length of Ulna (mm)
ulnaw: Diameter of Ulna (mm)
feml: Length of Femur (mm)
femw: Diameter of Femur (mm)
tibl: Length of Tibiotarsus (mm)
tibw: Diameter of Tibiotarsus (mm)
tarl: Length of Tarsometatarsus (mm)
tarw: Diameter of Tarsometatarsus (mm)
type: Ecological Group
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler


COLS = ['huml', 'humw', 'ulnal', 'ulnaw', 'feml', 
        'femw', 'tibl', 'tibw', 'tarl', 'tarw']
TEST_PROPORTION = 0.3
MINIMAX_SCALER = (-1, 1)
LR = 0.09
EPOCH = 1500


def bird_dataset(path="datasets/bird.csv"):
    """Loading bird dataset and split training/testing data"""
    bird_data = pd.read_csv(path)

    # feed -1 to NaN
    bird_data = bird_data.fillna(-1)

    # replace -1 to mean of each bird type 
    bird_type_class = np.unique(bird_data['type'])
    bird_summary = bird_data.groupby('type').mean()

    bird_data_no_miss = pd.DataFrame()
    for _type in bird_type_class:
        col_data = bird_data[bird_data['type']==_type]
        type_summary = bird_summary[bird_summary.index==_type]
        for col in COLS:
            np.place(col_data[col].values, col_data[col] == -1, 
                     np.round(type_summary[col].values, 2))
        bird_data_no_miss = bird_data_no_miss.append(col_data)
    bird_data_no_miss = bird_data_no_miss.reset_index(drop=True)

    # define dependent/independent variables
    bird_x = bird_data_no_miss[COLS].values
    bird_y = bird_data_no_miss['type']

    # one-hot encoding
    encoder = LabelBinarizer()
    bird_y = encoder.fit_transform(bird_y)

    return train_test_split(bird_x, bird_y, test_size=TEST_PROPORTION)


def add_layer(_input, in_size, out_size, activation_fun=None):
    """
    y = xW + b
    shape: y: (#, 6); x: (#, 10); W: (10, 6); b: (1, 6)
    """
    # random_normal: mean=0, var=1
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]))
    Wx_plus_b = tf.matmul(_input, Weights) + biases

    if activation_fun is None:
        output = Wx_plus_b
    else:
        output = activation_fun(Wx_plus_b)
    return output


def simple_bird_one_layer(train_x, test_x, train_y, test_y):
    # mini-max transform to similar scale
    scaler = MinMaxScaler(feature_range=MINIMAX_SCALER)
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)

    # placeholder
    tf_x = tf.placeholder(tf.float32, [None, 10], name='tf_x')
    tf_y = tf.placeholder(tf.float32, [None, 6], name='tf_y')

    # prediction with one layer, multi-class using softmax
    prediction = add_layer(tf_x, 10, 6, activation_fun=tf.nn.softmax)

    # cross entropy
    loss = tf.reduce_mean(-tf.reduce_sum(tf_y*tf.log(prediction),
                                         reduction_indices=[1]))

    # using Adam is better!
    # train = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)
    train = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss)

    def compute_accuracy(test_x, test_y, prediction):
        predict = sess.run(prediction, feed_dict={tf_x: test_x})
        correct = tf.equal(tf.argmax(predict, 1), tf.argmax(test_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        result = sess.run(accuracy, feed_dict={tf_x: test_x, tf_y: test_y})
        return result

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init)
        for i in range(EPOCH):
            sess.run(train, {tf_x: train_x, tf_y: train_y})
            if (i+1) % 100 == 0:
                l = sess.run(loss, {tf_x: train_x, tf_y: train_y})
                acc = compute_accuracy(test_x, test_y, prediction)
                print('loss:', l, ',accuracy:', acc)


if __name__ == "__main__":
    train_x, test_x, train_y, test_y = bird_dataset()
    simple_bird_one_layer(train_x, test_x, train_y, test_y)
