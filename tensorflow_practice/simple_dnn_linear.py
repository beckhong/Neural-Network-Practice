"""Linear examples using 1-layer neural network"""
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf


COEFF_RANGE = (-1, 1)
INPUT_X = \
{
    'x1': {'alpha': 0.45, 'bias': 0.2, 'range': (0, 1)},
    'x2': {'alpha': 0.15, 'bias': 0.7, 'range': (-1, 1)},
    'x3': {'alpha': -0.75, 'bias': 0.1, 'range': (-1, 0)}
}
MUN_INSTANCE = 100
NUM_NODE = 32
EPOCH = 1000


def create_data(x_info):
    x_data = None
    coeff = []
    bias = []
    for _, x_dict in x_info.items():
        _x = np.random.uniform(x_dict['range'][0], x_dict['range'][1],
                               MUN_INSTANCE).astype(np.float32)
        if x_data is None:
            x_data = _x
        else:
            x_data = np.vstack([x_data, _x])
        coeff.append(x_dict['alpha'])
        bias.append(x_dict['bias'])

    print('Coefficient:', coeff)
    print('Bias:', np.sum(bias))

    num_x = len(x_info.keys())
    if num_x > 1:
        x_data = x_data.T
    else:
        x_data = x_data.reshape((x_data.shape[0], num_x))
    y_data_array = np.array(coeff)*x_data + np.array(bias).astype(np.float32)
    y_data_array = y_data_array.reshape(y_data_array.shape[0], num_x)
    y_data = np.expand_dims(np.sum(y_data_array, axis=1), axis=1)

    return x_data, y_data


def add_layer(_input, in_size, out_size, activation_fun=None):
    # random_normal: mean=0, var=1
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]))
    Wx_plus_b = tf.matmul(_input, Weights) + biases

    if activation_fun is None:
        output = Wx_plus_b
    else:
        output = activation_fun(Wx_plus_b)
    return output


def simple_tf_run(x_data, y_data):
    """Simple tensorflow exampe: compare last epoch output with original 
    coefficient and bias.
    """
    num_x = x_data.shape[1]
    tf_x = tf.placeholder(tf.float32, [None, num_x])
    tf_y = tf.placeholder(tf.float32, [None, 1])
    
    # one layer
    l1 = add_layer(tf_x, num_x, NUM_NODE, tf.nn.relu)
    prediction = add_layer(l1, NUM_NODE, 1)

    # loss function
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf_y-prediction), 
                                        reduction_indices=[1]))
    
    # gradient descent
    train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)
    for i in range(EPOCH):
        sess.run(train, feed_dict={tf_x: x_data, tf_y: y_data})
        if (i+1) % 100 == 0:
            l, pred = sess.run([loss, prediction],
                               feed_dict={tf_x: x_data, tf_y: y_data})
            print('epoch:', i+1, ', loss:', l)
    sess.close()
    
    return pred


def plot_result(y, y_hat):
    plt.plot(y, label="observe")
    plt.plot(y_hat, label="prediction")
    plt.savefig("figures/simple_dnn_linear_output.png")
    plt.close()
    

if __name__ == "__main__":
    x_data, y_data = create_data(INPUT_X)
    pred = simple_tf_run(x_data, y_data)
    plot_result(y_data, pred)
    
    
    