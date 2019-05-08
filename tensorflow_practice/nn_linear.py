"""Linear examples"""
import numpy as np
import tensorflow as tf


COEFF_RANGE = (-1, 1)
INPUT_X = \
{
    'x1': {'alpha': 0.45, 'bias': 0.2, 'range': (0, 1)},
    'x2': {'alpha': 0.15, 'bias': 0.7, 'range': (-1, 1)},
    'x3': {'alpha': -0.75, 'bias': 0.1, 'range': (-1, 0)}
}
MUN_INSTANCE = 100
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
    y_data = np.sum(y_data_array, axis=1)

    return x_data, y_data


def simple_tf_run(x_data, y_data):
    """Simple tensorflow exampe: compare last epoch output with original 
    coefficient and bias.
    """
    num_x = x_data.shape[1]
    Weight = tf.Variable(tf.random_uniform(shape=[num_x], minval=COEFF_RANGE[0], 
                                           maxval=COEFF_RANGE[1]))
    bias = tf.Variable(tf.zeros(shape=[num_x]))

    y = tf.reduce_sum(x_data*Weight + bias, axis=1)

    # loss function
    loss = tf.reduce_mean(tf.square(y-y_data))
    
    # gradient descent
    opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = opt.minimize(loss)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)
    for i in range(EPOCH):
        sess.run(train)
        if (i+1) % 100 == 0:
            print('epoch:', i+1, ', weights:', sess.run(Weight), 
                  ', bias:', np.sum(sess.run(bias)))
    sess.close()


if __name__ == "__main__":
    x_data, y_data = create_data(INPUT_X)
    simple_tf_run(x_data, y_data)
    