import tensorflow as tf

def LeNet(x, logits=False, training=False):
    with tf.variable_scope('conv0'):
        z = tf.layers.conv2d(x, filters=6, kernel_size=[5, 5],
                             padding='valid', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('conv1'):
        z = tf.layers.conv2d(z, filters=16, kernel_size=[5, 5],
                             padding='valid', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('flatten'):
        shape = z.get_shape().as_list()
        z = tf.reshape(z, [-1, np.prod(shape[1:])])

    with tf.variable_scope('mlp'):
        z = tf.layers.dense(z, units=120, activation=tf.nn.relu)
        z = tf.layers.dense(z, units=84, activation=tf.nn.relu)

    logits_ = tf.layers.dense(z, units=10, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')

    if logits:
        return y, logits_
    return y


if __name__ == "__main__":
    x = tf.ones(shape=(128, 32, 32, 3))
    y = LeNet(x)
    print (y)