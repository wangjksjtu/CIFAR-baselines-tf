import tensorflow as tf

def All_CNNs(x, logits=False, training=False):
    with tf.variable_scope('convs0'):
        z = tf.layers.conv2d(x, filters=32, kernel_size=(3, 3),
                                padding='same', activation=tf.nn.relu)
        z = tf.layers.conv2d(z, filters=64, kernel_size=(3, 3),
                                padding='same', activation=tf.nn.relu)
        z = tf.layers.conv2d(z, filters=128, kernel_size=(3, 3), strides=(2, 2),
                                padding='same')
        z = tf.layers.dropout(z, 0.5, training=training)

    with tf.variable_scope('convs1'):
        z = tf.layers.conv2d(z, filters=128, kernel_size=(3, 3),
                                padding='same', activation=tf.nn.relu)
        z = tf.layers.conv2d(z, filters=128, kernel_size=(3, 3),
                                padding='same', activation=tf.nn.relu)
        z = tf.layers.conv2d(z, filters=128, kernel_size=(3, 3), strides=(2, 2),
                                padding='same')
        z = tf.layers.dropout(z, 0.5, training=training)

    with tf.variable_scope('convs2'):
        z = tf.layers.conv2d(z, filters=128, kernel_size=(3, 3),
                                padding='same', activation=tf.nn.relu)
        z = tf.layers.conv2d(z, filters=128, kernel_size=(1, 1),
                                padding='valid', activation=tf.nn.relu)
        z = tf.layers.conv2d(z, filters=10, kernel_size=(1, 1), strides=(2, 2),
                                padding='valid')

    logits_ = tf.reduce_mean(z, [1, 2])
    y = tf.nn.softmax(logits_, name='ybar')

    if logits:
        return y, logits_
    return y


if __name__ == "__main__":
    x = tf.ones(shape=(128, 32, 32, 3))
    y = All_CNNs(x)
    print (y)