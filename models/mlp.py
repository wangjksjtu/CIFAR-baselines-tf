import tensorflow as tf

def MLP(x, logits=False, training=False):
    with tf.variable_scope('flatten'):
        z = tf.layers.flatten(x)

    with tf.variable_scope('mlp'):
        z = tf.layers.dense(z, units=128, activation=tf.nn.relu)
        z = tf.layers.dense(z, units=128, activation=tf.nn.relu)
        z = tf.layers.dense(z, units=64, activation=tf.nn.relu)

    logits_ = tf.layers.dense(z, units=10, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')

    if logits:
        return y, logits_
    return y


if __name__ == "__main__":
    x = tf.ones(shape=(128, 32, 32, 3))
    y = MLP(x)
    print (y)