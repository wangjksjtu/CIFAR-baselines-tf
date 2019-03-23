import tensorflow as tf

def MobileNet(x, logits=False, training=False):
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def _block(x, in_planes, planes, stride=1, name=''):
        with tf.variable_scope('block' + name):
            z = tf.layers.conv2d(x, filters=planes, kernel_size=[3, 3],
                                padding='same', use_bias=False,
                                strides=(stride, stride))
            z = tf.layers.batch_normalization(z, training=training)
            z = tf.nn.relu(z)

            z = tf.layers.conv2d(z, filters=planes, kernel_size=[1, 1],
                                padding='valid', use_bias=False)
            z = tf.layers.batch_normalization(z, training=training)
            z = tf.nn.relu(z)
        return z

    def _global_avg_pool(x):
        return tf.layers.average_pooling2d(x, pool_size=[2,2], strides=2)


    with tf.variable_scope('init_conv'):
        z = tf.layers.conv2d(x, filters=32, kernel_size=[3,3],
                            padding='same', use_bias=False)
        z = tf.layers.batch_normalization(z, training=training)
        z = tf.nn.relu(z)

    block = _block
    num_blocks = cfg

    def _make_layer(z, in_planes, name=''):
        for index,(i) in enumerate(num_blocks):
            planes = i if isinstance(i, int) else i[0]
            stride = 1 if isinstance(i, int) else i[1]
            z = block(z, in_planes, planes, stride, name=name+str(index))
            in_planes = planes
        return z, in_planes

    z, _ = _make_layer(z, in_planes=32)

    with tf.variable_scope('mlp'):
        z = _global_avg_pool(z)
        z = tf.layers.flatten(z)

    logits_ = tf.layers.dense(z, units=10, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')

    if logits:
        return y, logits_
    return y


if __name__ == "__main__":
    x = tf.ones(shape=(128, 32, 32, 3))
    y = MobileNet(x)
    print (y)