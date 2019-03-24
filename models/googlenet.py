import tensorflow as tf

def GoogLeNet(x, logits=False, training=False):

    def inception(z, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        # 1x1 conv branch
        b1 = tf.layers.conv2d(z, filters=n1x1, kernel_size=[1, 1],
                                padding='valid')
        b1 = tf.layers.batch_normalization(b1, training=training)
        b1 = tf.nn.relu(b1)

        # 1x1 conv -> 3x3 conv branch
        b2 = tf.layers.conv2d(z, filters=n3x3red, kernel_size=[1, 1],
                                padding='valid')
        b2 = tf.layers.batch_normalization(b2, training=training)
        b2 = tf.nn.relu(b2)
        b2 = tf.layers.conv2d(b2, filters=n3x3, kernel_size=[3, 3],
                                padding='same')
        b2 = tf.layers.batch_normalization(b2, training=training)
        b2 = tf.nn.relu(b2)

        # 1x1 conv -> 5x5 conv branch
        b3 = tf.layers.conv2d(z, filters=n5x5red, kernel_size=[1, 1],
                                padding='valid')
        b3 = tf.layers.batch_normalization(b3, training=training)
        b3 = tf.nn.relu(b3)
        b3 = tf.layers.conv2d(b3, filters=n5x5, kernel_size=[3, 3],
                                padding='same')
        b3 = tf.layers.batch_normalization(b3, training=training)
        b3 = tf.nn.relu(b3)
        b3 = tf.layers.conv2d(b3, filters=n5x5, kernel_size=[3, 3],
                                padding='same')
        b3 = tf.layers.batch_normalization(b3, training=training)
        b3 = tf.nn.relu(b3)

        # 3x3 pool -> 1x1 conv branch
        b4 = tf.layers.max_pooling2d(z, pool_size=[1, 1], padding='same',
                                    strides=1)
        b4 = tf.layers.conv2d(b4, filters=pool_planes, kernel_size=[1, 1],
                                padding='valid')
        b4 = tf.layers.batch_normalization(b4, training=training)
        b4 = tf.nn.relu(b4)

        return tf.concat([b1, b2, b3, b4], 3)

    def pre_layer(z):
        z = tf.layers.conv2d(z, filters=192, kernel_size=[3, 3],
                                padding='same')
        z = tf.layers.batch_normalization(z, training=training)
        z = tf.nn.relu(z)

        return z

    with tf.variable_scope('init_conv'):
        z = pre_layer(x)
        z = inception(z, 192,  64,  96, 128, 16, 32, 32)
        z = inception(z, 256, 128, 128, 192, 32, 96, 64)

        z = tf.layers.max_pooling2d(z, pool_size=[3, 3], strides=2,
                                    padding='same')

    with tf.variable_scope('part_1'):
        z = inception(z, 480, 192,  96, 208, 16,  48,  64)
        z = inception(z, 512, 160, 112, 224, 24,  64,  64)
        z = inception(z, 512, 128, 128, 256, 24,  64,  64)
        z = inception(z, 512, 112, 144, 288, 32,  64,  64)
        z = inception(z, 528, 256, 160, 320, 32, 128, 128)
        z = tf.layers.max_pooling2d(z, pool_size=[3, 3], strides=2,
                                    padding='same')

    with tf.variable_scope('part_2'):
        z = inception(z, 832, 256, 160, 320, 32, 128, 128)
        z = inception(z, 832, 384, 192, 384, 48, 128, 128)
        z = tf.layers.max_pooling2d(z, pool_size=[8, 8], padding='valid',
                                    strides=1)

    with tf.variable_scope('mlp'):
        z = tf.layers.flatten(z)

    logits_ = tf.layers.dense(z, units=10, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')

    if logits:
        return y, logits_
    return y


if __name__ == "__main__":
    x = tf.ones(shape=(128, 32, 32, 3))
    y = GoogleNet(x)
    print (y)