import tensorflow as tf

def MobileNet_v2(x, logits=False, training=False, arch='MobileNetV2'):
    # NOTE: change stride 2 -> 1 for CIFAR10
    cfg = {'MobileNetV2': [(1,  16, 1, 1), (6,  24, 2, 1), (6,  32, 3, 2),
                           (6,  64, 4, 2), (6,  96, 3, 1), (6, 160, 3, 2),
                           (6, 320, 1, 1)]
    }
    params = cfg[arch]

    def block(x, in_planes, out_planes, expansion, stride, name=''):
        with tf.variable_scope('block' + name):
            planes = expansion * in_planes
            z = tf.layers.conv2d(x, filters=planes, kernel_size=[1, 1],
                                padding='valid', use_bias=False)
            z = tf.layers.batch_normalization(z, training=training)
            z = tf.nn.relu(z)
            z = tf.layers.conv2d(z, filters=planes, kernel_size=[3, 3],
                                padding='same', use_bias=False,
                                strides=(stride, stride))
            z = tf.layers.batch_normalization(z, training=training)
            z = tf.nn.relu(z)
            z = tf.layers.conv2d(z, filters=out_planes, kernel_size=[1, 1],
                                padding='valid', use_bias=False)
            z = tf.layers.batch_normalization(z, training=training)
            shortcut = tf.identity(x)
            if stride == 1 or in_planes != planes:
                shortcut = tf.layers.conv2d(shortcut, filters=out_planes,
                                            kernel_size=[1, 1],
                                            padding='valid', use_bias=False)
                shortcut = tf.layers.batch_normalization(shortcut, training=training)
            z = z + shortcut if stride == 1 else z
        return z

    def _make_layer(z, in_planes, name=''):
        for key, (expansion, out_planes, num_blocks, stride) in enumerate(params):
            strides = [stride] + [1]*(num_blocks-1)
            for i, stride in enumerate(strides):
                z = block(z, in_planes, out_planes, expansion, stride, name=name+'_'+str(key)+'_'+str(i))
                in_planes = out_planes
        return z, in_planes

    def _global_avg_pool(x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])

    with tf.variable_scope('init_conv'):
        z = tf.layers.conv2d(x, filters=32, kernel_size=[3, 3],
                            padding='valid', use_bias=False)
        z = tf.layers.batch_normalization(z, training=training)
        z = tf.nn.relu(z)

        z, _ = _make_layer(z, in_planes=32)

        z = tf.layers.conv2d(z, filters=1280, kernel_size=[1, 1],
                            padding='valid', use_bias=False)
        z = tf.layers.batch_normalization(z, training=training)
        z = tf.nn.relu(z)


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
    y = MobileNet_v2(x)
    print (y)