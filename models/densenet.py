import tensorflow as tf
import numpy as np

def DenseNet(x, logits=False, training=False, arch='DenseNet121'):
    cfg = {
        'DenseNet121': [[6,12,24,16], 32],
        'DenseNet169': [[6,12,32,32], 32],
        'DenseNet201': [[6,12,48,32], 32],
        'DenseNet161': [[6,12,36,24], 48],
        'DenseNet':    [[6,12,24,16], 12],   # densenet_cifar
    }
    params = cfg[arch]
    nblocks, growth_rate = params
    reduction = 0.5

    def _bottleneck(x, in_planes, growth_rate, name=''):
        with tf.variable_scope('bottleneck' + name):
            z = tf.layers.batch_normalization(x, training=training)
            z = tf.nn.relu(z)
            z = tf.layers.conv2d(x, filters=4*growth_rate, kernel_size=[1, 1],
                                 padding='valid', use_bias=False)

            z = tf.layers.batch_normalization(z, training=training)
            z = tf.nn.relu(z)
            z = tf.layers.conv2d(z, filters=growth_rate, kernel_size=[3, 3],
                                 padding='same', use_bias=False)
            z = tf.concat([z, x], axis=3)

        return z

    def _transition(x, in_planes, out_planes, name=''):
        with tf.variable_scope('transition' + name):
            z = tf.layers.batch_normalization(x, training=training)
            z = tf.nn.relu(z)
            z = tf.layers.conv2d(x, filters=out_planes, kernel_size=[1, 1],
                                 padding='valid', use_bias=False)

            z = tf.layers.average_pooling2d(z, pool_size=[2, 2], strides=2)

        return z

    def _make_dense_layer(z, block, in_planes, nblock, name=''):
        for i in range(nblock):
            z = block(z, in_planes, growth_rate, name=name+str(i))
            in_planes += growth_rate

        return z

    num_planes = 2 * growth_rate
    block = _bottleneck
    with tf.variable_scope('init_conv'):
        z = tf.layers.conv2d(x, filters=num_planes, kernel_size=[3, 3],
                             padding='same', use_bias=False)

    with tf.variable_scope('dense_block0'):
        z = _make_dense_layer(z, block, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(np.floor(num_planes * reduction))
        z = _transition(z, num_planes, out_planes)
        num_planes = out_planes

    with tf.variable_scope('dense_block1'):
        z = _make_dense_layer(z, block, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(np.floor(num_planes * reduction))
        z = _transition(z, num_planes, out_planes)
        num_planes = out_planes

    with tf.variable_scope('dense_block2'):
        z = _make_dense_layer(z, block, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        out_planes = int(np.floor(num_planes * reduction))
        z = _transition(z, num_planes, out_planes)
        num_planes = out_planes

    with tf.variable_scope('dense_block3'):
        z = _make_dense_layer(z, block, num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate


    with tf.variable_scope('mlp'):
        z = tf.layers.batch_normalization(x, training=training)
        z = tf.nn.relu(z)
        z = tf.layers.average_pooling2d(z, pool_size=[4, 4], strides=4)
        z = tf.layers.flatten(z)

    logits_ = tf.layers.dense(z, units=10, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')

    if logits:
        return y, logits_
    return y


if __name__ == "__main__":
    x = tf.ones(shape=(128, 32, 32, 3))
    y = DenseNet(x)
    print (y)