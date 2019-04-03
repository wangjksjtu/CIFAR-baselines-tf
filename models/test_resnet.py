import tensorflow as tf
import numpy as np

def TResNet(x, logits=False, training=False, arch='TResNet18'):
    # ResNet

    cfg = {
        'TResNet18':  ['BasicBlock', [2, 2, 2, 2]],
        'TResNet34':  ['BasicBlock', [3, 4, 6, 3]],
        'TResNet50':  ['BasicBlock', [3, 4, 6 ,3]],
        'TResNet101': ['BottleNeck', [3, 4, 23 ,3]],
        'TResNet152': ['BottleNeck', [3, 8, 36 ,3]],
    }
    params = cfg[arch]

    def _basic_block(x, in_planes, planes, stride=1, name=''):
        expansion = 1
        with tf.variable_scope('block' + name):
            z = tf.layers.conv2d(x, filters=planes, kernel_size=[3, 3],
                                padding='same', use_bias=False,
                                strides=(stride, stride))
            # z = tf.layers.batch_normalization(z, training=training)
            z = _batch_norm('basic_bn_'+name, z)
            # z = _relu(z, 0.1)
            z = tf.nn.relu(z)
            z = tf.layers.conv2d(z, filters=planes, kernel_size=[3, 3],
                                 padding='same', use_bias=False)
            z = _batch_norm('basic_bn_'+name, z)
            # z = tf.layers.batch_normalization(z, training=training)

            shortcut = tf.identity(x)
            if stride != 1 or in_planes != planes * expansion:
                shortcut = tf.layers.conv2d(shortcut, filters=expansion*planes,
                                            kernel_size=[1, 1],
                                            padding='same', use_bias=False,
                                            strides=(stride, stride))
                shortcut = tf.layers.batch_normalization(shortcut, training=training)
            z += shortcut
            z = tf.nn.relu(z)

        return z

    def _bottleneck(x, in_planes, planes, stride=1, name=''):
        expansion = 4
        with tf.variable_scope('bottleneck' + name):
            z = tf.layers.conv2d(x, filters=planes, kernel_size=[1, 1],
                                 padding='valid', use_bias=False)
            # z = tf.layers.batch_normalization(z, training=training)

            z = _batch_norm('bottle_bn_'+name, z)
            # z = _relu(z, 0.1)
            z = tf.nn.relu(z)

            z = tf.layers.conv2d(z, filters=planes, kernel_size=[3, 3],
                                 padding='same', use_bias=False,
                                 strides=(stride, stride))

            # z = tf.layers.batch_normalization(z, training=training)
            # z = tf.nn.relu(z)
            z = _batch_norm('bottle_bn_'+name, z)
            # z = _relu(z, 0.1)
            z = tf.nn.relu(z)
            z = _batch_norm('bottle_bn_'+name, z)

            z = tf.layers.conv2d(z, filters=expansion*planes, kernel_size=[1, 1],
                                 padding='valid', use_bias=False)
            z = _batch_norm('bottle_bn', z)
            # z = tf.layers.batch_normalization(z, training=training)

            shortcut = tf.identity(x)

            if stride != 1 or in_planes != planes * expansion:
                shortcut = tf.layers.conv2d(shortcut, filters=expansion*planes,
                                            kernel_size=[1, 1],
                                            padding='same', use_bias=False,
                                            strides=(stride, stride))
                shortcut = tf.layers.batch_normalization(shortcut, training=training)

            z += shortcut
            z = tf.nn.relu(z)
        return z

    def _batch_norm(name, x):
        """Batch normalization."""
        with tf.name_scope(name):
            return tf.contrib.layers.batch_norm(
                inputs=x,
                decay=.9,
                center=True,
                scale=True,
                activation_fn=None,
                updates_collections=None,
                is_training=training)

    def _decay():
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find('DW') > 0:
                costs.append(tf.nn.l2_loss(var))

        return tf.add_n(costs)


    def _conv(name, x, filter_size, in_filters, out_filters, strides):
        """Convolution."""
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            kernel = tf.get_variable(
                'DW', [filter_size, filter_size, in_filters, out_filters],
                tf.float32, initializer=tf.random_normal_initializer(
                stddev=np.sqrt(2.0/n)))
        return tf.nn.conv2d(x, kernel, strides, padding='SAME')

    def _relu(x, leakiness=0.0):
        """Relu, with optional leaky support."""
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _fully_connected(x, out_dim):
        """FullyConnected layer for final output."""
        num_non_batch_dimensions = len(x.shape)
        prod_non_batch_dimensions = 1
        for ii in range(num_non_batch_dimensions - 1):
            prod_non_batch_dimensions *= int(x.shape[ii + 1])
        x = tf.reshape(x, [tf.shape(x)[0], -1])
        w = tf.get_variable(
            'DW', [prod_non_batch_dimensions, out_dim],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer())

    def _global_avg_pool(x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])

    def _stride_arr(stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]


    with tf.variable_scope('init_conv'):
        z = tf.layers.conv2d(x, filters=64, kernel_size=[3, 3],
                            padding='same', use_bias=False)
        z = tf.layers.batch_normalization(z, training=training)
        z = tf.nn.relu(z)

    if params[0] == 'BasicBlock':
        block = _basic_block
        expansion = 1
    else:
        block =  _bottleneck
        expansion = 4
    num_blocks = params[1]

    in_planes = 64
    def _make_layer(z, block, in_planes, planes, num_blocks, stride, name=''):
        strides = [stride] + [1]*(num_blocks-1)
        for i, stride in enumerate(strides):
            z = block(z, in_planes, planes, stride, name=name+str(i))
            in_planes = planes * expansion
        return z, in_planes

    z, in_planes = _make_layer(z, block, in_planes, 64, num_blocks[0], stride=1, name='1_')
    z, in_planes = _make_layer(z, block, in_planes, 128, num_blocks[1], stride=2, name='2_')
    z, in_planes = _make_layer(z, block, in_planes, 256, num_blocks[2], stride=2, name='3_')
    z, in_planes = _make_layer(z, block, in_planes, 512, num_blocks[3], stride=2, name='4_')

    with tf.variable_scope('mlp'):
        z = _global_avg_pool(z)
        z = tf.layers.flatten(z)

    logits_ = tf.layers.dense(z, units=10, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')

    if logits:
        return y, logits_
    return y


def TResNet18(x, logits=False, training=False):
    return TResNet(x, logits, training, 'TResNet18')

def TResNet34(x, logits=False, training=False):
    return TResNet(x, logits, training, 'TResNet34')

def TResNet50(x, logits=False, training=False):
    return TResNet(x, logits, training, 'TResNet50')

def TResNet101(x, logits=False, training=False):
    return TResNet(x, logits, training, 'TResNet101')

def TResNet152(x, logits=False, training=False):
    return TResNet(x, logits, training, 'TResNet152')


if __name__ == "__main__":
    x = tf.ones(shape=(128, 32, 32, 3))
    y1, y2, y3, y4, y5 = TResNet18(x), TResNet34(x), TResNet50(x), TResNet101(x), TResNet152(x)
    print (y1, y2, y3, y4, y5)
