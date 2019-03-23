import tensorflow as tf

def ResNet(x, logits=False, training=False, arch='ResNet18'):
    cfg = {
        'ResNet18':  ['BasicBlock', [2, 2, 2, 2]],
        'ResNet34':  ['BasicBlock', [3, 4, 6, 3]],
        'ResNet50':  ['BasicBlock', [3, 4, 6 ,3]],
        'ResNet101': ['BottleNeck', [3, 4, 23 ,3]],
        'ResNet152': ['BottleNeck', [3, 8, 36 ,3]],
    }
    params = cfg[arch]

    def _basic_block(x, in_planes, planes, stride=1, name=''):
        expansion = 1
        with tf.variable_scope('block' + name):
            z = tf.layers.conv2d(x, filters=planes, kernel_size=[3, 3],
                                 padding='same', use_bias=False,
                                 strides=(stride, stride))
            z = tf.layers.batch_normalization(z, training=training)
            z = tf.nn.relu(z)

            z = tf.layers.conv2d(z, filters=planes, kernel_size=[3, 3],
                                 padding='same', use_bias=False)
            z = tf.layers.batch_normalization(z, training=training)

            shortcut = tf.identity(x)
            if stride != 1 or in_planes != planes:
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
            z = tf.layers.batch_normalization(z, training=training)
            z = tf.nn.relu(z)

            z = tf.layers.conv2d(z, filters=planes, kernel_size=[3, 3],
                                 padding='same', use_bias=False,
                                 stride=(stride, stride))
            z = tf.layers.batch_normalization(z, training=training)
            z = tf.nn.relu(z)

            z = tf.layers.conv2d(z, filters=expansion*planes, kernel_size=[1, 1],
                                 padding='valid', use_bias=False)
            z = tf.layers.batch_normalization(z, training=training)

            shortcut = tf.identity(x)
            if stride != 1 or in_planes != planes:
                shortcut = tf.layers.conv2d(shortcut, filters=expansion*planes,
                                            kernel_size=[1, 1],
                                            padding='same', use_bias=False,
                                            strides=(stride, stride))
                shortcut = tf.layers.batch_normalization(shortcut, training=training)

            z += shortcut
            z = tf.nn.relu(z)

    def _global_avg_pool(x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])

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

def ResNet18(x, logits=False, training=False):
    return ResNet(x, logits, training, 'ResNet18')

def ResNet34(x, logits=False, training=False):
    return ResNet(x, logits, training, 'ResNet34')

def ResNet50(x, logits=False, training=False):
    return ResNet(x, logits, training, 'ResNet50')

def ResNet101(x, logits=False, training=False):
    return ResNet(x, logits, training, 'ResNet101')

def ResNet152(x, logits=False, training=False):
    return ResNet(x, logits, training, 'ResNet152')


if __name__ == "__main__":
    x = tf.ones(shape=(128, 32, 32, 3))
    y1, y2, y3, y4, y5 = ResNet18(x), ResNet34(x), ResNet50(x), ResNet101(x), ResNet152(x)
    print (y1, y2, y3, y4, y5)