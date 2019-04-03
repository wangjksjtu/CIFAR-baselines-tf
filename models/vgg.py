import tensorflow as tf

def VGG(x, logits=False, training=False, arch='VGG16'):
    cfg = {
        'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }
    params = cfg[arch]

    z = x
    for i, param in enumerate(params):
        if param == 'M':
            with tf.variable_scope('maxpool' + str(i)):
                z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)
        else:
            with tf.variable_scope('conv' + str(i)):
                z = tf.layers.conv2d(z, filters=param, kernel_size=[3, 3],
                                     padding='same')
                z = tf.layers.batch_normalization(z, training=training)
                z = tf.nn.relu(z)
    z = tf.layers.max_pooling2d(z, pool_size=[1, 1], strides=1)

    with tf.variable_scope('mlp'):
        z = tf.layers.flatten(z)

    # without FC4096-FC4096-FC1000

    logits_ = tf.layers.dense(z, units=10, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')

    if logits:
        return y, logits_
    return y

def VGG11(x, logits=False, training=False):
    return VGG(x, logits, training, 'VGG11')

def VGG13(x, logits=False, training=False):
    return VGG(x, logits, training, 'VGG13')

def VGG16(x, logits=False, training=False):
    return VGG(x, logits, training, 'VGG16')

def VGG19(x, logits=False, training=False):
    return VGG(x, logits, training, 'VGG19')


if __name__ == "__main__":
    x = tf.ones(shape=(128, 32, 32, 3))
    y1, y2, y3, y4 = VGG11(x), VGG13(x), VGG16(x), VGG19(x)
    print (y1, y2, y3, y4)
