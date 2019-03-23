import argparse
import os
import random
import sys
import time
from collections import OrderedDict

import numpy as np
import scipy.misc
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from pandas import DataFrame
from tabulate import tabulate


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class MNIST():
    def __init__(self, one_hot=True, shuffle=False):
        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data(one_hot)
        self.num_train = self.x_train.shape[0]
        self.num_test = self.x_test.shape[0]
        if shuffle: self.shuffle_data()

    def load_data(self, one_hot):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # x_train.shape = (60000, 28, 28), range = [0, 255]
        # y_train.shape = (60000)
        
        x_train = np.reshape(x_train, [-1, 28, 28, 1])
        x_train = x_train.astype(np.float32) / 255
        x_test = np.reshape(x_test, [-1, 28, 28, 1])
        x_test = x_test.astype(np.float32) / 255
        
        if one_hot:
            # convert to one-hot labels
            y_train = tf.keras.utils.to_categorical(y_train)
            y_test = tf.keras.utils.to_categorical(y_test)

        return x_train, y_train, x_test, y_test

    def shuffle_data(self):
        ind = np.random.permutation(self.num_train)
        self.x_train, self.y_train = self.x_train[ind], self.y_train[ind]


class CIFAR10():
    def __init__(self, one_hot=True, shuffle=False, augument=False):
        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data(one_hot)
        self.num_train = self.x_train.shape[0]
        self.num_test = self.x_test.shape[0]
        
        # self.x_train = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), self.x_train)
        # self.x_test = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), self.x_test)

        if augument: self.x_train, self.y_train = self.augument_data()
        if shuffle: self.shuffle_data()

    def load_data(self, one_hot):
        cifar = tf.keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = cifar.load_data()
        # x_train.shape = (50000, 32, 32, 3), range = [0, 255]
        # y_train.shape = (50000, 1)
        
        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)
        x_train = x_train.astype(np.float32) / 255
        x_test = x_test.astype(np.float32) / 255

        if one_hot:
            # convert to one-hot labels
            y_train = tf.keras.utils.to_categorical(y_train)
            y_test = tf.keras.utils.to_categorical(y_test)

        return x_train, y_train, x_test, y_test

    def shuffle_data(self):
        ind = np.random.permutation(self.num_train)
        self.x_train, self.y_train = self.x_train[ind], self.y_train[ind]
    
    def augument_data(self):
        image_generator = ImageDataGenerator(
            rotation_range=90,
            # zoom_range = 0.05, 
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            # vertical_flip=True,
            )

        image_generator.fit(self.x_train)
        # get transformed images
        x_train, y_train = image_generator.flow(self.x_train, self.y_train,
                                                batch_size=self.num_train, 
                                                shuffle=False).next()

        return x_train, y_train


class Logger:
    def __init__(self, name='model', fmt=None, base="./logs"):
        self.handler = True
        self.scalar_metrics = OrderedDict()
        self.fmt = fmt if fmt else dict()

        # base = './logs'
        print (base)
        if not os.path.exists(base): os.makedirs(base)

        self.path = os.path.join(base, name + "_" + str(time.time())) 

        self.logs = self.path + '.csv'
        self.output = self.path + '.out'

        def prin(*args):
            str_to_write = ' '.join(map(str, args))
            with open(self.output, 'a') as f:
                f.write(str_to_write + '\n')
                f.flush()

            print(str_to_write)
            sys.stdout.flush()

        self.print = prin

    def add_scalar(self, t, key, value):
        if key not in self.scalar_metrics:
            self.scalar_metrics[key] = []
        self.scalar_metrics[key] += [(t, value)]

    def iter_info(self, order=None):
        names = list(self.scalar_metrics.keys())
        if order:
            names = order
        values = [self.scalar_metrics[name][-1][1] for name in names]
        t = int(np.max([self.scalar_metrics[name][-1][0] for name in names]))
        fmt = ['%s'] + [self.fmt[name] if name in self.fmt else '.1f' for name in names]

        if self.handler:
            self.handler = False
            self.print(tabulate([[t] + values], ['epoch'] + names, floatfmt=fmt))
        else:
            self.print(tabulate([[t] + values], ['epoch'] + names, tablefmt='plain', floatfmt=fmt).split('\n')[1])

    def save(self):
        result = None
        for key in self.scalar_metrics.keys():
            if result is None:
                result = DataFrame(self.scalar_metrics[key], columns=['t', key]).set_index('t')
            else:
                df = DataFrame(self.scalar_metrics[key], columns=['t', key]).set_index('t')
                result = result.join(df, how='outer')
        result.to_csv(self.logs)

        self.print('The log/output have been saved to: ' + self.path + ' + .csv/.out')


def test_mnist():
    print ("Testing MNIST dataloader...")
    data = MNIST()
    print (data.x_train.shape, data.y_train.shape, data.x_test.shape, data.y_test.shape)
    data = MNIST(one_hot=False)
    print (data.x_train.shape, data.y_train.shape, data.x_test.shape, data.y_test.shape)
    print (data.y_train[0:10])
    data = MNIST(shuffle=True, one_hot=False)
    print (data.x_train.shape, data.y_train.shape, data.x_test.shape, data.y_test.shape)
    print (data.y_train[0:10])


def test_cifar10():
    print ("Testing CIFAR10 dataloader...")
    data = CIFAR10()
    print (data.x_train.shape, data.y_train.shape, data.x_test.shape, data.y_test.shape)
    data = CIFAR10(one_hot=False)
    print (data.x_train.shape, data.y_train.shape, data.x_test.shape, data.y_test.shape)
    print (data.y_train[0:10])
    data = CIFAR10(shuffle=True, one_hot=False)
    print (data.x_train.shape, data.y_train.shape, data.x_test.shape, data.y_test.shape)
    print (data.y_train[0:10])


if __name__ == "__main__":
    test_mnist()
    test_cifar10()
