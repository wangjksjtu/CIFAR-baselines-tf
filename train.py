import argparse
import math
import os
from importlib import import_module

import numpy as np
import tensorflow as tf
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="cifar", type=str, help='Dataset: mnist/cifar')
parser.add_argument("--epochs", default=300, type=int, help='Epochs for training models')
parser.add_argument("--model", default="VGG16", type=str, help='Model architecture')
parser.add_argument("--batch_size", default=128, type=int, help='Batch size training models')
parser.add_argument("--weight_decay", default=5e-4, type=float, help='Weight decay')
args = parser.parse_args()

DATASET = args.dataset
MODEL = args.model
EPOCHS = args.epochs
batch_size = args.batch_size
weight_decay = args.weight_decay

models = import_module('models')
assert DATASET in ["mnist", "cifar"]

if DATASET == "mnist":
    data = MNIST()
else:
    data = CIFAR10()

fmt = {"lr": '.1f', "train_loss": '.4f', "train_acc": '.4f', "val_loss": '.4f', "val_acc": '.4f'}
logger = Logger(fmt=fmt, name=MODEL, base="./logs/"+DATASET)


def evaluate(sess, env, X_data, y_data, batch_size=128):
    """
    Evaluate TF model by running env.loss and env.acc.
    """
    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    loss, acc = 0, 0

    for batch in range(n_batch):
        # print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start
        batch_loss, batch_acc = sess.run(
            [env.loss, env.acc],
            feed_dict={env.x: X_data[start:end],
                       env.y: y_data[start:end],
                       env.training: False})
        loss += batch_loss * cnt
        acc += batch_acc * cnt
    loss /= n_sample
    acc /= n_sample

    # print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
    return loss, acc


def train(sess, env, X_data, y_data, X_val=None, y_val=None, epochs=1,
          load=False, shuffle=True, batch_size=128, name='model', dataset='mnist'):
    """
    Train a TF model by running env.train_op.
    """
    if load:
        if not hasattr(env, 'saver'):
            return print('\nError: cannot find saver op')
        print('\nLoading saved model')
        return env.saver.restore(sess, 'models/' + dataset + '/{}'.format(name))

    print('\nTrain model')
    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)

    if hasattr(env, 'saver'):
        os.makedirs('models', exist_ok=True)

    max_acc = 0
    for epoch in range(epochs):
        # print('\nEpoch {0}/{1}'.format(epoch + 1, epochs))
        if shuffle:
            ind = np.arange(n_sample)
            np.random.shuffle(ind)
            X_data = X_data[ind]
            y_data = y_data[ind]

        for batch in range(n_batch):
            # print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
            start = batch * batch_size
            end = min(n_sample, start + batch_size)
            sess.run(env.train_op, feed_dict={env.x: X_data[start:end],
                                              env.y: y_data[start:end],
                                              env.training: True})
            lr = sess.run(env.lr)

        if X_val is not None:
            train_loss, train_acc = evaluate(sess, env, X_data, y_data)
            val_loss, val_acc = evaluate(sess, env, X_val, y_val)
            logger.add_scalar(epoch, "lr", lr)
            logger.add_scalar(epoch, "train_loss", train_loss)
            logger.add_scalar(epoch, "train_acc", train_acc)
            logger.add_scalar(epoch, "val_loss", val_loss)
            logger.add_scalar(epoch, "val_acc", val_acc)
            logger.iter_info()

        if max_acc < val_acc:
            print('\nSaving model {:f} -> {:f}'.format(max_acc, train_acc))
            max_acc = val_acc
            env.saver.save(sess, 'saves/' + dataset + '/{}'.format(name))


def predict(sess, env, X_data, batch_size=128):
    """
    Do inference by running env.ybar.
    """
    print('\nPredicting')
    n_classes = env.ybar.get_shape().as_list()[1]

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    yval = np.empty((n_sample, n_classes))

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        y_batch = sess.run(env.ybar, feed_dict={env.x: X_data[start:end]})
        yval[start:end] = y_batch
    print()
    return yval


class Dummy:
    pass

env = Dummy()


def main():
    x_train, y_train, x_test, y_test = data.x_train, data.y_train, \
                                       data.x_test, data.y_test

    model = getattr(models, MODEL)

    if DATASET == "mnist":
        env.x = tf.placeholder(tf.float32, (None, 28, 28, 1),
                                name='x')
    else:
        env.x = tf.placeholder(tf.float32, (None, 32, 32, 3),
                                name='x')
    env.y = tf.placeholder(tf.float32, (None, 10), name='y')
    env.training = tf.placeholder_with_default(False, (), name='mode')

    env.global_step = tf.Variable(0, trainable=False)
    n_batch = math.ceil(x_train.shape[0] / float(batch_size))
    boundaries = [150 * n_batch, 250 * n_batch]
    values = [0.1, 0.01, 0.001]
    env.lr = tf.train.piecewise_constant(env.global_step, boundaries, values)

    with tf.variable_scope('model'+MODEL):
        env.ybar, logits = model(env.x, logits=True, training=env.training)

        with tf.variable_scope('acc'):
            count = tf.equal(tf.argmax(env.y, axis=1), tf.argmax(env.ybar, axis=1))
            env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')

        with tf.variable_scope('loss'):
            xent = tf.nn.softmax_cross_entropy_with_logits(labels=env.y,
                                                        logits=logits)
            env.loss = tf.reduce_mean(xent, name='loss')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.variable_scope('train_op'):
            # optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            optimizer = tf.train.MomentumOptimizer(learning_rate=env.lr, momentum=0.9, use_nesterov=True)
            costs = []
            for var in tf.trainable_variables():
                costs.append(tf.nn.l2_loss(var))

        with tf.control_dependencies(update_ops):
            # env.train_op = optimizer.minimize(env.loss)
            env.train_op = optimizer.minimize(env.loss + weight_decay * tf.add_n(costs), global_step=env.global_step)

        env.saver = tf.train.Saver()

    # Initializing graph
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Training
    train(sess, env, x_train, y_train, x_test, y_test, load=False, epochs=EPOCHS,
          name=MODEL, dataset=DATASET, batch_size=batch_size)


if __name__ == "__main__":
    main()
