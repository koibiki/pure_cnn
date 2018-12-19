import os
import os.path as osp

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import *

from config import cfg


def load_data():
    mnist = input_data.read_data_sets('../data/', one_hot=True)
    return mnist


def write_tfrecords(save_dir, data, name):
    """
    :param image_dir:
    :param name: Name of the dataset (e.g. "train", "test", or "validation")
    :param save_dir: Where to store the tf records
    """
    tfrecord_path = osp.join(save_dir, '%s_feature.tfrecords' % name)
    print('Writing tf records for %s at %s...' % (name, tfrecord_path))

    with tf.python_io.TFRecordWriter(tfrecord_path) as writer:
        images = data.images
        labels = data.labels

        for i in tqdm(range(len(labels)), desc='write ' + name):
            image = images[i]
            image = np.reshape(image, (28, 28, 1))
            label = np.argmax(labels[i])

            image = image.tobytes()

            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                }))

            writer.write(example.SerializeToString())


if __name__ == '__main__':
    save_dir = osp.join(cfg.PATH.ROOT_DIR, '../tf_data')

    os.makedirs(save_dir, exist_ok=True)

    mnist = load_data()

    print('Initializing the dataset provider... ', end='', flush=True)
    print('done.')

    write_tfrecords(save_dir, mnist.train, 'train')
    write_tfrecords(save_dir, mnist.test_origin_net, 'test')
