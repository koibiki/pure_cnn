import os.path as osp
from typing import Tuple

import tensorflow as tf


def read_features(tfrecords_path: str, input_size: Tuple[int, int, int]):
    assert osp.exists(tfrecords_path), "tfrecords file not found: %s" % tfrecords_path

    filename_queue = tf.train.string_input_producer([tfrecords_path])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image': tf.FixedLenFeature((), tf.string),
                                           'label': tf.FixedLenFeature((), tf.int64),
                                       })
    # 注意存储和读取时的 dtype 需要一致
    image = tf.decode_raw(features['image'], tf.float32)
    h, w, c = input_size
    image = tf.reshape(image, [h, w, c])
    label = tf.cast(features['label'], tf.int32)
    return image, label
