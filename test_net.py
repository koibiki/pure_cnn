from dataset.dataset_reader import *
from config import cfg
from net.origin_net import Net
import math

test_tf_path = './tf_data/test_feature.tfrecords'

image, label = read_features(test_tf_path, (28, 28, 1))

images, labels = tf.train.shuffle_batch(
    tensors=[image, label], batch_size=cfg.TRAIN.BATCH_SIZE,
    capacity=100 + 2 * cfg.TRAIN.BATCH_SIZE, min_after_dequeue=100, num_threads=1)

test_sample_count = sum(1 for _ in tf.python_io.tf_record_iterator(test_tf_path))
num_iterations = int(math.ceil(test_sample_count / cfg.TEST.BATCH_SIZE))

net = Net()

net.test_origin_net(images, labels, num_iterations,
                    weight_path='./checkpoints/net_2018-12-19-10-05-17.ckpt-99900')
