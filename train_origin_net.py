from dataset.dataset_reader import *
from config import cfg
from net.origin_net import Net

train_tf_path = './tf_data/train_feature.tfrecords'
image, label = read_features(train_tf_path, (28, 28, 1))

images, labels = tf.train.shuffle_batch(
    tensors=[image, label], batch_size=cfg.TRAIN.BATCH_SIZE,
    capacity=100 + 2 * cfg.TRAIN.BATCH_SIZE, min_after_dequeue=100, num_threads=1)

net = Net()

net.train_origin_net(images, labels, weight_path='./checkpoints/net_2018-12-19-10-05-17.ckpt-99900')
