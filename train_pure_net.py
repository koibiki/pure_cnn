from dataset.dataset_reader import *
from config import cfg
from net.origin_net import Net
import pickle

train_tf_path = './tf_data/train_feature.tfrecords'

pure_weights_path = './pure_checkpoints/pure_1.pb'

image, label = read_features(train_tf_path, (28, 28, 1))

images, labels = tf.train.shuffle_batch(
    tensors=[image, label], batch_size=cfg.TRAIN.BATCH_SIZE,
    capacity=100 + 2 * cfg.TRAIN.BATCH_SIZE, min_after_dequeue=100, num_threads=1)

with open(pure_weights_path, 'rb') as f:
    load = pickle.load(f)

channel_dict = load['channel_dict']
pure_weights = load['pure_weights']
print('channel dict:'.format(str(channel_dict)))

net = Net()

net.train_pure_net(images, labels, channel_dict, pure_weights)
