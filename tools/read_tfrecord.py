from config import cfg
from dataset.dataset_reader import *

train_tf_path = '../tf_data/train_feature.tfrecords'
image, label = read_features(train_tf_path, (28, 28, 1))

images, labels = tf.train.shuffle_batch(
    tensors=[image, label], batch_size=cfg.TRAIN.BATCH_SIZE,
    capacity=100 + 2 * cfg.TRAIN.BATCH_SIZE, min_after_dequeue=100, num_threads=1)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(2):
        example, l = sess.run([images, labels])
        print('train:')
        print(example)
        print(l)

    coord.request_stop()
