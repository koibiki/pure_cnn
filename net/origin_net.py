import os
import os.path as osp
import time

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from config import cfg


class Net(object):

    def __init__(self):
        self._batch_size = 8
        self.inputs = tf.placeholder(dtype=tf.float32, shape=(self._batch_size, 28, 28, 1), name='input')
        self.outputs = tf.placeholder(dtype=tf.int32, shape=(self._batch_size,), name='output')

    def build_origin_net(self, keep_prob):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l1_regularizer(0.001)):
            conv1 = slim.layers.conv2d(self.inputs, 16, 3, padding="SAME", scope='conv1')

            # 2 x 2 / 1
            pool1 = slim.layers.max_pool2d(conv1, kernel_size=[2, 2], stride=[2, 2], scope='pool1')

            # 128 / 3 x 3 / 1 / 1
            conv2 = slim.layers.conv2d(pool1, 32, 3, padding="SAME", scope='conv2')

            # 2 x 2 / 1
            pool2 = slim.layers.max_pool2d(conv2, kernel_size=[2, 2], stride=[2, 2], scope='pool2')

            # 256 / 3 x 3 / 1 / 1
            conv3 = slim.layers.conv2d(pool2, 32, 3, padding="SAME", scope='conv3')

            # 1 x 2 / 1
            pool3 = slim.layers.max_pool2d(conv3, kernel_size=[2, 2], stride=[2, 2], padding="SAME", scope='pool3')

            flatten = slim.flatten(pool3)

            fc1 = slim.layers.fully_connected(flatten, 128, scope='fc1')

            fc2 = slim.layers.fully_connected(fc1, 256, scope='fc2')

        dp2 = slim.layers.dropout(fc2, keep_prob=keep_prob)

        logits = slim.layers.fully_connected(dp2, 10, scope='logits')

        pred = slim.layers.softmax(logits=logits, scope='pred')

        out = {'conv1': conv1, 'pool1': pool1, 'conv2': conv2, 'pool2': pool2, 'conv3': conv3, 'pool3': pool3,
               'flatten': flatten, 'fc1': fc1, 'fc2': fc2,
               'logits': logits, 'pred': pred}
        return out

    def build_pured_net(self, keep_prob, channel_dict):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l1_regularizer(0.001)):
            conv1 = slim.layers.conv2d(self.inputs, channel_dict['conv1'], 3, padding="SAME", scope='conv1')

            # 2 x 2 / 1
            pool1 = slim.layers.max_pool2d(conv1, kernel_size=[2, 2], stride=[2, 2], scope='pool1')

            # 128 / 3 x 3 / 1 / 1
            conv2 = slim.layers.conv2d(pool1, channel_dict['conv2'], 3, padding="SAME", scope='conv2')

            # 2 x 2 / 1
            pool2 = slim.layers.max_pool2d(conv2, kernel_size=[2, 2], stride=[2, 2], scope='pool2')

            # 256 / 3 x 3 / 1 / 1
            conv3 = slim.layers.conv2d(pool2, channel_dict['conv3'], 3, padding="SAME", scope='conv3')

            # 1 x 2 / 1
            pool3 = slim.layers.max_pool2d(conv3, kernel_size=[2, 2], stride=[2, 2], padding="SAME", scope='pool3')

            flatten = slim.flatten(pool3)

            fc1 = slim.layers.fully_connected(flatten, channel_dict['fc1'], scope='fc1')

            fc2 = slim.layers.fully_connected(fc1, channel_dict['fc2'], scope='fc2')

        dp2 = slim.layers.dropout(fc2, keep_prob=keep_prob)

        logits = slim.layers.fully_connected(dp2, 10, scope='logits')

        pred = slim.layers.softmax(logits=logits, scope='pred')

        out = {'conv1': conv1, 'pool1': pool1, 'conv2': conv2, 'pool2': pool2, 'conv3': conv3, 'pool3': pool3,
               'flatten': flatten, 'fc1': fc1, 'fc2': fc2,
               'logits': logits, 'pred': pred}
        return out

    # 加载剪枝后的权重
    def restore_pure_weight(self, sess, pure_weights):
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        trainable_tensors = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for trainable_tensor in trainable_tensors:
            tensor_name = trainable_tensor.name.replace(':0', '')
            pure_weight = pure_weights[tensor_name]
            if pure_weight is None:
                raise Exception('{:s}的权重不存在'.format(tensor_name))
            else:
                trainable_tensor.load(pure_weight, sess)

    def build_loss(self, out):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.outputs, logits=out['logits'])
        regularization_loss = tf.add_n(slim.losses.get_regularization_losses())
        total_loss = loss + regularization_loss
        return total_loss, loss, regularization_loss

    def build_optimizer(self, cost):
        global_step = tf.Variable(0, name='global_step', trainable=False)

        starter_learning_rate = 0.001
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 10000, 0.9, staircase=True)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=cost,
                                                                                                global_step=global_step)
        return optimizer, learning_rate

    def build_summary(self, cost, learning_rate):
        os.makedirs(cfg.PATH.TBOARD_SAVE_DIR, exist_ok=True)

        tf.summary.scalar(name='Cost', tensor=cost)
        tf.summary.scalar(name='Learning_Rate', tensor=learning_rate)
        merge_summary_op = tf.summary.merge_all()
        return merge_summary_op

    def build_sess(self):
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = 0.7
        sess_config.gpu_options.allow_growth = True

        sess = tf.Session(config=sess_config)
        return sess

    def build_saver(self, prefix):
        saver = tf.train.Saver(max_to_keep=3)
        os.makedirs('./logs', exist_ok=True)
        train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        model_name = '{:s}_{:s}.ckpt'.format(prefix, str(train_start_time))
        model_save_path = osp.join(cfg.PATH.MODEL_SAVE_DIR, model_name)
        return saver, model_save_path

    '''
    rebirth 模型
    '''

    def train_pure_net(self, images, labels, channel_dict, pure_weights):
        out = self.build_pured_net(keep_prob=0.4, channel_dict=channel_dict)

        total_loss, loss, l1_loss = self.build_loss(out)

        optimizer, learning_rate = self.build_optimizer(total_loss)

        saver, model_save_path = self.build_saver('pure_net')

        sess = self.build_sess()

        train_epochs = cfg.TRAIN.EPOCHS
        with sess.as_default():
            self.restore_pure_weight(sess, pure_weights)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for epoch in range(train_epochs):
                input_data, input_label = sess.run([images, labels])

                _, total_c, c, l1_c, pred = sess.run(
                    [optimizer, total_loss, loss, l1_loss, out['pred']],
                    feed_dict={self.inputs: input_data,
                               self.outputs: input_label})

                if epoch % cfg.TRAIN.DISPLAY_STEP == 0:
                    print(
                        'Epoch: {:d} cost= {:9f} class_c={:9f} l1_c={:9f}'.format(epoch + 1, np.sum(total_c), np.sum(c),
                                                                                  np.sum(l1_c)))
                    p_argmax = [np.argmax(pred[i]) for i in range(len(pred))]
                    print(input_label)
                    print(p_argmax)
                    tf.train.write_graph(sess.graph_def, 'checkpoints', 'net_txt.pb', as_text=True)
                    saver.save(sess=sess, save_path=model_save_path, global_step=epoch)

        coord.request_stop()
        coord.join(threads=threads)

        sess.close()

    def train_origin_net(self, images, labels, weight_path=None):
        out = self.build_origin_net(0.4)

        total_loss, loss, l1_loss = self.build_loss(out)

        optimizer, learning_rate = self.build_optimizer(total_loss)

        saver, model_save_path = self.build_saver('origin_net')

        sess = self.build_sess()

        train_epochs = cfg.TRAIN.EPOCHS

        with sess.as_default():
            if weight_path is None:
                print('Training from scratch')
                sess.run(tf.local_variables_initializer())
                sess.run(tf.global_variables_initializer())
            else:
                print('Restore model from {:s}'.format(weight_path))
                saver.restore(sess=sess, save_path=weight_path)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for epoch in range(train_epochs):

                input_data, input_label = sess.run([images, labels])

                _, total_c, c, l1_c, pred = sess.run(
                    [optimizer, total_loss, loss, l1_loss, out['pred']],
                    feed_dict={self.inputs: input_data,
                               self.outputs: input_label})

                if epoch % cfg.TRAIN.DISPLAY_STEP == 0:
                    print(
                        'Epoch: {:d} cost= {:9f} class_c={:9f} l1_c={:9f}'.format(epoch + 1, np.sum(total_c), np.sum(c),
                                                                                  np.sum(l1_c)))
                    p_argmax = [np.argmax(pred[i]) for i in range(len(pred))]
                    print(input_label)
                    print(p_argmax)
                    tf.train.write_graph(sess.graph_def, 'checkpoints', 'net_txt.pb', as_text=True)
                    saver.save(sess=sess, save_path=model_save_path, global_step=epoch)

                    # simple_save(sess,
                    #             'save_model',
                    #             inputs={"input": self._input},
                    #             outputs={"predict_num": out['pred_num'],
                    #                      "predict_one": out['pred_one'],
                    #                      "predict_first": out['pred_first'],
                    #                      "predict_second": out['pred_second']})

        coord.request_stop()
        coord.join(threads=threads)

        sess.close()

    def test_origin_net(self, images, labels, num_iterations, weight_path):
        out = self.build_origin_net(1.0)

        saver, model_save_path = self.build_saver('origin_net')

        sess = self.build_sess()
        real_label = []
        pred_label = []
        with sess.as_default():
            saver.restore(sess=sess, save_path=weight_path)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for epoch in range(num_iterations):
                input_data, input_label = sess.run([images, labels])
                pred = sess.run([out['pred']],
                                feed_dict={self.inputs: input_data, self.outputs: input_label})
                pred = [np.argmax(pred[0][i]) for i in range(len(pred[0]))]

                real_label += list(input_label)

                pred_label += pred

                print('label = ' + str(real_label[-1]) + ' pred label= ' + str(pred_label[-1]) + ' iter {:0f}/{:0f}'.
                      format(epoch, num_iterations))

            print('all accuracy = {:9f} '.format(accuracy_score(real_label, pred_label)))
            print(classification_report(real_label, pred_label))

            coord.request_stop()
            coord.join(threads=threads)
        sess.close()

    def test_pure_net(self, images, labels, channel_dict, pure_weights, num_iterations, weight_path):
        out = self.build_pured_net(1.0, channel_dict=channel_dict)

        saver, model_save_path = self.build_saver('pure_net')

        sess = self.build_sess()
        real_label = []
        pred_label = []
        with sess.as_default():
            # saver.restore(sess=sess, save_path=weight_path)
            self.restore_pure_weight(sess, pure_weights)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for epoch in range(num_iterations):
                input_data, input_label = sess.run([images, labels])
                pred = sess.run([out['pred']],
                                feed_dict={self.inputs: input_data, self.outputs: input_label})
                pred = [np.argmax(pred[0][i]) for i in range(len(pred[0]))]

                real_label += list(input_label)

                pred_label += pred

                print('label = ' + str(real_label[-1]) + ' pred label= ' + str(pred_label[-1]) + ' iter {:0f}/{:0f}'.
                      format(epoch, num_iterations))

            print('all accuracy = {:9f} '.format(accuracy_score(real_label, pred_label)))
            print(classification_report(real_label, pred_label))

            coord.request_stop()
            coord.join(threads=threads)
        sess.close()
