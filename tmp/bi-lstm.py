# -*- coding: utf-8 -*-

import argparse
import tensorflow as tf
import tensorflow.contrib as contrib
import datetime
import numpy as np
from config import cfg
import data_fetch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

batch_size = 256
word_dict = {}
max_len = -1
with open('dict.txt', 'r') as f:
    for line in f:
        elements = line.strip('\r\n').split('\t')
        word_dict[elements[0]] = int(elements[1])
    f.close()

class BiLSTM(object):
    def __init__(self, sess, flag='concat'):
        self.input = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input")
        self.label = tf.placeholder(dtype=tf.float32, shape=[None])
        self.length = tf.placeholder(dtype=tf.int32, shape=[None], name="length")
        self.max_size = tf.placeholder(dtype=tf.int32, shape=[])
        self.global_step = tf.placeholder('int64', None, name='learning_rate_step')
        self.sess = sess
        self.flag = flag
        with tf.device('/gpu:0'):
            with tf.variable_scope("embedding"):
                self.word_embed_weight = tf.get_variable(
                    'word_emb',
                    shape=(len(word_dict), cfg.word_embedding_size),
                    initializer=tf.random_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )
            word_embed_init = tf.nn.embedding_lookup(self.word_embed_weight, self.input)
            lstm_cell_fw = contrib.rnn.LSTMCell(cfg.hidden_size)
            lstm_cell_bw = contrib.rnn.LSTMCell(cfg.hidden_size)
            out, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_fw, cell_bw=lstm_cell_bw, inputs=word_embed_init,
                                                         sequence_length=self.length, dtype=tf.float32)
            bilstm_output = tf.concat([out[0][:,-1,:], out[1][:,-1,:]], 1)
            self.dense_layer = tf.squeeze(tf.layers.dense(inputs=bilstm_output, units=1, activation=tf.nn.leaky_relu), name="logits")
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.dense_layer))
            global_value = tf.identity(self.global_step)
            self.learning_rate = tf.train.exponential_decay(1.0,
                                                       global_value,
                                                       decay_steps=cfg.epoch_size,
                                                       decay_rate=0.03,
                                                       staircase=True)
            # self.learning_rate = tf.train.polynomial_decay(1.0, global_value, decay_steps=cfg.epoch_size)

            # truncated adam learning method
            optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5)
            grads = optimizer.compute_gradients(self.loss)
            for i, (g, v) in enumerate(grads):
                if g is not None:
                    grads[i] = (tf.clip_by_norm(g, 5), v)  # clip gradients
            self.opt = optimizer.apply_gradients(grads)

            # self.opt = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss, global_step=None)

            # dynamic learning rate adam
            # self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=None)
            '''
            self.opt = tf.train.AdamOptimizer().minimize(self.loss)
            '''

            self.eval = tf.reduce_sum(
                tf.cast(tf.equal(tf.cast(self.label, dtype=tf.int32),
                                tf.cast(tf.argmax(
                                    tf.concat([tf.expand_dims(1.0 - self.dense_layer, -1), tf.expand_dims(self.dense_layer, -1)], 1), 1), dtype=tf.int32)), dtype=tf.int32))

            self.epoch_loss = tf.placeholder(tf.float32)
            self.epoch_loss_summary = tf.summary.scalar('epoch_loss', self.epoch_loss)
            self.epoch_accu = tf.placeholder(tf.float32)
            self.epoch_accu_summary = tf.summary.scalar('epoch_accu', self.epoch_accu)

    def get_dense_layer(self, input, label, length, global_step):
        return self.sess.run([self.learning_rate], feed_dict={
            self.input: input,
            self.label: label,
            self.length: length,
            self.global_step: global_step})

    def train(self, input, label, length, max_size, global_step):
        return self.sess.run([self.opt, self.loss], feed_dict={
            self.input: input,
            self.label: label,
            self.length: length,
            self.max_size: max_size,
            self.global_step: global_step})

    def predict(self, input, label, length):
        return self.sess.run([self.eval], feed_dict={
            self.input: input,
            self.label: label,
            self.length: length})

    def get_loss_summary(self, epoch_loss):
        return self.sess.run(self.epoch_loss_summary, feed_dict={self.epoch_loss: epoch_loss})

    def get_accu_summary(self, epoch_accu):
        return self.sess.run(self.epoch_accu_summary, feed_dict={self.epoch_accu : epoch_accu})

# tf.train.shuffle_batch([train_samples, train_labels], batch_size=cfg.batch_size, num_threads=3, capacity=capacity, min_after_dequeue=min_after_dequeue)

parser = argparse.ArgumentParser(description='sentiment training')
parser.add_argument('path', type=str, default='', help='model path')
args = parser.parse_args()
print("path is {0}".format(args.path))

#slice_features, slice_lengths, slice_labels = data_fetch.slice_input_producer_data_fetch("./train.csv")

train_size = 167412
config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:
    train_writer = tf.summary.FileWriter(cfg.summaries_dir + cfg.train_summary_writer_path, sess.graph)
    modelObj = BiLSTM(sess)
    tf.global_variables_initializer().run()
    sess.run(tf.local_variables_initializer())
    trainable = False
    epoch_iter = 0

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    global_max_len = -1
    best_accu = -1.0
    train_sample_size = 0
    total_loss = 0.0

    validation_list, validation_label_list = data_fetch.extract_val_ds('./val.csv')
    reader_iterator = data_fetch.streaming_data_fetch("./train.tfrecord")
    try:
        starttime = datetime.datetime.now()
        while not coord.should_stop():
            train_feature, train_len, train_label = sess.run(reader_iterator)
            #train_feature, train_len, train_label = sess.run([slice_features, slice_lengths, slice_labels])
            train_sample_size += train_feature.shape[0]
            samples, length, labels = data_fetch.pad_train_set(train_feature, train_len, train_label)
            if length[0] > global_max_len:
                global_max_len = length[0]
            _, loss = modelObj.train(samples, labels, length, max_len, epoch_iter)
            # _, loss = modelObj.train(train_feature, train_label, train_len, max_len, epoch_iter)
            total_loss += loss
            #lr = modelObj.get_dense_layer(samples, labels, length, epoch_iter)

            if train_sample_size >= train_size:
                total_loss /= float(train_sample_size)
                loss_summary = modelObj.get_loss_summary(total_loss)
                train_writer.add_summary(loss_summary, epoch_iter)
                print("epoch is {0}, train sample size is {1}, loss is {2}".format(
                    epoch_iter, train_sample_size, total_loss))
                train_sample_size = 0
                total_loss = 0.0

                test_reader_iterator = data_fetch.streaming_data_fetch("./test.tfrecord")
                test_feature, test_len, test_label = sess.run(test_reader_iterator)
                test_size = test_feature.shape[0]
                epoch_accu = 0.0
                for i in range(test_size):
                    test_feature_item = np.reshape(test_feature[i], (1, test_feature[i].shape[0]))
                    print(test_feature_item.shape)
                    test_label_item = np.atleast_1d(test_label[i])
                    print(test_label_item.shape)
                    test_len_item = np.atleast_1d(test_len[i])
                    print(test_len_item.shape)
                    eval_val = modelObj.predict(test_feature_item, test_label_item, test_len_item)
                    epoch_accu += float(eval_val[0])
                '''
                samples, validation_length = data_fetch.extract_val_samples(validation_list, global_max_len)
                validation_label = np.asarray(validation_label_list[:data_fetch.validation_truc_size], dtype=np.int32)
                print("in validation size 0 is {0}, 1 is {1}".format(np.sum(validation_label == 0), np.sum(validation_label == 1)))
                eval_val = modelObj.predict(samples, validation_label, validation_length)
                '''
                epoch_accu = float(eval_val[0]) / float(test_size)
                accu_summary = modelObj.get_accu_summary(epoch_accu)
                train_writer.add_summary(accu_summary, epoch_iter)
                if epoch_accu > best_accu:
                    best_accu = epoch_accu
                    best_accu_idx = epoch_iter
                print("epoch is {0}, accu is {1}".format(epoch_iter, epoch_accu))
                sess_input = sess.graph.get_operation_by_name("input").outputs[0]
                sess_length = sess.graph.get_operation_by_name("length").outputs[0]
                sess_logits = sess.graph.get_operation_by_name("logits").outputs[0]
                tf.saved_model.simple_save(sess,
                                           './model' + args.path + '/' + str(epoch_iter) + "/",
                                           inputs={"input": sess_input,
                                                   "length": sess_length},
                                           outputs={"logits": sess_logits})
                epoch_iter += 1
    except tf.errors.OutOfRangeError:
        print("done! now lets kill all the threads……")
    finally:
        coord.request_stop()
        print('all threads are asked to stop!')
    coord.join(threads)

    total_loss /= float(train_sample_size)
    loss_summary = modelObj.get_loss_summary(total_loss)
    train_writer.add_summary(loss_summary, epoch_iter)
    print("epoch is {0}, train sample size is {1}, loss is {2}".format(
        epoch_iter, train_sample_size, total_loss))
    test_reader_iterator = data_fetch.streaming_data_fetch("./test.tfrecord")
    test_feature, test_len, test_label = sess.run(test_reader_iterator)
    test_size = test_feature.shape[0]
    epoch_accu = 0.0
    for i in range(test_size):
        eval_val = modelObj.predict(test_feature[i], test_label[i], test_len[i])
        epoch_accu += float(eval_val[0])
    epoch_accu = float(eval_val[0]) / float(test_size)
    '''
    samples, validation_length = data_fetch.extract_val_samples(validation_list, global_max_len)
    eval_val = modelObj.predict(samples, np.asarray(validation_label_list[:data_fetch.validation_truc_size], dtype=np.int32),
                                validation_length)
    epoch_accu = float(eval_val[0]) / float(samples.shape[0])
    '''
    accu_summary = modelObj.get_accu_summary(epoch_accu)
    train_writer.add_summary(accu_summary, epoch_iter)
    if epoch_accu > best_accu:
        best_accu = epoch_accu
        best_accu_idx = epoch_iter
    print("epoch is {0}, accu is {1}".format(epoch_iter, epoch_accu))
    sess_input = sess.graph.get_operation_by_name("input").outputs[0]
    sess_length = sess.graph.get_operation_by_name("length").outputs[0]
    sess_logits = sess.graph.get_operation_by_name("logits").outputs[0]
    tf.saved_model.simple_save(sess,
                               './model' + args.path + '/' + str(epoch_iter) + "/",
                               inputs={"input": sess_input,
                                       "length": sess_length},
                               outputs={"logits": sess_logits})
    endtime = datetime.datetime.now()
    print("total time cost is {0}".format((endtime - starttime).seconds))

    '''
    total_loss /= float(train_sample_size)
    print("epoch is {0}, loss is {1}".format(epoch_iter, total_loss))
    loss_summary = modelObj.get_loss_summary(total_loss)
    train_writer.add_summary(loss_summary, epoch_iter)

    val_max_len = -1
    for item in validation_list:
        if len(item) > val_max_len:
            val_max_len = len(item)
    if val_max_len > global_max_len:
        val_max_len = global_max_len
    length = np.zeros(shape=[len(validation_list)], dtype=np.int32)
    length.fill(val_max_len)
    samples = tf.keras.preprocessing.sequence.pad_sequences(validation_list, maxlen=max_len, padding='post')
    eval_val = modelObj.predict(samples, np.asarray(validation_label_list, dtype=np.int32), length)
    epoch_accu = float(eval_val[0]) / float(length.shape[0])
    accu_summary = modelObj.get_accu_summary(epoch_accu)
    train_writer.add_summary(accu_summary, epoch_iter)
    if epoch_accu > best_accu:
        best_accu = epoch_accu
        best_accu_idx = epoch_iter
    print("epoch is {0}, accu is {1}".format(epoch_iter, epoch_accu))
    epoch_iter += 1
    sess_input = sess.graph.get_operation_by_name("input").outputs[0]
    sess_length = sess.graph.get_operation_by_name("length").outputs[0]
    sess_logits = sess.graph.get_operation_by_name("logits").outputs[0]
    tf.saved_model.simple_save(sess,
                               './model/' + str(epoch_iter) + "/",
                               inputs={"input": sess_input,
                                       "length": sess_length},
                               outputs={"logits": sess_logits})
    '''

    '''
    while epoch_iter < cfg.epoch_size:
        total_loss = 0.0
        try:
            train_sample_size = 0
            train_samples, train_samples_len, train_labels = tfrecord_data_fetch()
            while True:
                if trainable is True:
                    tf.get_variable_scope().reuse_variables()
                trainable = True
                batch_samples = []
                batch_labels = []
                max_len = -1
                iter = 0
                while iter < cfg.batch_size:
                    sess_feature, sess_len, sess_label = sess.run([train_samples, train_samples_len, train_labels])
                    batch_samples.append(sess_feature)
                    batch_labels.append(sess_label[0])
                    if sess_len[0] > max_len:
                        max_len = sess_len[0]
                    iter += 1
                    train_sample_size += 1
                # max_len = 10
                if max_len > global_max_len:
                    global_max_len = max_len
                samples = tf.keras.preprocessing.sequence.pad_sequences(batch_samples, maxlen=max_len, padding='post')
                length = np.zeros(shape=[len(batch_samples)], dtype=np.int32)
                length.fill(max_len)
                labels = np.asarray(batch_labels, dtype=np.float32)
                _, loss = modelObj.train(samples, labels, length, max_len, epoch_iter)
                total_loss += loss
                iter += 1
            total_loss /= float(train_sample_size)
            print("epoch is {0}, loss is {1}".format(epoch_iter, total_loss))
            loss_summary = modelObj.get_loss_summary(total_loss)
            train_writer.add_summary(loss_summary, epoch_iter)
        except tf.errors.OutOfRangeError:
            print("There are examples")
        val_max_len = -1
        for item in validation_list:
            if len(item) > val_max_len:
                val_max_len = len(item)
        if val_max_len > global_max_len:
            val_max_len = global_max_len
        length = np.zeros(shape=[len(validation_list)], dtype=np.int32)
        length.fill(val_max_len)
        samples = tf.keras.preprocessing.sequence.pad_sequences(validation_list, maxlen=val_max_len, padding='post')
        eval_val = modelObj.predict(samples, np.asarray(validation_label_list, dtype=np.int32), length)
        epoch_accu = float(eval_val[0]) / float(length.shape[0])
        accu_summary = modelObj.get_accu_summary(epoch_accu)
        train_writer.add_summary(accu_summary, epoch_iter)
        if epoch_accu > best_accu:
            best_accu = epoch_accu
            best_accu_idx = epoch_iter
        print("epoch is {0}, accu is {1}".format(epoch_iter, epoch_accu))
        epoch_iter += 1
        sess_input = sess.graph.get_operation_by_name("input").outputs[0]
        sess_length = sess.graph.get_operation_by_name("length").outputs[0]
        sess_logits = sess.graph.get_operation_by_name("logits").outputs[0]
        tf.saved_model.simple_save(sess,
                                   './model/' + str(epoch_iter) + "/",
                                   inputs={"input": sess_input,
                                           "length": sess_length},
                                   outputs={"logits": sess_logits})
    print("best epoch model is {0}".format(best_accu_idx))
    train_writer.close()
    '''