# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib as contrib
import numpy as np
from config import cfg
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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
        self.input = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.label = tf.placeholder(dtype=tf.float32, shape=[None])
        self.length = tf.placeholder(dtype=tf.int32, shape=[None])
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
            self.dense_layer = tf.squeeze(tf.layers.dense(inputs=bilstm_output, units=1, activation=tf.nn.leaky_relu))
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.dense_layer))
            global_value = tf.identity(self.global_step)
            learning_rate = tf.train.exponential_decay(0.001,
                                                       global_value,
                                                       decay_steps=cfg.epoch_size,
                                                       decay_rate=0.03)
            optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
            grads = optimizer.compute_gradients(self.loss)
            for i, (g, v) in enumerate(grads):
                if g is not None:
                    grads[i] = (tf.clip_by_norm(g, 5), v)  # clip gradients
            self.opt = optimizer.apply_gradients(grads)
            '''
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=None)
            self.opt = tf.train.AdamOptimizer().minimize(self.loss)
            '''

            self.eval = tf.reduce_sum(
                tf.cast(tf.equal(tf.cast(self.label, dtype=tf.int32),
                                tf.cast(tf.argmax(tf.concat([tf.expand_dims(1.0 - self.dense_layer, -1), tf.expand_dims(self.dense_layer, -1)], 1), 1), dtype=tf.int32)), dtype=tf.int32))

    def get_dense_layer(self, input, label, length):
        return self.sess.run([self.tmp], feed_dict={
            self.input: input,
            self.label: label,
            self.length: length})

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

data_list = []
label_list = []
with open("train.csv", 'r') as f:
    for line in f:
        elements = line.strip('\r\n').split(',')
        data_list.append([int(item) for item in elements[:-1]])
        label_list.append(int(elements[-1]))
    f.close()
validation_list = []
validation_label_list = []
with open("val.csv", 'r') as f:
    for line in f:
        elements = line.strip('\r\n').split(',')
        validation_list.append([int(item) for item in elements[:-1]])
        validation_label_list.append(int(elements[-1]))
    f.close()

config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:
    modelObj = BiLSTM(sess)
    tf.global_variables_initializer().run()
    trainable = False
    filename = "train.csv"
    epoch_iter = 0

    global_max_len = -1
    while epoch_iter < cfg.epoch_size:
        '''
        filename_queue = tf.train.string_input_producer([filename], num_epochs=1)
        sess.run(tf.initialize_local_variables())
        reader = tf.TextLineReader(skip_header_lines=0)
        key, value = reader.read(filename_queue)
        tf.train.start_queue_runners()
        batch_iter = 0
        batch_samples = []
        batch_labels = []
        '''
        total_loss = 0.0
        try:
            iter = 0
            offset = 0
            while offset < len(data_list):
                prev_offset = offset
                if trainable is True:
                    tf.get_variable_scope().reuse_variables()
                trainable = True
                if (iter + 1) * cfg.batch_size > len(data_list):
                    offset = len(data_list)
                else:
                    offset = (iter + 1) * cfg.batch_size
                batch_samples = []
                batch_labels = []
                max_len = -1
                for i in range(prev_offset, offset, 1):
                    batch_samples.append(data_list[i])
                    batch_labels.append(label_list[i])
                    if len(data_list[i]) > max_len:
                        max_len = len(data_list[i])
                # max_len = 10
                if max_len > global_max_len:
                    global_max_len = max_len
                samples = tf.keras.preprocessing.sequence.pad_sequences(batch_samples, maxlen=max_len, padding='post')
                length = np.zeros(shape=[len(batch_samples)], dtype=np.int32)
                length.fill(max_len)
                labels = np.asarray(batch_labels, dtype=np.float32)
                _, loss = modelObj.train(samples, labels, length, max_len, epoch_iter)
                #print(modelObj.get_dense_layer(samples, labels, length))
                # global_val = modelObj.get_dense_layer(samples, labels, length)
                total_loss += loss
                iter += 1
            print("epoch is {0}, loss is {1}".format(epoch_iter, loss))
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
        samples = tf.keras.preprocessing.sequence.pad_sequences(validation_list, maxlen=max_len, padding='post')
        eval_val = modelObj.predict(samples, np.asarray(validation_label_list, dtype=np.int32), length)
        print("epoch is {0}, accu is {1}".format(epoch_iter, float(eval_val[0]) / float(length.shape[0])))
        epoch_iter += 1