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
        # word_embed_init = tf.reshape(tf.nn.embedding_lookup(self.word_embed_weight, self.input), shape=[self.length[-1], -1, cfg.word_embedding_size])
            word_embed_init = tf.nn.embedding_lookup(self.word_embed_weight, self.input)
            lstm_cell_fw = contrib.rnn.LSTMCell(cfg.hidden_size)
            lstm_cell_bw = contrib.rnn.LSTMCell(cfg.hidden_size)
            out, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_fw, cell_bw=lstm_cell_bw, inputs=word_embed_init,
                                                         sequence_length=self.length, dtype=tf.float32)
            bilstm_output = tf.concat([out[0][:,-1,:], out[1][:,-1,:]], 1)
            dense_layer = tf.squeeze(tf.layers.dense(inputs=bilstm_output, units=1, activation=tf.nn.relu))
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=dense_layer))
            optimizer = tf.train.AdamOptimizer()
            self.grads_and_vars = optimizer.compute_gradients(self.loss)
            self.opt = optimizer.apply_gradients(self.grads_and_vars)
            # self.opt = tf.train.AdamOptimizer().minimize(self.loss)

    def get_dense_layer(self, input, label, length):
        return self.sess.run([self.grads_and_vars], feed_dict={
            self.input: input,
            self.label: label,
            self.length: length})

    def train(self, input, label, length, max_size):
        return self.sess.run([self.opt, self.loss], feed_dict={
            self.input: input,
            self.label: label,
            self.length: length,
            self.max_size: max_size})


data_list = []
label_list = []
with open("train.csv", 'r') as f:
    for line in f:
        elements = line.strip('\r\n').split(',')
        data_list.append([int(item) for item in elements[:-1]])
        label_list.append(int(elements[-1]))
    f.close()

config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:
    modelObj = BiLSTM(sess)
    tf.global_variables_initializer().run()
    trainable = False
    filename = "train.csv"
    epoch_iter = 0

    while epoch_iter < cfg.epoch_size:
        epoch_iter += 1
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
                '''
                _, b_value = sess.run([key, value])
                s_value = str(b_value, encoding="utf-8")
                data = s_value.split(",")
                features = data[:-1]
                label = data[-1]
                '''
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
                max_len = 10
                samples = tf.keras.preprocessing.sequence.pad_sequences(batch_samples, maxlen=max_len, padding='post')
                length = np.zeros(shape=[len(batch_samples)], dtype=np.int32)
                length.fill(max_len)
                labels = np.asarray(batch_labels, dtype=np.float32)
                #print(modelObj.get_dense_layer(samples, labels, length)[0])
                _, loss = modelObj.train(samples, labels, length, max_len)
                grads_and_vars = modelObj.get_dense_layer(samples, labels, length)
                print(offset)
                for gv in grads_and_vars:
                    print(gv[0][1])
                    #print(gv[1][0].shape)
                    #print(gv[1][1].shape)
                #print(modelObj.get_dense_layer(samples, labels, length)[0])
                total_loss += loss
                iter += 1
            print("epoch is {0}, loss is {1}".format(epoch_iter, loss))
        except tf.errors.OutOfRangeError:
            print("There are examples")
    '''
    for epoch_index in range(cfg.epoch_size):
        loss_sum = 0.0
        for i in range(train_set_size):
            if trainable is True:
                tf.get_variable_scope().reuse_variables()
            trainable = True
            _, iter_loss = modelObj.train(word1_list[i * cfg.batch_size:(i+1) * cfg.batch_size],
                                             word2_list[i * cfg.batch_size:(i+1) * cfg.batch_size],
                                             position_list[i * cfg.batch_size:(i+1) * cfg.batch_size])
            loss_sum += iter_loss
        print("epoch_index %d, loss is %f" % (epoch_index, np.sum(loss_sum) / cfg.batch_size))
        train_loss = PosModelObj.get_loss_summary(np.sum(loss_sum) / cfg.batch_size)
        train_writer.add_summary(train_loss, epoch_index + 1)

        accuracy = 0.0
        for j in range(total_batch_size - train_set_size):
            j += train_set_size
            iter_accuracy, index_score = PosModelObj.validate(word1_list[j*cfg.batch_size : (j+1)*cfg.batch_size],
                                                 position_list[j*cfg.batch_size : (j+1) * cfg.batch_size],
                                                 labels[j*cfg.batch_size : (j+1)*cfg.batch_size],
                                                 targets[j*cfg.batch_size : (j+1)*cfg.batch_size])
            accuracy += iter_accuracy
        print("iter %d : accuracy %f" % (epoch_index, accuracy / (total_batch_size - train_set_size)))
        test_accuracy = PosModelObj.get_accuracy_summary(accuracy / (total_batch_size - train_set_size))
        test_writer.add_summary(test_accuracy, epoch_index + 1)

    embed_weight = PosModelObj.get_word_emb()
    output_embed_file = open(sys.argv[5], 'w')
    for embed_item in embed_weight:
        embed_list = list(embed_item)
        embed_list = [str(item) for item in embed_list]
        output_embed_file.write(','.join(embed_list) + '\n')
    output_embed_file.close()

    PosModelObj.save()
    sess.close()
    '''