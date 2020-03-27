# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib as contrib
from config import cfg

word_dict = {}
with open('dict.txt', 'r') as f:
    for line in f:
        elements = line.split('\r\n').split('\t')
        word_dict[elements[0]] = int(elements[1])
    f.close()

class BiLSTM(object):
    def __init__(self, sess, flag='concat'):
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, cfg.max_len_size])
        self.label = tf.placeholder(dtype=tf.int32, shape=[None])
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
        word_embed_init = tf.reshape(tf.nn.embedding_lookup(self.word_embed_weight, self.input), shape=[cfg.max_len_size, -1, cfg.word_embedding_size])
        lstm_cell_fw = contrib.rnn.LSTMCell(cfg.hidden_size)
        lstm_cell_bw = contrib.rnn.LSTMCell(self.hidden_size)
        out, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_fw, cell_bw=lstm_cell_bw, inputs=word_embed_init,
                                                     sequence_lenth=self.max_len_size, dtype=tf.float32)
        bilstm_output = tf.concat([out[0][-1,:,:], out[1][-1,:,:]], 1)
        dense_layer = tf.squeeze(tf.layers.dense(inputs=bilstm_output, units=1, activation=tf.nn.relu))
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=dense_layer))
        self.opt = tf.train.AdamOptimizer().minimize(self.loss)

    def train(self, input, label):
        return self.sess.run([self.opt, self.loss], feed_dict={
            self.input: input,
            self.label: label})

config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:
    modelObj = BiLSTM(sess)
    tf.global_variables_initializer().run()

    trainable = False

    reader = tf.TFRecordReader()
    filename = "train.tfrecords"
    filename_queue = tf.train.string_input_producer([filename], num_epochs=1)
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'feature': tf.FixedLenFeature([], tf.string),
                                       })
    feature = tf.decode_raw(features['feature'], tf.int64)
    label = tf.cast(features['label'], tf.int64)
    feature_batch, label_batch = tf.train.shuffle_batch([feature, label], batch_size=100,
                                                        capacity=200, min_after_dequeue=100, num_threads=2)

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