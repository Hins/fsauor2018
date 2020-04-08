# -*- coding: utf-8 -*-

from config import cfg
import random
import tensorflow as tf

val_file = open('val.csv', 'w')
train_writer = tf.python_io.TFRecordWriter('train.tfrecord')

with open('train.csv.all', 'r') as f:
    for line in f:
        if random.random() > cfg.train_set_ratio:
            val_file.write(line)
        else:
            feature_bin = {}
            elements = line.strip('\r\n').split(',')
            features = [int(item) for item in elements[:-1]]
            prev_length = len(features)
            label = int(elements[-1])
            if len(features) < cfg.train_max_len_size:
                features += [0] * (cfg.train_max_len_size - len(features))
            else:
                features = features[:cfg.train_max_len_size]
            feature_bin["feature"] = tf.train.Feature(int64_list=tf.train.Int64List(value=features))
            feature_bin["feature_len"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[prev_length]))
            feature_bin["label"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            train_writer.write(tf.train.Example(features=tf.train.Features(feature=feature_bin)).SerializeToString())
    f.close()

val_file.close()
train_writer.close()