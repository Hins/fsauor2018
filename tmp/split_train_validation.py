# -*- coding: utf-8 -*-

from config import cfg
import random
import tensorflow as tf
import numpy as np

'''
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
'''

train_max_len_size = 100
word_dict = {}
writer = tf.python_io.TFRecordWriter("train.tfrecords")
train_size = 0
with open('train.csv', 'r') as f:
    for line in f:
        elems = line.strip('\r\n').split('\t')
        if len(elems) is not 2:
            continue
        train_size += 1
        feature_bin = {}
        words = list(elems[0])
        label = int(elems[1])
        features = []
        for word in words:
            if word not in word_dict:
                word_dict[word] = len(word_dict) + 1
            features.append(word_dict[word])
        prev_length = len(features) if len(features) < train_max_len_size else train_max_len_size
        if len(features) < train_max_len_size:
            features += [0] * (train_max_len_size - len(features))
        else:
            features = features[:train_max_len_size]
        features = np.asarray(features)
        feature_bin["feature"] = tf.train.Feature(int64_list=tf.train.Int64List(value=features))
        feature_bin["feature_len"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[prev_length]))
        feature_bin["label"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        writer.write(tf.train.Example(features=tf.train.Features(feature=feature_bin)).SerializeToString())
f.close()
writer.close()

word_dict = {}
writer = tf.python_io.TFRecordWriter("test.tfrecords")
test_size = 0
with open('val.csv', 'r') as f:
    for line in f:
        elems = line.strip('\r\n').split('\t')
        if len(elems) is not 2:
            continue
        test_size += 1
        feature_bin = {}
        words = list(elems[0])
        label = int(elems[1])
        features = []
        for word in words:
            if word not in word_dict:
                word_dict[word] = len(word_dict) + 1
            features.append(word_dict[word])
        prev_length = len(features) if len(features) < train_max_len_size else train_max_len_size
        if len(features) < train_max_len_size:
            features += [0] * (train_max_len_size - len(features))
        else:
            features = features[:train_max_len_size]
        features = np.asarray(features)
        feature_bin["feature"] = tf.train.Feature(int64_list=tf.train.Int64List(value=features))
        feature_bin["feature_len"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[prev_length]))
        feature_bin["label"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        writer.write(tf.train.Example(features=tf.train.Features(feature=feature_bin)).SerializeToString())
f.close()
writer.close()
print("train size is {0}, test size is {1}".format(train_size, test_size))

with open('dict.txt', 'w') as f:
    for k,v in word_dict.items():
        f.write(k + '\t' + str(v) + '\n')
    f.close()