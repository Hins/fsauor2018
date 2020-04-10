# -*- coding: utf-8 -*-

import tensorflow as tf
from config import cfg
import numpy as np

def generate_numpy_data(file_name):
    data_list = []
    length_list = []
    label_list = []
    max_len = -1
    in_file = open(file_name, 'r')
    for line in in_file:
        elements = line.strip('\r\n').split(',')
        label_list.append(int(elements[-1]))
        length_list.append(len(elements) - 1)
        if max_len < len(elements) - 1:
            max_len = len(elements) - 1
    in_file.seek(0)
    for line in in_file:
        elements = line.strip('\r\n').split('\t')
        features = elements[:-1]
        features += [0] * (max_len - len(features))
        data_list.append(features)
    in_file.close()
    return np.asarray(data_list, dtype=np.int32), np.asarray(length_list, dtype=np.int32), np.asarray(label_list, dtype=np.int32)

def vanilla_data_fetch(file_name):
    data_list = []
    label_list = []
    with open(file_name, 'r') as f:
        for line in f:
            elements = line.strip('\r\n').split(',')
            data_list.append([int(item) for item in elements[:-1]])
            label_list.append(int(elements[-1]))
        f.close()
    return data_list, label_list

def tfrecord_data_fetch(file_name):
    filename_queues = tf.train.string_input_producer([file_name])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queues)
    features = tf.parse_single_example(serialized_example,
        features={
            'feature': tf.FixedLenFeature([cfg.train_max_len_size], tf.int64),
            'feature_len': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    train_samples = tf.cast(features["feature"], tf.int32)
    train_samples_len = tf.cast(features["feature_len"], tf.int32)
    train_labels = tf.cast(features["label"], tf.int32)
    return train_samples, train_samples_len, train_labels

def streaming_data_fetch(file_name):
    def tfrd_extraction(serial_exmp):
        feature_list = tf.parse_single_example(serial_exmp,
                                               features={
                                                   'feature': tf.FixedLenFeature([cfg.train_max_len_size], tf.int64),
                                                   'feature_len': tf.FixedLenFeature([], tf.int64),
                                                   'label': tf.FixedLenFeature([], tf.int64),
                                               })
        feature = tf.cast(feature_list["feature"], dtype=tf.int32)
        len = tf.cast(feature_list["feature_len"], dtype=tf.int32)
        label = tf.cast(feature_list["label"], tf.int32)
        return feature, len, label

    dataset = tf.data.TFRecordDataset(file_name)
    dataset = dataset.map(tfrd_extraction)
    dataset = dataset.shuffle(10000)
    dataset = dataset.repeat(cfg.epoch_size)
    dataset = dataset.batch(cfg.batch_size)
    dataset = dataset.prefetch(1)  # prefetch
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

def slice_input_producer_data_fetch(file_name):
    features, lengths, labels = generate_numpy_data(file_name)
    input_queue = tf.train.slice_input_producer([features, lengths, labels], num_epochs=cfg.epoch_size, shuffle=True)
    features_batch, lengths_batch, labels_batch = tf.train.batch(input_queue, batch_size=cfg.batch_size, num_threads=4, capacity=64)
    return features_batch, lengths_batch, labels_batch

def string_input_producer_data_fetch(file_name):
    files = tf.train.string_input_producer([file_name])
    reader = tf.FixedLengthRecordReader(record_bytes=1+32*32*3)
    key, value = reader.read(files)
    return value
    '''
    features_batch, lengths_batch, labels_batch = tf.train.batch(input_queue, batch_size=cfg.batch_size, num_threads=4,
                                                                 capacity=64)
    return features_batch, lengths_batch, labels_batch
    '''

def pad_train_set(train_feature, train_len, train_label):
    batch_samples = []
    batch_labels = []
    max_len = -1
    for i in range(train_feature.shape[0]):
        batch_samples.append(train_feature[i][:train_len[i]].astype(np.int32).tolist())
        batch_labels.append(train_label[i])
        if train_len[i] > max_len:
            max_len = train_len[i]
    samples = tf.keras.preprocessing.sequence.pad_sequences(batch_samples, maxlen=max_len, padding='post')
    length = np.zeros(shape=[len(batch_samples)], dtype=np.int32)
    length.fill(max_len)
    labels = np.asarray(batch_labels, dtype=np.float32)
    return samples, length, labels

def extract_val_ds(enable_dedup):
    validation_list = []
    validation_label_list = []
    dedup_val_dict = {}
    with open("val.csv", 'r') as f:
        for line in f:
            if enable_dedup:
                if line in dedup_val_dict:
                    continue
                dedup_val_dict[line] = 1
            elements = line.strip('\r\n').split(',')
            validation_list.append([int(item) for item in elements[:-1]])
            validation_label_list.append(int(elements[-1]))
        f.close()
    return validation_list, validation_label_list

validation_truc_size = 90000
def extract_val_samples(validation_list, global_max_len):
    val_max_len = -1
    for item in validation_list:
        if len(item) > val_max_len:
            val_max_len = len(item)
    if val_max_len > global_max_len:
        val_max_len = global_max_len
    validation_length = np.zeros(shape=[len(validation_list)], dtype=np.int32)
    validation_length.fill(val_max_len)
    samples = tf.keras.preprocessing.sequence.pad_sequences(validation_list, maxlen=val_max_len, padding='post')
    return samples[:validation_truc_size], validation_length[:validation_truc_size]