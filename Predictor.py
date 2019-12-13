# -*- coding: utf-8 -*-
# @Time    : 2019/12/11 12:02
# @Author  : Xiaotong Pan
# @Desc    : predict corpus data

import tensorflow as tf
import jieba
import argparse
import numpy as np

from datetime import timedelta,datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header

parser = argparse.ArgumentParser(description='corpus prediction')
parser.add_argument('--segment_gran', type=str, default='word', help='segmentation granularity')
#parser.add_argument('--input', type=str, default='/home/work/xtpan/hotel/data/2019-12-10.corpus', help='validation file')
parser.add_argument('--input', type=str, default='/home/work/xtpan/hotel/data/label.dat', help='validation file')
parser.add_argument('--word_vocab_file', type=str, default='/home/work/xtpan/hotel/data/vocab.txt', help='word vocabulary file')
#parser.add_argument('--word_model_file', type=str, default='/home/work/xtpan/hotel/model/256binary/59/model_eval_500', help='word model file')
parser.add_argument('--word_model_file', type=str, default='/home/work/xtpan/hotel/model/512New/0/model_eval_600', help='word model file')
parser.add_argument('--char_vocab_file', type=str, default='/home/work/xtpan/hotel/data/vocab.char.txt', help='char vocabulary file')
parser.add_argument('--char_model_file', type=str, default='/home/work/xtpan/hotel/model/256binaryChar/37/model_eval_500/', help='word model file')
parser.add_argument('--word_output', type=str, default='/home/work/xtpan/hotel/data/badcase.word.txt', help='word badcase file')
parser.add_argument('--char_output', type=str, default='/home/work/xtpan/hotel/data/badcase.char.txt', help='char badcase file')
parser.add_argument('--negative_dict', type=str, default='/home/work/xtpan/hotel/data/negative.dict', help='char badcase file')
parser.add_argument('--mode', type=str, default='daily', help='validation mode')
parser.add_argument('--batch_size', type=int, default=512, help='prediction batch size')
args = parser.parse_args()

if args.segment_gran == 'word':
    word_dict = {}
    with open(args.word_vocab_file, 'r') as f:
        for idx, line in enumerate(f):
            word_dict[line.strip('\r\n')] = idx
        f.close()
elif args.segment_gran == 'char':
    char_dict = {}
    with open(args.char_vocab_file, 'r') as f:
        for idx, line in enumerate(f):
            char_dict[line.strip('\r\n')] = idx
        f.close()

output_file = open(args.word_output if args.segment_gran == "word" else args.char_output, 'w')
negative_set = set()
with open(args.negative_dict, 'r') as f:
    for line in f:
        negative_set.add(line.strip('\r\n'))
    f.close()

if args.mode == "daily":
    output_list = []
    with open(args.input, 'r') as f:
        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess, ["serve"], args.word_model_file if args.segment_gran == "word" else args.char_model_file)
            for line in f:
                line = line.strip('\r\n')
                kw_flag = False
                for kw in negative_set:
                    if kw in line:
                        output_file.write(line + "\t0\n")
                        output_list.append(line + "\t0")
                        kw_flag = True
                        break
                if kw_flag == True:
                    continue
                if args.segment_gran == 'word':
                    features = [0 if item not in word_dict else word_dict[item] for item in jieba.cut(line, cut_all=False) if item != ',']
                elif args.segment_gran == 'char':
                    features = [0 if item not in char_dict else char_dict[item] for item in line if item != ',']
                features = np.asarray(features, dtype=np.int32)
                features = np.reshape(features, newshape=[1, features.shape[0]])
                ret = sess.run('lstm-output/final_logits:0',
                               feed_dict={'source_tokens:0': features,
                                          "sequence_length:0": [features.shape[1]]})
                label = "0" if ret[0][0] > ret[0][1] else "1"
                output_list.append(line + "\t" + label)
                output_file.write(line + "\t" + label + "\n")
        f.close()
else:
    if args.segment_gran == 'word':
        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess, ["serve"], args.word_model_file)
            graph = tf.get_default_graph()
            total_count = 0
            correct_count = 0
            with open(args.input, 'r') as f:
                for line in f:
                    total_count += 1
                    features = np.asarray([0 if item not in word_dict else word_dict[item] for item in jieba.cut(line.split("####")[0], cut_all=False) if item != ','],
                                          dtype=np.int32)
                    features = np.reshape(features, newshape=[1, features.shape[0]])
                    ret = sess.run('lstm-output/final_logits:0',
                             feed_dict={'source_tokens:0': features,
                                        "sequence_length:0": [features.shape[1]]})
                    if ret[0][0] >= ret[0][1]:
                        output_file.write(line.split("####")[0] + '\n')
                    else:
                        correct_count += 1
                f.close()
            print("total count is {0}, correct count is {1}".format(total_count, correct_count))
    elif args.segment_gran == 'char':
        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess, ["serve"], args.char_model_file)
            graph = tf.get_default_graph()
            total_count = 0
            correct_count = 0
            with open(args.input, 'r') as f:
                for line in f:
                    total_count += 1
                    features = np.asarray([0 if item not in char_dict else char_dict[item] for item in
                                           line.split("####")[0] if item != ','],
                                          dtype=np.int32)
                    features = np.reshape(features, newshape=[1, features.shape[0]])
                    ret = sess.run('lstm-output/final_logits:0',
                                   feed_dict={'source_tokens:0': features,
                                              "sequence_length:0": [features.shape[1]]})
                    if ret[0][0] >= ret[0][1]:
                        print(ret)
                        output_file.write(line.split("####")[0] + '\n')
                    else:
                        correct_count += 1
                f.close()
            print("total count is {0}, correct count is {1}".format(total_count, correct_count))
output_file.close()