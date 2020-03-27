# -*- coding: utf-8 -*-

import argparse
import json as js
import jieba
import numpy as np
import random
import tensorflow as tf

parser = argparse.ArgumentParser(description='segmentation solution')
parser.add_argument('--segment_gran', type=str, default='word', help='segmentation granularity')
parser.add_argument('--mode', type=str, default='all', help='negative samples mode')
parser.add_argument('--sample_negative_all', type=str, default='neg_sample.dat.all', help='negative samples from all data')
parser.add_argument('--sample_negative', type=str, default='neg_sample.dat', help='negative samples')
parser.add_argument('--dict', type=str, default='baike.dict', help='self-defined dictionary')
parser.add_argument('--negative_label', type=str, default='negative.20191218.labelled', help='labeled negative samples')
args = parser.parse_args()

if args.segment_gran == "word":
    jieba.load_userdict(args.dict)

material = []
neu_material = set()
neg_material = set()

# load neural samples from knowledge database
kb_json = js.loads(open('../data/kmQuery.json').read())
neu_size = 0
for obj in kb_json["data"]:
    if "query" not in obj or "group" not in obj or "key" not in obj or "kId" not in obj:
        continue
    if args.segment_gran == "word":
        neu_material.add(" ".join([item for item in jieba.cut(obj["query"].encode('utf-8'), cut_all=False) if item != ","]) + "\t1")
    else:
        neu_material.add(" ".join([str(item) for item in obj["query"] if str(item) != ","]) + "\t1")
    if "similarQueries" in obj:
        for sim in obj["similarQueries"]:
            if "query" not in sim:
                continue
            if args.segment_gran == "word":
                neu_material.add(" ".join([item for item in jieba.cut(sim["query"].encode('utf-8'), cut_all=False) if item != ","]) + "\t1")
            else:
                neu_material.add(" ".join([str(item) for item in obj["query"] if str(item) != ","]) + "\t1")

# load neural samples from corpus
with open('/home/work/xtpan/hotel/data/2019-12-10.corpus', 'r') as f:
    for line in f:
        elements = line.strip('\r\n').split("####")
        if args.segment_gran == "word":
            neu_material.add(" ".join([item for item in jieba.cut(elements[0], cut_all=False) if item != ","]) + "\t1")
        else:
            neu_material.add(" ".join([str(item) for item in elements[0] if str(item) != ","]) + "\t1")
    f.close()

'''
# load negative samples from log
negative_file = args.sample_negative_all if args.mode == "all" else args.sample_negative
with open(negative_file, 'r') as f:
    for line in f:
        line = line.strip('\r\n')
        if random.random() < 0.5:
            continue
        if args.segment_gran == "word":
            neg_material.add(" ".join([item for item in jieba.cut(line, cut_all=False) if item != ","]) + "\t0")
        else:
            neg_material.add(" ".join([str(item) for item in line if str(item) != ","]) + "\t0")
    f.close()
'''

# load neural samples from sentiment dictionary
with open('sentiment.dict', 'r') as f:
    for line in f:
        line = line.strip('\r\n')
        elements = line.split('\t')
        label = "0" if elements[1].strip() == "-1" else "1"
        if label == "0":    # skip negative samples
            continue
        if args.segment_gran == "word":
            neu_material.add(" ".join([item for item in jieba.cut(elements[0], cut_all=False) if item != ","]) + "\t" + label)
        else:
            neu_material.add(" ".join([str(item) for item in elements[0] if str(item) != ","]) + "\t" + label)

# load samples from labeled data
with open('../data/new.txt', 'r') as f:
    for line in f:
        elements = line.strip('\r\n').split('\t')
        if elements[1] == "负面":
            if args.segment_gran == "word":
                neg_material.add(" ".join([item for item in jieba.cut(elements[0], cut_all=False) if item != ","]) + "\t0")
            else:
                neg_material.add(" ".join([str(item) for item in elements[0] if str(item) != ","]) + "\t0")
        elif elements[1] == "正向":
            if args.segment_gran == "word":
                neu_material.add(" ".join([item for item in jieba.cut(elements[0], cut_all=False) if item != ","]) + "\t1")
            else:
                neu_material.add(" ".join([str(item) for item in elements[0] if str(item) != ","]) + "\t1")
    f.close()

with open(args.negative_label, 'r') as f:
    for line in f:
        line = line.strip('\r\n')
        if args.segment_gran == "word":
            neg_material.add(" ".join([item for item in jieba.cut(line, cut_all=False) if item != ","]) + "\t0")
        else:
            neg_material.add(" ".join([str(item) for item in line if str(item) != ","]) + "\t0")

neu_sample_size = len(neu_material)
neg_sample_size = len(neg_material)
print("negative sample size is {0}, neural sample size is {1}".format(neg_sample_size, neu_sample_size))
neu_material = list(neu_material)
neg_material = list(neg_material)
if neu_sample_size > neg_sample_size:
    random_list = np.random.randint(0, len(neg_material) - 1, neu_sample_size - neg_sample_size)
    for i in random_list:
        material.append(neg_material[i])
else:
    random_list = np.random.randint(0, len(neu_material) - 1, neg_sample_size - neu_sample_size)
    for i in random_list:
        material.append(neu_material[i])
material.extend(neu_material)
material.extend(neg_material)
random.shuffle(material)

word_dict = {}
writer = tf.python_io.TFRecordWriter("train.tfrecords")
for line in material:
    words = line.split('\t')[0].split(' ')
    label = int(line.split('\t')[1])
    features = []
    for word in words:
        if word not in word_dict:
            word_dict[word] = len(word_dict) + 1
        features.append(word_dict[word])
    features = np.asarray(features)
    example = tf.train.Example(features=tf.train.Features(feature={
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        'feature': tf.train.Feature(bytes_list=tf.train.BytesList(value=[features.tostring()]))
    }))
    writer.write(example.SerializeToString())
writer.close()

with open('dict.txt', 'w') as f:
    for k,v in word_dict.items():
        f.write(k + '\t' + str(v) + '\n')
    f.close()

'''
with open('train.csv', 'w') as f:
    f.write("id,content,label\n")
    for idx, line in enumerate(material):
        f.write(str(idx) + ',' + line.split('\t')[0] + ',' + line.split('\t')[1] + '\n')
    f.close()
'''
