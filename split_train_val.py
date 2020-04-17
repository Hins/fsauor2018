# -*- coding: utf-8 -*-

import random

pos_samples = []
neg_samples = []

with open("samples.csv", 'r') as f:
    for line in f:
        elems = line.strip('\r\n').split('\t')
        if elems[1] == "1":
            pos_samples.append(line)
        else:
            neg_samples.append(line)
    f.close()

random.shuffle(pos_samples)
random.shuffle(neg_samples)

print(len(pos_samples))
print(len(neg_samples))
pos_train_sample = pos_samples[200:]
pos_val_sample = pos_samples[:200]
print(len(pos_train_sample))
print(len(pos_val_sample))
neg_train_sample = neg_samples[200:]
neg_val_sample = neg_samples[:200]
print(len(neg_train_sample))
print(len(neg_val_sample))

pos_train_sample.extend(neg_train_sample)
print(len(pos_train_sample))
random.shuffle(pos_train_sample)
pos_val_sample.extend(neg_val_sample)
random.shuffle(pos_val_sample)

train_output_file = open("train.csv", 'w')
for sample in pos_train_sample:
    train_output_file.write(sample)
train_output_file.close()

val_output_file = open("val.csv", 'w')
for sample in pos_val_sample:
    val_output_file.write(sample)
val_output_file.close()