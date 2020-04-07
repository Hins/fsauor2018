# -*- coding: utf-8 -*-

from config import cfg
import random

train_file = open('train.csv', 'w')
val_file = open('val.csv', 'w')

with open('train.csv.all', 'r') as f:
    for line in f:
        if random.random() > cfg.train_set_ratio:
            val_file.write(line)
        else:
            train_file.write(line)
    f.close()

val_file.close()
train_file.close()