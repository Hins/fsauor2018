# -*- coding: utf-8 -*-
# @Time    : 2019/12/12 13:56
# @Author  : Xiaotong Pan
# @Desc    :

import argparse
import numpy as np

parser = argparse.ArgumentParser(description='sentiment offline mine logic')
parser.add_argument('--cond', type=str, default='provider', help='query log')
parser.add_argument('--dict', type=str, default='sentiment.dict.20191212', help='sentiment dict')
parser.add_argument('--log', type=str, default='log', help='query log')
#parser.add_argument('--log', type=str, default='sample.txt', help='query log')
parser.add_argument('--log_all', type=str, default='log.all', help='all query log')
parser.add_argument('--output', type=str, default='neg_sample.dat', help='negative sample file')
parser.add_argument('--output_all', type=str, default='neg_sample.dat.all', help='negative all sample file')
args = parser.parse_args()

sentiment_set = set()
with open(args.dict, 'r') as f:
    for line in f:
        sentiment_set.add(line.strip('\r\n'))
    f.close()
print("sentiment dict size is {0}".format(len(sentiment_set)))

output_file = open(args.output if args.cond == "provider" else args.output_all, 'w')
dedup_set = set()
START_FLAG = "\"query\":\""
END_FLAG = "\",\"confidence\""
START_FLAG_PROVIDER = "request={\"query\":\""
END_FLAG_PROVIDER = "\",\"requestid"
start_flag = START_FLAG_PROVIDER if args.cond == "provider" else START_FLAG
end_flag = END_FLAG_PROVIDER if args.cond == "provider" else END_FLAG
input_file = args.log if args.cond == "provider" else args.log_all
with open(input_file, 'r') as f:
    for line in f:
        line = line.strip('\r\n').lower()
        if start_flag not in line:
            continue
        start_index = line.find(start_flag)
        if start_index == -1:
            continue
        end_index = line.find(end_flag, start_index + len(start_flag))
        if end_index == -1:
            continue
        query = line[start_index + len(start_flag) : end_index]
        if query in dedup_set:
            continue
        dedup_set.add(query)
        for kw in sentiment_set:
            if kw in query:
                output_file.write(query + "\n")
                break
    f.close()
'''
dedup_set = list(dedup_set)
random_list = np.random.randint(0, len(dedup_set) - 1, 1000)
for i in random_list:
    output_file.write(dedup_set[i] + '\n')
'''
output_file.close()