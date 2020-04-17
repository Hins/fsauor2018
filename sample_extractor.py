# -*- coding: utf-8 -*-

import sys
import json as js

o_f = open('samples.csv', 'w')
dedup_query = {}
with open('sentiment.dict', 'r') as f:
    for line in f:
        elems = line.strip('\r\n').split('\t')
        if elems[1] == "0":
            key = elems[0]
            if key not in dedup_query:
                dedup_query[key] = 1
                o_f.write(elems[0] + "\t1\n")
        else:
            key = elems[0]
            if key not in dedup_query:
                dedup_query[key] = 1
                o_f.write(elems[0] + "\t0\n")
    f.close()

with open("new.txt", 'r') as f:
    for line in f:
        elems = line.strip('\r\n').split('\t')
        if elems[1] == "负面":
            key = elems[0]
            if key not in dedup_query:
                dedup_query[key] = 1
                o_f.write(elems[0] + "\t0\n")
        else:
            key = elems[0]
            if key not in dedup_query:
                dedup_query[key] = 1
                o_f.write(elems[0] + "\t1\n")
    f.close()

with open("negative.20191218.labelled", 'r') as f:
    for line in f:
        key = line.strip('\r\n')
        if key not in dedup_query:
            dedup_query[key] = 1
            o_f.write(line.strip('\r\n') + "\t0\n")
    f.close()

with open("labeled.1", 'r') as f:
    for line in f:
        elems = line.strip('\r\n').split('\t')
        if elems[2] == "正" or elems[2] == "1":
            key = elems[0]
            if key not in dedup_query:
                dedup_query[key] = 1
                o_f.write(elems[0] + "\t1\n")
        else:
            key = elems[0]
            if key not in dedup_query:
                dedup_query[key] = 1
                o_f.write(elems[0] + "\t0\n")
    f.close()

with open("labeled.2", 'r') as f:
    for line in f:
        elems = line.strip('\r\n').split('\t')
        if elems[1] == "正" or elems[1] == "0":
            key = elems[0]
            if key not in dedup_query:
                dedup_query[key] = 1
                o_f.write(elems[0] + "\t1\n")
        else:
            key = elems[0]
            if key not in dedup_query:
                dedup_query[key] = 1
                o_f.write(elems[0] + "\t0\n")
    f.close()

kb_json = js.loads(open('./kb').read())
neu_size = 0
for obj in kb_json["data"]:
    if "query" not in obj or "group" not in obj or "key" not in obj or "kId" not in obj:
        continue
    key = obj["query"].encode('utf-8')
    if key not in dedup_query:
        dedup_query[key] = 1
        o_f.write(key + "\t1\n")

kb_json = js.loads(open('./corpus').read())
for obj in kb_json:
    if "query" not in obj or "matchStdQ" not in obj:
        continue
    key = obj["query"].replace('\n', '').strip('\r\n').encode('utf-8')
    if key not in dedup_query:
        dedup_query[key] = 1
        o_f.write(key + "\t1\n")
    key = obj["matchStdQ"].strip('\r\n').encode('utf-8')
    if key not in dedup_query:
        dedup_query[key] = 1
        o_f.write(key + "\t1\n")

o_f.close()