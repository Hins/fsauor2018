# -*- coding: utf-8 -*-

import json as js

json_obj = js.loads(open('corpus.json').read())
kb_dict = {}
for obj in json_obj:
    if "query" not in obj or "matchEId" not in obj:
        continue
    if obj["matchEId"] not in kb_dict:
        kb_dict[obj["matchEId"]] = []
    kb_dict[obj["matchEId"]].append(obj["query"].replace('\n', '').encode('utf-8'))
o_f = open('enhanced_buy8.dat', 'w')
with open('input.txt', 'r') as f:
    for line in f:
        elements = line.strip('\r\n').split('\t')
        orig_query = elements[0]
        xiaomiwang_query = elements[1]
        kbs = elements[2].split(';')
        dedup_dict = {}
        for kb_id in kbs:
            for match in kb_dict[kb_id]:
                if match in dedup_dict:
                    continue
                dedup_dict[match] = 1
                match = match.replace(r'\r','')
                o_f.write(orig_query + "\t" + xiaomiwang_query + "\t" + match + "\n")
    f.close()
o_f.close()