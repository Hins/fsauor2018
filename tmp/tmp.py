# -*- coding: utf-8 -*-

import numpy as np
from textrank4zh import TextRank4Sentence
import json
import jieba

keywords_exist_dict = {}
with open('dedupe.csv', 'r') as f:
    for line in f:
        elements = line.strip('\r\n').split('\t')
        keywords_exist_dict[elements[0]] = elements[1]
    f.close()

rr_keywords_dict = {}
with open("keywords.csv", 'r') as f:
    for line in f:
        elements = line.strip('\r\n').split('\t')
        js_obj = json.loads(elements[1])
        keywords_list = []
        for k,v in js_obj.items():
            keywords_list.append(k)
        rr_keywords_dict[elements[0]] = keywords_list
    f.close()

white_list_kw_list = []
with open('keyword.csv', 'r') as f:
    for line in f:
        white_list_kw_list.append(line.strip('\r\n'))
    f.close()

o_f = open('all.csv', 'w')
with open('cluster_all.csv', 'r') as f:
    for line in f:
        elements = line.strip('\r\n').split('\t')
        if len(elements) < 2:
            o_f.write(line)
            print(line)
            continue
        query = elements[0]
        answer = elements[1]
        sub_query_list = [item for item in query.split('？') if item != "" and item.find("谢谢") == -1 and item.find("<br") == -1]
        if query in keywords_exist_dict:
            keyword_string = keywords_exist_dict[query]
        else:
            if query in rr_keywords_dict:
                keywords = rr_keywords_dict[query]
                for item in white_list_kw_list:
                    if query.find(item) != -1:
                        keywords.append("关键字:" + item)
                seg_list = [item for item in jieba.cut(query, cut_all=False)]
                for word in seg_list:
                    if word.endswith("法规") or word.endswith("依据") or word.endswith("费") or word.endswith("车") or word.endswith("企业") or word.endswith("公司") or word.endswith("机构") or word.endswith("资产") or word.endswith("服务") or word.endswith("税") or word.endswith("发票") or word.endswith("赔偿"):
                        if ("关键字:" + word) not in keywords:
                            keywords.append("关键字:" + word)
                    if "法规" in seg_list and seg_list[0] != "法规":
                        if ("关键字:" + seg_list[(seg_list.index("法规") if seg_list.count("法规") > 1 else seg_list.index("法规")) - 1] + "法规") not in keywords:
                            keywords.append("关键字:" + seg_list[
                                (seg_list.index("法规") if seg_list.count("法规") > 1 else seg_list.index("法规")) - 1] + "法规")
                    if "依据" in seg_list and seg_list[0] != "依据":
                        if ("关键字:" + seg_list[(seg_list.index("依据") if seg_list.count("依据") > 1 else seg_list.index("依据")) - 1] + "依据") not in keywords:
                            keywords.append("关键字:" + seg_list[
                            (seg_list.index("依据") if seg_list.count("依据") > 1 else seg_list.index("依据")) - 1] + "依据")
                    if "费" in seg_list and seg_list[0] != "费":
                        if ("关键字:" + seg_list[(seg_list.index("费") if seg_list.count("费") > 1 else seg_list.index("费")) - 1] + "费") not in keywords:
                            keywords.append("关键字:" + seg_list[(seg_list.index("费") if seg_list.count("费") > 1 else seg_list.index("费")) - 1] + "费")
                    if "车" in seg_list and seg_list[0] != "车":
                        if ("关键字:" + seg_list[(seg_list.index("车") if seg_list.count("车") > 1 else seg_list.index("车")) - 1] + "车") not in keywords:
                            keywords.append("关键字:" + seg_list[(seg_list.index("车") if seg_list.count("车") > 1 else seg_list.index("车")) - 1] + "车")
                    if "法规" in seg_list and seg_list[0] != "法规":
                        if ("关键字:" + seg_list[(seg_list.index("法规") if seg_list.count("法规") > 1 else seg_list.index("法规")) - 1] + "法规") not in keywords:
                            keywords.append("关键字:" + seg_list[(seg_list.index("法规") if seg_list.count("法规") > 1 else seg_list.index("法规")) - 1] + "法规")
                    if "企业" in seg_list and seg_list[0] != "企业":
                        if ("关键字:" + seg_list[(seg_list.index("企业") if seg_list.count("企业") > 1 else seg_list.index("企业")) - 1] + "企业") not in keywords:
                            keywords.append("关键字:" + seg_list[(seg_list.index("企业") if seg_list.count("企业") > 1 else seg_list.index("企业")) - 1] + "企业")
                    if "公司" in seg_list and seg_list[0] != "公司":
                        if ("关键字:" + seg_list[(seg_list.index("公司") if seg_list.count("公司") > 1 else seg_list.index("公司")) - 1] + "公司") not in keywords:
                            keywords.append("关键字:" + seg_list[(seg_list.index("公司") if seg_list.count("公司") > 1 else seg_list.index("公司")) - 1] + "公司")
                    if "机构" in seg_list and seg_list[0] != "机构":
                        if ("关键字:" + seg_list[(seg_list.index("机构") if seg_list.count("机构") > 1 else seg_list.index(
                                "机构")) - 1] + "机构") not in keywords:
                            keywords.append("关键字:" + seg_list[(seg_list.index("机构") if seg_list.count("机构") > 1 else seg_list.index("机构")) - 1] + "机构")
                    if "资产" in seg_list and seg_list[0] != "资产":
                        if ("关键字:" + seg_list[(seg_list.index("资产") if seg_list.count("资产") > 1 else seg_list.index("资产")) - 1] + "资产") not in keywords:
                            keywords.append("关键字:" + seg_list[(seg_list.index("资产") if seg_list.count("资产") > 1 else seg_list.index("资产")) - 1] + "资产")
                    if "服务" in seg_list and seg_list[0] != "服务":
                        if ("关键字:" + seg_list[(seg_list.index("服务") if seg_list.count("服务") > 1 else seg_list.index("服务")) - 1] + "服务") not in keywords:
                            keywords.append("关键字:" + seg_list[(seg_list.index("服务") if seg_list.count("服务") > 1 else seg_list.index("服务")) - 1] + "服务")
                    if "税" in seg_list and seg_list[0] != "税":
                        if ("关键字:" + seg_list[(seg_list.index("税") if seg_list.count("税") > 1 else seg_list.index("税")) - 1] + "税") not in keywords:
                            keywords.append("关键字:" + seg_list[(seg_list.index("税") if seg_list.count("税") > 1 else seg_list.index("税")) - 1] + "税")
                    if "发票" in seg_list and seg_list[0] != "发票":
                        if ("关键字:" + seg_list[(seg_list.index("发票") if seg_list.count("发票") > 1 else seg_list.index("发票")) - 1] + "发票") not in keywords:
                            keywords.append("关键字:" + seg_list[(seg_list.index("发票") if seg_list.count("发票") > 1 else seg_list.index("发票")) - 1] + "发票")
                    if "赔偿" in seg_list and seg_list[0] != "赔偿":
                        if ("关键字:" + seg_list[(seg_list.index("赔偿") if seg_list.count("赔偿") > 1 else seg_list.index("赔偿")) - 1] + "赔偿") not in keywords:
                            keywords.append("关键字:" + seg_list[(seg_list.index("赔偿") if seg_list.count("赔偿") > 1 else seg_list.index("赔偿")) - 1] + "赔偿")
                keywords = list(set(keywords))
                keywords_bak = []
                for k in keywords:
                    if k.find("关键字:") == -1:
                        keywords_bak.append("关键字:" + k)
                    else:
                        keywords_bak.append(k)
                keyword = ",".join(keywords_bak)
        header = True
        dedup_list = []
        for sub_query in sub_query_list:
            if sub_query in dedup_list:
                continue
            dedup_list.append(sub_query)
            if header == True:
                o_f.write(query + "\t" + sub_query + "\t" + answer + "\t" + keyword + "\n")
                header = False
            else:
                o_f.write("\t" + sub_query + "\t\n")
    f.close()
o_f.close()
'''
o_f = open('short.csv', 'w')
with open('merged.txt', 'r') as f:
    for line in f:
        l = line.strip('\r\n')
        elements = l.split('\t')
        if len(elements[0]) < 40 and elements[0] != "":
            o_f.write(line)
    f.close()
o_f.close()
'''

'''
with open('yesorno.txt', 'r') as f:
    for line in f:
        elements = line.strip('\r\n').split('\t')
        s = elements[0]
        if elements[0] == "是否" and
    f.close()
'''

'''
tax_dict = {}
with open('tax.txt', 'r') as f:
    for line in f:
        tax = line.strip('\r\n')
        if tax in tax_dict:
            continue
        tax_dict[tax] = len(tax_dict)
    f.close()
'''

'''
tax_key_dict = {}
i_f = open('key.txt', 'r')
counter = 0
for line in i_f:
    l = line.strip('\r\n')
    counter += 1
    for tax in tax_dict:
        if tax in l:
            tax_key_dict[l] = tax
            break
print(len(tax_key_dict))
print(counter)
i_f.seek(0)
'''

'''
keywords_dict = {}
with open('keywords.csv', 'r') as f:
    for line in f:
        elements = line.strip('\r\n').split('\t')
        j = json.loads(elements[1]).items()
        keyword_list = []
        for k,v in j:
            if float(v) > 0.5:
                keyword_list.append(k)
        keywords_dict[elements[0]] = keyword_list
    f.close()
print(len(keywords_dict))

o_f = open('o.csv', 'w')
i_f = open('key.txt', 'r')
pre_query = ""
for line in i_f:
    elements = line.strip('\r\n').split('\t')
    query = elements[0]
    if query == "":
        query = pre_query
    else:
        pre_query = query
    sub_query = elements[1]
    keywords = ""
    keyword_list = []
    if query in keywords_dict:
        for keyword in keywords_dict[query]:
            if keyword in sub_query:
                keyword_list.append(keyword)
    if len(keyword_list) > 0:
        o_f.write(",".join([("关键字:" + item) for item in keyword_list]) + '\n')
    else:
        o_f.write("\n")
i_f.close()
o_f.close()
'''

'''
prev_dict = {}
with open('part.txt', 'r') as f:
    for line in f:
        prev_dict[line.strip('\r\n')] = len(prev_dict)
    f.close()

all_dict = {}
with open('all.txt', 'r') as f:
    for line in f:
        elements = line.strip('\r\n').split('\t')
        if len(elements) < 2:
            continue
        query = elements[0]
        answer = elements[1]
        if query in prev_dict:
            continue

        flag = False
        for k,v in tax_dict.items():
            if k in query:
                flag = True
                break
        if flag == False:
            continue
        if query.count('？') == 1:
            all_dict[len(all_dict)] = query + '\t' + answer
        if query.count('？') == 2 and query[-1] == "？":
            all_dict[len(all_dict)] = query + '\t' + answer
    f.close()

random_id = np.random.choice(len(all_dict), 500)
tr4s = TextRank4Sentence()

o_f = open('output.txt', 'w')
for id in random_id:
    sub_ques = all_dict[id].split('\t')[0].split('？', 2)
    tr4s.analyze(text=sub_ques[0], lower=True, source='all_filters')
    for item in tr4s.get_key_sentences(num=1):
        abstract = item.sentence
    target_tax = ""
    for tax in tax_dict:
        if tax in all_dict[id].split('\t')[0]:
            target_tax = tax
            break
    o_f.write(all_dict[id] + '\t' + abstract + '\t' + target_tax + '\n')
    if len(sub_ques) > 2:
        tr4s.analyze(text=sub_ques[1], lower=True, source='all_filters')
        for item in tr4s.get_key_sentences(num=1):
            abstract = item.sentence
        o_f.write('\t\t' + abstract + '\t' + target_tax + '\n')
o_f.close()
'''