
from collections import namedtuple
from utils import print_out
from thrid_utils import read_vocab
import numpy as np
import jieba

UNK_ID = 0
SOS_ID = 1
EOS_ID = 2

def _padding(tokens_list, max_len):
    ret = np.zeros((len(tokens_list), max_len), np.int32)
    for i, t in enumerate(tokens_list):
        t = t + (max_len - len(t)) * [EOS_ID]
        ret[i] = t
    return ret

def _tokenize(content, w2i, max_tokens=1200, reverse=False, split=True):
    def get_tokens(content):
        tokens = content.strip().split()
        ids = []
        for t in tokens:
            if t in w2i:
                ids.append(w2i[t])
            else:
                for c in t:
                    ids.append(w2i.get(c, UNK_ID))
        return ids

    if split:
        ids = get_tokens(content)
    else:
        ids = [w2i.get(t, UNK_ID) for t in content.strip().split()]
    if reverse:
        ids = list(reversed(ids))
    tokens = [SOS_ID] + ids[:max_tokens] + [EOS_ID]
    return tokens

class DataItem(namedtuple("DataItem", ('content', 'length','labels','id'))):
    pass

class DataSet(object):
    def __init__(self, data, vocab_file, label_file,  reverse=False, split_word=True, max_len=1200):
        self.reverse = reverse
        self.split_word = split_word
        self.data = data
        self.max_len = max_len

        self.vocab, self.w2i = read_vocab(vocab_file)
        self.i2w = {v: k for k, v in self.w2i.items()}
        self.label_names, self.l2i = read_vocab(label_file)
        self.i2l = {v: k for k, v in self.l2i.items()}

        self.tag_l2i = {"-1": 0, "1": 1}
        self.tag_i2l = {v: k for k, v in self.tag_l2i.items()}

        self._raw_data = []
        self.items = []
        self._preprocess()

    def get_label(self, labels, l2i, normalize=False):
        one_hot_labels = np.zeros(len(l2i), dtype=np.float32)
        for n in labels:
            if n:
                one_hot_labels[l2i[n]] = 1

        if normalize:
            one_hot_labels = one_hot_labels / len(labels)
        return one_hot_labels

    def _preprocess(self):
        print_out("# Start to preprocessing data...")
        content = _tokenize(self.data, self.w2i, self.max_len, self.reverse, self.split_word)
        item_labels = []
        for label_name in self.label_names:
            labels = [""]
            labels = self.get_label(labels, self.tag_l2i)
            item_labels.append(labels)
        self._raw_data.append(DataItem(content=content,labels=np.asarray(item_labels), length=len(content),id=int("0")))

        self.num_batches = 1
        self.data_size = len(self._raw_data)

    def process_sentence(self):
        item = self._raw_data[0]
        content = [item.content]
        content = _padding(content,item.length)
        length = np.asarray([item.length])
        target = np.asarray([item.labels])
        id = [item.id]
        return content, length,target,id
        
    def get_sentence(self):
        yield process_sentence()

