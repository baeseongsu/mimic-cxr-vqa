# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         create_dictionary
# Description:  question->word->dictionary for validation & training
# Author:       Boliu.Kelvin
# Date:         2020/4/5
# -------------------------------------------------------------------------------

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import _pickle as cPickle


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        if "? -yes/no" in sentence:
            sentence = sentence.replace("? -yes/no", "")
        if "? -open" in sentence:
            sentence = sentence.replace("? -open", "")
        if "? - open" in sentence:
            sentence = sentence.replace("? - open", "")
        sentence = sentence.replace(",", "").replace("?", "").replace("'s", " 's").replace("...", "").replace("x ray", "x-ray").replace(".", "")
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                # if a word is not in dictionary, it will be replaced with the last word of dictionary.
                tokens.append(self.word2idx.get(w, self.padding_idx - 1))
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, "wb"))
        print("dictionary dumped to %s" % path)

    @classmethod
    def load_from_file(cls, path):
        print("loading dictionary from %s" % path)
        print(os.getcwd())
        word2idx, idx2word = cPickle.load(open(path, "rb"))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def create_dictionary(dataroot, train_file, test_file):
    dictionary = Dictionary()
    files = [train_file, test_file]
    for path in files:
        data_path = os.path.join(dataroot, path)
        df = json.load(open(data_path))
        df = pd.DataFrame(df)
        print("processing the {}".format(path))
        for id, row in df.iterrows():
            dictionary.tokenize(row["question"], True)  # row[0]: id , row[1]: question , row[2]: answer

    return dictionary


def create_glove_embedding_init(idx2word, glove_file):
    print("creating glove embeddings...")
    word2emb = {}
    with open(glove_file, "r") as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(" ")) - 1
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(" ")
        word = vals[0]
        vals = list(map(float, vals[1:]))
        word2emb[word] = np.array(vals)
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Med VQA")
    parser.add_argument("--input_path", type=str, help="Path to input data")
    parser.add_argument("--glove_path", type=str, default="../language", help="Path to glove embedding")

    parser.add_argument("--trainfile", type=str, help="Name of the train file", default="train.json")
    parser.add_argument("--validfile", type=str, help="Name of the test file", default="valid.json")

    args = parser.parse_args()
    data = args.input_path
    train_file = args.trainfile
    valid_file = args.validfile
    d = create_dictionary(data, train_file, valid_file)
    os.makedirs(os.path.join(data, "preprocess_pubmedclip"), exist_ok=True)
    d.dump_to_file(os.path.join(data, "preprocess_pubmedclip/dictionary.pkl"))

    d = Dictionary.load_from_file(os.path.join(data, "preprocess_pubmedclip/dictionary.pkl"))
    emb_dim = 300
    glove_file = os.path.join(args.glove_path, "glove.6B.%dd.txt" % emb_dim)
    weights, word2emb = create_glove_embedding_init(d.idx2word, glove_file)
    print(weights.shape)
    np.save(os.path.join(data, "preprocess_pubmedclip/glove6b_init_%dd.npy" % emb_dim), weights)
    print("Process finished successfully!")
