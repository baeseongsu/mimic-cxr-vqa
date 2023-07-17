import os
import math
import json
import pickle
import numpy as np
import pandas as pd
import _pickle as cPickle

from tqdm import tqdm
from PIL import Image, ImageFile
from random import random as rand
from random import randint, shuffle, choices
from sklearn.preprocessing import MultiLabelBinarizer
from loader_utils import get_random_word, batch_list_to_batch_tensors, Pipeline

ImageFile.LOAD_TRUNCATED_IMAGES = True  # Whether or not to load truncated image files. User code may change this.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


def truncate_tokens_pair(tokens_a, tokens_b, max_len, max_len_a=0, max_len_b=0, trunc_seg=None, always_truncate_tail=False):
    num_truncated_a = [0, 0]
    num_truncated_b = [0, 0]
    while True:
        if len(tokens_a) + len(tokens_b) <= max_len:
            break
        if (max_len_a > 0) and len(tokens_a) > max_len_a:
            trunc_tokens = tokens_a
            num_truncated = num_truncated_a
        elif (max_len_b > 0) and len(tokens_b) > max_len_b:
            trunc_tokens = tokens_b
            num_truncated = num_truncated_b
        elif trunc_seg:
            # truncate the specified segment
            if trunc_seg == "a":
                trunc_tokens = tokens_a
                num_truncated = num_truncated_a
            else:
                trunc_tokens = tokens_b
                num_truncated = num_truncated_b
        else:
            # truncate the longer segment
            if len(tokens_a) > len(tokens_b):
                trunc_tokens = tokens_a
                num_truncated = num_truncated_a
            else:
                trunc_tokens = tokens_b
                num_truncated = num_truncated_b
        # whether always truncate source sequences
        if (not always_truncate_tail) and (rand() < 0.5):
            del trunc_tokens[0]
            num_truncated[0] += 1
        else:
            trunc_tokens.pop()
            num_truncated[1] += 1
    return num_truncated_a, num_truncated_b


def is_howmany(q, a, label2ans):
    if "how many" in q.lower() or ("number of" in q.lower() and "number of the" not in q.lower()) or "amount of" in q.lower() or "count of" in q.lower():
        if a is None or answer_filter(a, label2ans):
            return True
        else:
            return False
    else:
        return False


def answer_filter(answers, label2ans, max_num=10):
    for ans in answers["labels"]:
        if label2ans[ans].isdigit() and max_num >= int(label2ans[ans]):
            return True
    return False


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
                tokens.append(self.word2idx.get(w, self.padding_idx - 1))
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, "wb"))
        print("dictionary dumped to %s" % path)

    @classmethod
    def load_from_file(cls, path):
        print("loading dictionary from %s" % path)
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


def pre_processing(tokenizer, sentence):
    sentence = sentence.lower()
    if "? -yes/no" in sentence:
        sentence = sentence.replace("? -yes/no", "")
    if "? -open" in sentence:
        sentence = sentence.replace("? -open", "")
    if "? - open" in sentence:
        sentence = sentence.replace("? - open", "")
    sentence = sentence.replace(",", "").replace("?", "").replace("'s", " 's").replace("...", "").replace("x ray", "x-ray").replace(".", "")
    token = tokenizer.tokenize(sentence)
    return token


def _create_entry(img, data, answer):
    if None != answer:
        answer.pop("image_name")
        answer.pop("qid")
    entry = {
        "qid": data["qid"],
        "image_name": data["image_name"],
        "image": img,
        "question": data["question"],
        "answer": answer,
        "answer_type": data["answer_type"],
        "question_type": data["question_type"],
        "phrase_type": data["phrase_type"],
        "image_organ": data["image_organ"],
    }
    return entry


def _load_dataset(args, dataroot, name, img_id2val, label2ans):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test'
    """
    data_path = os.path.join(dataroot, name + "set.json")
    samples = json.load(open(data_path))
    samples = sorted(samples, key=lambda x: x["qid"])
    answer_path = os.path.join(dataroot, "cache", "%s_target.pkl" % name)
    answers = cPickle.load(open(answer_path, "rb"))
    answers = sorted(answers, key=lambda x: x["qid"])
    entries = []
    for sample, answer in zip(samples, answers):
        img_id = sample["image_name"]
        if args.vqa_rad == "all":
            entries.append(_create_entry(img_id2val[img_id], sample, answer))
        elif args.vqa_rad == "chest":
            if sample["image_organ"] in {"CHEST", " CHEST", "CHEST "}:
                entries.append(_create_entry(img_id2val[img_id], sample, answer))
        elif args.vqa_rad == "head":
            if sample["image_organ"] in {"HEAD", " HEAD", "HEAD "}:
                entries.append(_create_entry(img_id2val[img_id], sample, answer))
        elif args.vqa_rad == "abd":
            if sample["image_organ"] in {"ABD", " ABD", "ABD "}:
                entries.append(_create_entry(img_id2val[img_id], sample, answer))
    return entries


class VQAMIMICDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
        split,
        file_src,
        img_root,
        batch_size,
        tokenizer,
        preproc_pipeline=None,
    ):
        super().__init__()
        self.args = args
        self.split = split
        self.file_src = file_src
        self.img_root = img_root
        self.batch_size = batch_size

        self.tokenizer = tokenizer  # tokenize function
        self.preproc_pipeline = preproc_pipeline

        # load dataset
        if args.exp_type == "all":
            file_path = os.path.join(file_src, f"{split}.json")
            entries = pd.DataFrame(json.load(open(file_path)))
            print(f"load {file_path}")
        elif args.exp_type == "ref":
            file_path = os.path.join(file_src, f"{split}_ref.json")
            entries = pd.DataFrame(json.load(open(file_path)))
            print("Use grounding dataset")
            print(f"load {file_path}")
        else:
            raise ValueError
        

        # label indexer
        mlb = MultiLabelBinarizer()

        # ans2idx
        ans2idx_fpath = os.path.join(file_src, "ans2idx.json")
        self.ans2idx = json.load(open(ans2idx_fpath, "rb"))
        mlb.fit([sorted([k for k in self.ans2idx.keys()])])
        print(len(mlb.classes_), mlb.classes_)

        self.ex_list = []
        entries = entries.to_dict("records")

        for entry in tqdm(entries):
            tokens = pre_processing(self.tokenizer, entry["question"])
            answer = str(entry["answer"])  # True, False
            assert answer != None

            img_path = os.path.join(img_root, entry["image_path"])

            # TODO: use the collator...
            if not isinstance(entry["answer"], list):
                entry["answer"] = [entry["answer"]]
            target = mlb.transform([entry["answer"]])
            target = torch.FloatTensor(target).squeeze(0)

            self.ex_list.append((img_path, tokens, target, "CLOSED", None))

        print(f"Load {len(self.ex_list)} documents")

        del entries

    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):
        instance = self.ex_list[idx]
        instance = self.preproc_pipeline(instance)
        return instance

    def __iter__(self):  # iterator to load data
        for __ in range(math.ceil(len(self.ex_list) / float(self.batch_size))):
            batch = []
            for __ in range(self.batch_size):
                idx = randint(0, len(self.ex_list) - 1)
                batch.append(self.__getitem__(idx))
            # To Tensor
            yield batch_list_to_batch_tensors(batch)


class PipelineForVQAMIMIC(Pipeline):
    """
    Modified Version (2022.07.16, Seongsu Bae)
    """

    def __init__(
        self,
        args,
        tokenizer,
        max_seq_len=512,
        len_vis_input=256,
    ):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer  # tokenizer # function from token to token index
        self.max_seq_len = max_seq_len
        self.len_vis_input = len_vis_input

        # for images
        self._transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize([512, 512]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, instance):
        img_path, tokens_b, ans_tk, ans_type, organ = instance

        # 1) input ids
        tokens_a = ["[UNK]"] * self.len_vis_input
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # 2) segment ids
        segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
        
        # zero padding
        n_pad = self.max_seq_len - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        # 3) attention mask
        attention_mask = torch.tensor([1] * len(tokens) + [0] * n_pad, dtype=torch.long)
        attention_mask = attention_mask.unsqueeze(0).expand(self.max_seq_len, self.max_seq_len).clone()

        # load images
        img = Image.open(img_path)
        # transform images
        img = self._transform(img)

        # positional embedding for visual part
        vis_pe = torch.arange(2048, dtype=torch.float)
        vis_pe = vis_pe.unsqueeze(0).expand(len(tokens_a), 2048)

        # answer type
        ans_type = torch.tensor(0)
        # elif ans_type in ["OPEN", "OPEN "]:
        #     ans_type = torch.tensor(1)

        # organ type
        organ = torch.tensor(0)

        return (input_ids, segment_ids, attention_mask, img, vis_pe, ans_tk, ans_type, organ)
