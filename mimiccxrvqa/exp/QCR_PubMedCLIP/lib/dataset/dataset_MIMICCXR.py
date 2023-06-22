from __future__ import print_function
import os
import json
import torch
import warnings
import numpy as np
import pandas as pd
import _pickle as cPickle

from tqdm import tqdm
from PIL import Image
from lib.utils import utils
from torch.utils.data import Dataset, DataLoader

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
COUNTING_ONLY = False


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
                if w not in self.word2idx:
                    print(f"{w} not in dictionary")
                # if a word is not in dictionary, it will be replaced with the last word of dictionary.
                tokens.append(self.word2idx.get(w, self.padding_idx-1))

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


def _create_entry(data, answer):
    entry = {
        "image_path": data["image_path"],
        "image_name": data["image_id"],
        "question": data["question"],
        "answer": answer,
        "answer_text": data["answer"],
        "answer_type": "CLOSED" if data["semantic_type"] == "verify" else "OPEN",
        "semantic_type": data["semantic_type"],
        "content_type": data["content_type"]
    }
    return entry


def is_json(myjson):
    try:
        json_object = json.loads(myjson)
    except ValueError:
        return False
    return True


def _load_dataset(dataroot, phase, ans2label, dataset_type="vqa", debug=False):
    """Load entries

    img2id: dict {img -> id} id can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test'
    """

    if dataset_type == "ub":
        pass
    else:
        data_path = os.path.join(os.path.dirname(dataroot), phase + ".json")
    
    samples = json.load(open(data_path))
    samples = pd.DataFrame(samples)

    entries = []
    for i, sample in tqdm(samples.iterrows()):
        labels = []
        answers = sample["answer"]
        for _ans in answers:
            if _ans in ans2label:
                labels.append(ans2label[_ans])
            else:
                print(f"{_ans} not exist in answer set")
                raise NotImplementedError
        entries.append(_create_entry(sample, {"labels": np.array(labels), "text": sample["answer"]}))

    return entries


class VQAMIMICCXRFeatureDataset(Dataset):
    def __init__(self, phase, cfg, dictionary, dataroot='data'):
        super(VQAMIMICCXRFeatureDataset, self).__init__()
        self.cfg = cfg
        self.phase = phase if not cfg.DEBUG else "test"
        self.dataroot = dataroot
        question_len = cfg.TRAIN.QUESTION.LENGTH
        assert phase in ["train", "valid", "test"]
        ans2label_path = os.path.join(dataroot, "cache", "ans2label_multilabel.pkl")
        label2ans_path = os.path.join(dataroot, "cache", "label2ans_multilabel.pkl")
        self.ans2label = cPickle.load(open(ans2label_path, "rb"))
        self.label2ans = cPickle.load(open(label2ans_path, "rb"))
        self.num_open_candidates = len(self.ans2label) - 2
        self.num_close_candidates = 2
        self.num_ans_candidates = len(self.ans2label)
        print(f"# of answers: {self.num_ans_candidates}")

        # End get the number of answer type class
        self.dictionary = dictionary
        print(f"Load {len(dictionary.word2idx)} words dictionary")

        # load dataset instances
        self.entries = _load_dataset(dataroot, self.phase, self.ans2label, self.cfg.DATASET.DATASET_TYPE, self.cfg.DEBUG)
        self.img_resolution = {}
        # load image data for MAML module
        if self.cfg.TRAIN.VISION.MAML:
            # TODO: load images
            self.img_resolution["maml"] = 84
            # images_path = os.path.join(dataroot, "resized_img", self.phase, "images84x84.pkl")
            # print("loading MAML image data from file: " + images_path)
            # starttime = time.time()
            # self.maml_images_data = cPickle.load(open(images_path, "rb"))
            # endtime = time.time()
            # print(f"finish: {endtime - starttime:.2f} sec")
            
        # load image data for Auto-encoder module
        if self.cfg.TRAIN.VISION.AUTOENCODER:
            # TODO: load images
            self.img_resolution["autoencoder"] = 128
            # images_path = os.path.join(dataroot, "resized_img", self.phase, "images128x128.pkl")
            # print("loading DAE image data from file: " + images_path)
            # starttime = time.time()
            # self.ae_images_data = cPickle.load(open(images_path, "rb"))
            # endtime = time.time()
            # print(f"finish: {endtime - starttime:.2f} sec")

        if self.cfg.TRAIN.VISION.CLIP:
            if self.cfg.TRAIN.VISION.CLIP_VISION_ENCODER == "RN50x4":
                # images_path = os.path.join(dataroot, "resized_img", self.phase, "images288x288.pkl")
                self.img_resolution["clip"] = 288
            else:
                # images_path = os.path.join(dataroot, "resized_img", self.phase, "images250x250.pkl")
                self.img_resolution["clip"] = 250
            # print(f"loading CLIP image data from file: {images_path}")
            # starttime = time.time()
            # self.clip_images_data = cPickle.load(open(images_path, "rb"))
            # endtime = time.time()
            # print(f"finish: {endtime - starttime:.2f} sec")

        # tokenization
        self.tokenize(question_len)
        self.tensorize()
        if cfg.TRAIN.VISION.AUTOENCODER and cfg.TRAIN.VISION.MAML:
            self.v_dim = cfg.TRAIN.VISION.V_DIM * 2
        else:
            self.v_dim = cfg.TRAIN.VISION.V_DIM 

    def tokenize(self, max_length):
        """Tokenizes the questions.
        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry["question"], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            entry["q_token"] = tokens

    def tensorize(self):
        # if self.cfg.TRAIN.VISION.MAML:
        #     self.maml_images_data = {_i: torch.from_numpy(_img).type("torch.FloatTensor") for _i, _img in self.maml_images_data.items()}
        # if self.cfg.TRAIN.VISION.AUTOENCODER:
        #     self.ae_images_data = {_i: torch.from_numpy(_img).type("torch.FloatTensor") for _i, _img in self.ae_images_data.items()}
        # if self.cfg.TRAIN.VISION.CLIP:
        #     self.clip_images_data = {_i: torch.from_numpy(_img).type("torch.FloatTensor") for _i, _img in self.clip_images_data.items()}

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry["q_token"]))
            entry["q_token"] = question
            answer = entry["answer"]
            if len(answer["labels"]) > 0:
                entry["answer"]["labels"] = torch.from_numpy(answer["labels"])
            else:
                entry["answer"]["labels"] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        question = entry["q_token"]
        answer = entry["answer"]
        answer_type = entry["answer_type"]

        image_data = [0, 0, 0]
        if self.cfg.TRAIN.VISION.MAML:
            reshape_size = self.img_resolution["maml"]
            img = Image.open(entry["image_path"]).convert("L")
            img = img.resize((reshape_size, reshape_size))
            img = np.array(img) / 255
            img = img.reshape((reshape_size, reshape_size, 1))
            img = torch.from_numpy(img).type("torch.FloatTensor").reshape(reshape_size*reshape_size)
            image_data[0] = img
        if self.cfg.TRAIN.VISION.AUTOENCODER:
            reshape_size = self.img_resolution["autoencoder"]
            img = Image.open(entry["image_path"]).convert("L")
            img = img.resize((reshape_size, reshape_size))
            img = np.array(img) / 255
            img = img.reshape((reshape_size, reshape_size, 1))
            img = torch.from_numpy(img).type("torch.FloatTensor").reshape(reshape_size*reshape_size)
            image_data[1] = img
        if self.cfg.TRAIN.VISION.CLIP:
            reshape_size = self.img_resolution["clip"]    
            img = Image.open(entry["image_path"]).convert("RGB")
            img = img.resize((reshape_size, reshape_size))
            img = np.array(img) / 255
            img = img.reshape((reshape_size, reshape_size, 3))
            img = torch.from_numpy(img).type("torch.FloatTensor").reshape(reshape_size*reshape_size*3)
            image_data[2] = img

        # if None != answer:
        if answer_type == 'CLOSED':
            answer_target = 0
        else :
            answer_target = 1

        labels = answer["labels"]
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, 1.0)
        outputs = {"image": image_data, "question_logit": question, "target": target, "answer_type": answer_type, "answer_target": answer_target}

        if self.phase in ["valid", "test"]:
            outputs["semantic_type"] = entry["semantic_type"]
            outputs["content_type"] = entry["content_type"]
            outputs["answer_text"] = entry["answer"]["text"]
            outputs["question_text"] = entry["question"]
            outputs["image_name"] = entry["image_name"]

        return outputs

    def __len__(self):
        return len(self.entries)
