# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         process_dataset
# Description:  convert original .txt file to train.json and validate.json
# Author:       Boliu.Kelvin, Sedigheh Eslami
# Date:         2020/4/5
# -------------------------------------------------------------------------------

import argparse
import pandas as pd
import os
import sys
import json
import numpy as np
import re
import _pickle as cPickle


def filter_answers(train_qa_pairs, val_qa_pairs, min_occurence):
    """This will change the answer to preprocessed version"""
    occurence = {}
    qa_pairs = train_qa_pairs.append(val_qa_pairs)
    # qa_pairs["answer"] = qa_pairs["answer"].apply(lambda x: str(x).lower())

    for id, row in qa_pairs.iterrows():  # row:[id,ques,ans]
        gtruth = row["answer"] if row["answer"] != "nan" else ""
        # gtruth = gtruth.split('|') if len(gtruth) > 0 else []
        for _gtruth in gtruth:
            if _gtruth not in occurence and _gtruth is not None:
                occurence[_gtruth] = set()
            occurence[_gtruth].add(row["question"])
    # import pdb;pdb.set_trace()
    for answer in list(occurence):
        if len(occurence[answer]) < min_occurence:
            occurence.pop(answer)

    print("Num of answers that appear >= %d times: %d" % (min_occurence, len(occurence)))
    return occurence


def create_ans2label(occurence, filename="trainval", root="data"):
    """Note that this will also create label2ans.pkl at the same time

    occurence: dict {answer -> whatever}
    name: prefix of the output file
    cache_root: str
    """
    df = train_qa_pairs.append(val_qa_pairs)
    df = df.append(test_qa_pairs)
    # df["answer"] = df["answer"].apply(lambda x: str(x).lower())

    ans2label = {}
    label2ans = []
    label = 0
    for answer in occurence:
        label2ans.append(answer)
        ans2label[answer] = label
        label += 1

    print("ans2lab", len(ans2label))
    print("lab2abs", len(label2ans))

    if not os.path.exists(os.path.join(root, "cache")):
        os.mkdir(os.path.join(root, "cache"))

    file = os.path.join(root, "cache", "ans2label_multilabel.pkl")
    cPickle.dump(ans2label, open(file, "wb"))
    file = os.path.join(root, "cache", "label2ans_multilabel.pkl")
    cPickle.dump(label2ans, open(file, "wb"))

    return ans2label


def compute_target(answers_dset, ans2label, name, image_id_col="image_id", root="data"):
    """Augment answers_dset with soft score as label

    ***answers_dset should be preprocessed***

    Write result into a cache file
    """
    target = []
    # answers_dset["answer"] = answers_dset["answer"].apply(lambda x: str(x).lower())
    for id, qa_pair in answers_dset.iterrows():
        answers = qa_pair["answer"]  # .split('|') if qa_pair["answer"] != 'nan' else []
        labels = []
        for _ans in answers:
            if _ans in ans2label:
                labels.append(ans2label[_ans])
            else:
                import pdb

                pdb.set_trace()
                print(f"{_ans} not exist in answer set")
                raise NotImplementedError

        target.append({"labels": labels, "image_path": qa_pair["image_path"], "question": qa_pair["question"]})

    file = os.path.join(root, "cache", name + "_target.pkl")
    cPickle.dump(target, open(file, "wb"))
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Med VQA")
    parser.add_argument("--input_path", type=str, help="Path to input data")
    parser.add_argument("--output_path", type=str, help="Path to output data")
    parser.add_argument("--trainfile", type=str, help="Name of the train file", default="train.json")
    parser.add_argument("--validfile", type=str, help="Name of the valid file", default="valid.json")
    parser.add_argument("--testfile", type=str, help="Name of the test file", default="test.json")
    args = parser.parse_args()
    dataroot = args.input_path
    output_path = args.output_path
    train_file = args.trainfile
    valid_file = args.validfile
    test_file = args.testfile

    train = pd.DataFrame(json.load(open(os.path.join(dataroot, train_file))))
    valid = pd.DataFrame(json.load(open(os.path.join(dataroot, valid_file))))
    test = pd.DataFrame(json.load(open(os.path.join(dataroot, test_file))))

    img_col = "image_id"

    train_qa_pairs = train[[img_col, "question", "answer", "image_path"]]
    val_qa_pairs = valid[[img_col, "question", "answer", "image_path"]]
    test_qa_pairs = test[[img_col, "question", "answer", "image_path"]]

    occurence = filter_answers(train_qa_pairs, val_qa_pairs, 0)  # select the answer with frequence over min_occurence

    label_path = output_path + "/cache/ans2label_multilabel.pkl"
    if os.path.isfile(label_path):
        print("found %s" % label_path)
        total_ans2label = cPickle.load(open(label_path, "rb"))
    else:
        total_ans2label = create_ans2label(occurence, filename="trainval", root=output_path)  # create ans2label and label2ans

    compute_target(train_qa_pairs, total_ans2label, "train_multilabel", img_col, output_path)  # dump train target to .pkl {question,image_name,labels,scores}
    compute_target(val_qa_pairs, total_ans2label, "valid_multilabel", img_col, output_path)  # dump validate target to .pkl {question,image_name,labels,scores}
    compute_target(test_qa_pairs, total_ans2label, "test_multilabel", img_col, output_path)  # dump validate target to .pkl {question,image_name,labels,scores}

    print("Process finished successfully!")
