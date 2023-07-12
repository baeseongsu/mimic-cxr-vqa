#!/usr/bin/env python


import argparse
import pandas as pd
import os
import json
import numpy as np
import pickle
from tqdm import tqdm


def create_img2idx(root, train_csv, val_csv, test_csv, out_json_path):
    train = pd.DataFrame(json.load(open(os.path.join(root, train_csv))))
    valid = pd.DataFrame(json.load(open(os.path.join(root, val_csv))))
    test = pd.DataFrame(json.load(open(os.path.join(root, test_csv))))
    img2idx = {}

    df = train.append(valid)
    df = df.append(test)

    df_imgs = df["image_path"].unique().tolist()
    for i, row in tqdm(df.iterrows()):
        img_name = row["image_path"]
        img_id = df_imgs.index(img_name)  # starts from 0
        if img_name not in img2idx:
            img2idx[img_name] = img_id
        else:
            assert img2idx[img_name] == img_id

    with open(out_json_path, "w") as f:
        json.dump(img2idx, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create img2idx.json.")
    parser.add_argument("--input_path", type=str, help="Path to csv files")
    parser.add_argument("--trainfile", type=str, default="train_qa_dataset_all.csv", help="train.json")
    parser.add_argument("--validfile", type=str, default="valid_qa_dataset_all.csv", help="valid.json")
    parser.add_argument("--testfile", type=str, default="test_qa_dataset_all.csv", help="test.json")
    parser.add_argument("--out_path", type=str, help="Path to output file")
    args = parser.parse_args()
    create_img2idx(args.input_path, args.trainfile, args.validfile, args.testfile, args.out_path)
