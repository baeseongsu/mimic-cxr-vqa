import os
import re
import sys
import json
import random
import pandas as pd

from tqdm import tqdm
from make_arrow import make_arrow_mmehr


def prepro_vqa_mimiccxrvqa(args):
    data_root = args.data_root
    image_root = args.image_root
    save_root = args.save_root 

    random.seed(42)

    data = {"train": [], "val": [], "test": []}

    for split, file in zip(["train", "val", "test"], ["train.json", "valid.json", "test.json"]):
        print(f"Start preprocess {split} dataset")
        samples = json.load(open(os.path.join(data_root, file), "r"))
        samples = pd.DataFrame(samples)

        question2qid = {k: i for i, k in enumerate(samples.question.unique())}
        samples["qid"] = samples["question"].apply(lambda x: question2qid[x])

        for _, sample in tqdm(samples.iterrows(), total=len(samples)):
            img_path = os.path.join(image_root, sample["image_path"])
            question = sample["question"]
            qid = sample["qid"]
            answer = sample["answer"]
            answer_type = "CLOSED" if sample["semantic_type"].lower() in ["verify", "choose"] else "OPEN"
            content_type = sample["content_type"]
            semantic_type = sample["semantic_type"]
            data[split].append(
                {
                    "img_path": img_path,
                    "qid": qid,
                    "question": question,
                    "answer": answer,
                    "answer_type": answer_type,
                    "content_type": content_type,  # only exist in mimiccxrvqa
                    "semantic_type": semantic_type,  # only exist in mimiccxrvqa
                }
            )

    for split, questions in data.items():
        print(f"Make arrow file for {split} data")
        _ = make_arrow_mmehr(questions, split, "vqa_mmehr", save_root)
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--save_root", type=str, required=True)
    args = parser.parse_args()
    prepro_vqa_mimiccxrvqa(args)

# def prepro_vqa_vqa_rad(
#     data_root="data/finetune_data/vqa_rad",
#     image_root=f"data/finetune_data/vqa_rad/images",
# ):
#     # data_root = "data/finetune_data/vqa_rad/"
#     # image_root = f"{data_root}/images"

#     random.seed(42)

#     data = {"train": [], "val": [], "test": []}

#     for split, file in zip(["train", "val", "test"], ["trainset.json", "valset.json", "testset.json"]):
#         with open(f"{data_root}/{file}", "r") as fp:
#             samples = json.load(fp)
#             for sample in samples:
#                 img_path = os.path.join(image_root, sample["image_name"])
#                 qid = sample["qid"]
#                 question = sample["question"]
#                 answer = sample["answer"]
#                 answer_type = sample["answer_type"]
#                 data[split].append(
#                     {
#                         "img_path": img_path,
#                         "qid": qid,
#                         "question": question,
#                         "answer": answer,
#                         "answer_type": answer_type,
#                     }
#                 )
#     make_arrow_vqa(data, "vqa_vqa_rad", "data/finetune_arrows/")


# def prepro_vqa_slake(
#     data_root="data/finetune_data/slake",
#     image_root="data/finetune_data/slake/imgs",
# ):
#     # data_root = "data/finetune_data/slake/"
#     # image_root = f"{data_root}/imgs"

#     random.seed(42)

#     data = {"train": [], "val": [], "test": []}

#     for split, file in zip(["train", "val", "test"], ["train.json", "validate.json", "test.json"]):
#         with open(f"{data_root}/{file}", "r") as fp:
#             samples = json.load(fp)
#             for sample in samples:
#                 if sample["q_lang"] != "en":
#                     continue
#                 img_path = os.path.join(image_root, sample["img_name"])
#                 qid = sample["qid"]
#                 question = sample["question"]
#                 answer = sample["answer"]
#                 answer_type = sample["answer_type"]
#                 data[split].append(
#                     {
#                         "img_path": img_path,
#                         "qid": qid,
#                         "question": question,
#                         "answer": answer,
#                         "answer_type": answer_type,
#                     },
#                 )
#     make_arrow_vqa(data, "vqa_slake", "data/finetune_arrows/")

