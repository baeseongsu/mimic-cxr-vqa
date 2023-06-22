import os
import re
import sys
import json
import random
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from tqdm import tqdm
from make_arrow import make_arrow_mmehr


def prepro_mmehr():
    random.seed(42)

    data = {
        "train":[],
        "val":[],
        "test":[],
    }

    # NOTE: dirs are hard-coded
    DATA_ROOT = "../../../dataset" 
    IMAGE_ROOT = "../../../physionet.org/files/mimic-cxr-jpg/2.0.0/re512_3ch_contour_cropped"
        
    for split, file in zip(
        ["train", "val", "test"],
        ["train", "valid", "test"]):
        print(f"Start preprocess {split} dataset")
        samples = json.load(open(f"{DATA_ROOT}/{file}.json"))
        samples = pd.DataFrame(samples)

        q2id = {k: i for i, k in enumerate(set(samples.question))}
        samples["qid"] = samples["question"].apply(lambda x: q2id[x])

        for sample_idx, sample in tqdm(samples.iterrows()):
            img_path = os.path.join(IMAGE_ROOT, sample["image_path"])
            question = sample["question"]
            qid = sample["qid"]
            answer = sample["answer"]
            answer_type = "CLOSED"  
            question_type = sample["content_type"] 
            data[split].append({
                "img_path": img_path,
                "qid": qid,
                "question": question,
                "answer": answer,
                "answer_type": answer_type,
                "question_type" : question_type,
            })
    
    for split, questions in data.items():
        print(f"Make arrow file for {split} data")
        _ = make_arrow_mmehr(questions, split, "vqa_mmehr", "./data/finetune_arrows/")
    

if __name__ == '__main__':
    prepro_mmehr()
