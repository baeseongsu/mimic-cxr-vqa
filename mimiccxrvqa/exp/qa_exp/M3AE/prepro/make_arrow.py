import os
import sys
import json
from collections import Counter, defaultdict

import pandas as pd
import pyarrow as pa

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tqdm import tqdm
from prepro.glossary import normalize_word
from sklearn.preprocessing import MultiLabelBinarizer


def statistics(iid2captions, iid2split):
    all_images = {"train": [], "val": [], "test": []}
    all_texts = {"train": [], "val": [], "test": []}

    for iid, texts in iid2captions.items():
        split = iid2split[iid]
        all_images[split].append(iid)
        all_texts[split].extend(texts)

    for split, images in all_images.items():
        print(f"+ {split} set: {len(images)} images")

    for split, texts in all_texts.items():
        lengths = [len(text.split()) for text in texts]
        avg_len = sum(lengths) / len(lengths)
        print(f"+ {split} set: {avg_len} words in average.")
        lengths = [length // 10 * 10 for length in lengths]
        print(Counter(lengths))


def path2rest(path, iid2captions, iid2split):
    name = path
    with open(path, "rb") as fp:
        binary = fp.read()
    captions = iid2captions[name]
    split = iid2split[name]
    return [binary, captions, name, split]


def make_arrow(data, dataset_name, save_dir):
    print(f"+ Pre-processing {dataset_name}...")
    iid2captions = defaultdict(list)
    iid2split = dict()

    for split, split_data in data.items():
        for sample in split_data:
            iid2captions[sample["img_path"]].extend(sample["texts"])
            iid2split[sample["img_path"]] = split

    path = len(iid2captions)
    caption_paths = [path for path in iid2captions if os.path.exists(path)]
    print(f"+ {len(caption_paths)} images / {path} annotations")
    statistics(iid2captions, iid2split)
    bs = [path2rest(path, iid2captions, iid2split) for path in tqdm(caption_paths)]

    for split in ["train", "val", "test"]:
        batches = [b for b in bs if b[-1] == split]
        dataframe = pd.DataFrame(batches, columns=["image", "caption", "image_id", "split"])
        table = pa.Table.from_pandas(dataframe)
        os.makedirs(save_dir, exist_ok=True)
        with pa.OSFile(f"{save_dir}/{dataset_name}_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)


def path2rest_mimic_cxr(path, iid2captions, iid2chexpert, iid2split):
    name = path
    with open(path, "rb") as fp:
        binary = fp.read()
    captions = iid2captions[name]
    chexpert = iid2chexpert[name]
    split = iid2split[name]
    return [binary, captions, name, chexpert, split]


def make_arrow_mimic_cxr(data, dataset_name, save_dir):
    print(f"+ Pre-processing {dataset_name}...")
    iid2captions = defaultdict(list)
    iid2chexpert = defaultdict(list)
    iid2split = dict()

    for split, split_data in data.items():
        for sample in split_data:
            iid2captions[sample["img_path"]].extend(sample["texts"])
            iid2chexpert[sample["img_path"]].extend(sample["chexpert"])
            iid2split[sample["img_path"]] = split

    path = len(iid2captions)
    caption_paths = [path for path in iid2captions if os.path.exists(path)]
    print(f"+ {len(caption_paths)} images / {path} annotations")
    statistics(iid2captions, iid2split)
    bs = [path2rest_mimic_cxr(path, iid2captions, iid2chexpert, iid2split) for path in tqdm(caption_paths)]

    for split in ["train", "val", "test"]:
        batches = [b for b in bs if b[-1] == split]
        dataframe = pd.DataFrame(batches, columns=["image", "caption", "image_id", "chexpert", "split"])
        table = pa.Table.from_pandas(dataframe)
        os.makedirs(save_dir, exist_ok=True)
        with pa.OSFile(f"{save_dir}/{dataset_name}_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)


def get_score(occurences):
    return 1.0


def path2rest_mmehr(path, split, annotations, label2ans):
    with open(path, "rb") as fp:
        binary = fp.read()

    iid = path
    _annotation = annotations[iid]
    _annotation = list(_annotation.items())
    qids, qas = [a[0] for a in _annotation], [a[1] for a in _annotation]
    questions = [qa[0] for qa in qas]
    answers = [qa[1] for qa in qas]
    answer_labels = [a["labels"] for a in answers]
    answer_scores = [a["scores"] for a in answers]
    answer_types = [a["answer_type"] for a in answers]
    question_types = [a["question_type"] for a in answers]
    answers = [a["answers"] for a in answers]

    return [binary, questions, answers, answer_labels, answer_scores, iid, qids, answer_types, question_types, split]


def make_arrow_mmehr(questions, split, dataset_name, save_dir, save=True):
    # Record Questions
    _annotation = defaultdict(dict)
    for q in tqdm(questions):
        _annotation[q["img_path"]][q["qid"]] = [q["question"]]
        
    # Construct Vocabulary
    # NOTE: dir is hard-coded
    MMEHR_DATA_ROOT = "../../../dataset" 
    ans2label = json.load(open(os.path.join(MMEHR_DATA_ROOT, "ans2idx.json"), "r"))
    label2ans = list(ans2label.keys())
    print("Label size ({}): {}.".format(dataset_name, len(ans2label)))

    # Record Answers
    mlb = MultiLabelBinarizer()
    mlb.fit([sorted([k for k in ans2label.keys()])])

    for q in tqdm(questions):
        answers = q["answer"]
        labels = [mlb.transform([answers]).squeeze()]
        scores = [1.0]
        question_type = q["question_type"]
        assert q['answer_type'].strip().lower() == "closed"
        _annotation[q["img_path"]][q["qid"]].append(
            {"labels": labels, "scores": scores, "answer_type": 0, 'answers': answers, "question_type": question_type})
    

    # Write to the files
    annot = _annotation
    annot_paths = [path for path in annot if os.path.exists(path)]
    assert len(annot_paths) == len(annot) or len(annot_paths) == len(annot) - 1
    print("{} set: {} images, {} questions".format(split,
                                                    len(annot),
                                                    len([vv for k, v in annot.items() for kk, vv in v.items()])))
    bs = [
        path2rest_mmehr(path, split, _annotation, label2ans) for path in tqdm(annot_paths)
    ]
    dataframe = pd.DataFrame(
        bs,
        columns=[
            "image",
            "questions",
            "answers",
            "answer_labels",
            "answer_scores",
            "image_id",
            "question_id",
            "answer_type",
            "question_type",
            "split",
        ],
    )
    
    table = pa.Table.from_pandas(dataframe)
    
    if save:
        os.makedirs(save_dir, exist_ok=True)
        with pa.OSFile(f"{save_dir}/{dataset_name}_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)

    return table
