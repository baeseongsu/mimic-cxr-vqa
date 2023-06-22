import os
import sys
import json
from collections import Counter, defaultdict
from tqdm import tqdm
import pandas as pd
import pyarrow as pa
from sklearn.preprocessing import MultiLabelBinarizer

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from prepro.glossary import normalize_word


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


def make_arrow(data, dataset_name, save_dir, is_fast=True):
    print(f"+ Pre-processing {dataset_name}...")
    iid2captions = defaultdict(list)
    iid2split = dict()

    for split, split_data in tqdm(data.items()):
        for sample in split_data:
            iid2captions[sample["img_path"]].extend(sample["texts"])
            iid2split[sample["img_path"]] = split

    caption_paths = [path for path in tqdm(iid2captions)] if is_fast else [path for path in tqdm(iid2captions) if os.path.exists(path)]
    print(f"+ {len(caption_paths)} images / {len(iid2captions)} annotations")
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


def get_score(occurences):
    return 1.0


def path2rest_vqa(dataset_name, path, split, annotations, label2ans):
    with open(path, "rb") as fp:
        binary = fp.read()

    iid = path
    _annotation = annotations[split][iid]
    _annotation = list(_annotation.items())
    qids, qas = [a[0] for a in _annotation], [a[1] for a in _annotation]

    if "mimiccxrvqa" in dataset_name:
        questions = [qa[0] for qa in qas]
        answers = [qa[1] for qa in qas]
        answer_labels = [a["labels"] for a in answers]
        answer_scores = [a["scores"] for a in answers]
        answer_types = [a["answer_type"] for a in answers]
        content_types = [a["content_type"] for a in answers]
        semantic_types = [a["semantic_type"] for a in answers]
        answers = [a["answers"] for a in answers]
        return [binary, questions, answers, answer_labels, answer_scores, iid, qids, answer_types, content_types, semantic_types, split]

    elif "vqa_rad" in dataset_name or "slake" in dataset_name:
        questions = [qa[0] for qa in qas]
        answers = [qa[1] for qa in qas]
        answer_labels = [a["labels"] for a in answers]
        answer_scores = [a["scores"] for a in answers]
        question_types = [a["answer_type"] for a in answers]  # NOTE: answer_type will be used as question_type
        answers = [[label2ans[l] for l in al] for al in answer_labels]
        return [binary, questions, answers, answer_labels, answer_scores, iid, qids, question_types, split]

    else:
        raise ValueError("Unknown dataset name: {}".format(dataset_name))


def make_arrow_vqa(data, dataset_name, save_dir, is_fast=True):
    questions_train, questions_val, questions_test = data["train"], data["val"], data["test"]

    """
    Record Questions
    """
    annotations = dict()
    for split, questions in zip(["train", "val", "test"], [questions_train, questions_val, questions_test]):
        _annotation = defaultdict(dict)
        for q in tqdm(questions):
            _annotation[q["img_path"]][q["qid"]] = [q["question"]]
        annotations[split] = _annotation

    """
    Construct Vocabulary
    """
    if "mimiccxrvqa" in dataset_name:
        # ans2label = json.load(open(os.path.join(data, "ans2idx.json"), "r"))
        all_major_answers = list()
        for split, questions in zip(["train", "val", "test"], [questions_train, questions_val, questions_test]):
            for q in tqdm(questions):
                # NOTE: MIMIC-CXR-VQA is a multi-label classification task
                answer_list = q["answer"]
                for answer in answer_list:
                    all_major_answers.append(str(answer).lower())
        counter = {k: v for k, v in Counter(all_major_answers).items() if v >= 0}
        assert len(counter) == 110
    elif "vqa_rad" in dataset_name or "slake" in dataset_name:
        all_major_answers = list()
        for split, questions in zip(["train", "val", "test"], [questions_train, questions_val, questions_test]):
            for q in tqdm(questions):
                all_major_answers.append(str(q["answer"]).lower())
        all_major_answers = [normalize_word(word) for word in tqdm(all_major_answers)]
        counter = {k: v for k, v in Counter(all_major_answers).items() if v >= 0}
    else:
        raise ValueError("Unknown dataset: {}".format(dataset_name))

    ans2label = {k: i for i, k in enumerate(sorted(counter.keys()))}
    label2ans = sorted(list(counter.keys()))
    print("Label size ({}): {}.".format(dataset_name, len(ans2label)))

    """
    Record Answers
    """
    if "mimiccxrvqa" in dataset_name:
        # NOTE: MIMIC-CXR-VQA is a multi-label classification task
        mlb = MultiLabelBinarizer()
        mlb.fit([label2ans])

        for split, questions in zip(["train", "val", "test"], [questions_train, questions_val, questions_test]):
            _annotation = annotations[split]
            for q in tqdm(questions):
                answers = q["answer"]
                labels = [mlb.transform([answers]).squeeze()]
                scores = [1.0]
                assert q["answer_type"].strip().lower() == "closed" or q["answer_type"].strip().lower() == "open"
                answer_type = 0 if q["answer_type"].strip().lower() == "closed" else 1
                content_type = q["content_type"]
                answer_type = q["answer_type"]
                semantic_type = q["semantic_type"]
                _annotation[q["img_path"]][q["qid"]].append(
                    {
                        "labels": labels,
                        "scores": scores,
                        "answer_type": answer_type,
                        "content_type": content_type,
                        "semantic_type": semantic_type,
                    }
                )

    elif "vqa_rad" in dataset_name or "slake" in dataset_name:
        for split, questions in zip(["train", "val", "test"], [questions_train, questions_val, questions_test]):
            _annotation = annotations[split]
            for q in tqdm(questions):
                answers = normalize_word(str(q["answer"]).lower())
                answer_count = {}
                answer_count[answers] = answer_count.get(answers, 0) + 1
                labels = []
                scores = []
                for answer in answer_count:
                    assert answer in ans2label
                    labels.append(ans2label[answer])
                    score = get_score(answer_count[answer])
                    scores.append(score)
                assert q["answer_type"].strip().lower() == "closed" or q["answer_type"].strip().lower() == "open"
                answer_type = 0 if q["answer_type"].strip().lower() == "closed" else 1
                _annotation[q["img_path"]][q["qid"]].append(
                    {
                        "labels": labels,
                        "scores": scores,
                        "answer_type": answer_type,
                    }
                )

    else:
        raise ValueError("Unknown dataset: {}".format(dataset_name))

    """
    Write to the files
    """
    for split in ["train", "val", "test"]:
        annot = annotations[split]
        annot_paths = [path for path in annot] if is_fast else [path for path in annot if os.path.exists(path)]
        assert len(annot_paths) == len(annot) or len(annot_paths) == len(annot) - 1
        print("{} set: {} images, {} questions".format(split, len(annot), len([vv for k, v in annot.items() for kk, vv in v.items()])))

        bs = [path2rest_vqa(dataset_name, path, split, annotations, label2ans) for path in tqdm(annot_paths)]
        if "mimiccxrvqa" in dataset_name:
            # [binary, questions, answers, answer_labels, answer_scores, iid, qids, answer_types, content_types, split]
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
                    "answer_types",
                    "content_types",
                    "semantic_types",
                    "split",
                ],
            )
        elif "vqa_rad" in dataset_name or "slake" in dataset_name:
            # [binary, questions, answers, answer_labels, answer_scores, iid, qids, question_types, split]
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
                    "question_types",
                    "split",
                ],
            )
        else:
            raise ValueError("Unknown dataset: {}".format(dataset_name))

        table = pa.Table.from_pandas(dataframe)

        os.makedirs(save_dir, exist_ok=True)
        with pa.OSFile(f"{save_dir}/{dataset_name}_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
