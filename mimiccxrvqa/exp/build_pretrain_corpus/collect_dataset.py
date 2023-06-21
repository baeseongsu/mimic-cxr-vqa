import os
import sys
import cv2
import json
import argparse
import numpy as np
import pandas as pd

from multiprocessing import Pool
from transformers import AutoTokenizer


def write_jsonl(data, path):
    with open(path, "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")


def load_mimic_cxr_meta(mimic_cxr_db_dir):
    
    # load raw data
    cxr_meta = pd.read_csv(os.path.join(mimic_cxr_db_dir, "mimic-cxr-2.0.0-metadata.csv"))

    # build a new column: StudyDateTime
    cxr_meta["StudyDateTime"] = pd.to_datetime(
        cxr_meta.StudyDate.astype(str).apply(lambda x: f"{x[:4]}-{x[4:6]}-{x[6:]}") + " " + \
        cxr_meta.StudyTime.apply(lambda x: "%010.3f" % x)
    )
    
    # build a new column: StudyOrder
    cxr_meta_ = cxr_meta.copy()
    cxr_meta_ = cxr_meta_.sort_values(by=["subject_id", "study_id", "StudyDateTime"])
    cxr_meta_ = cxr_meta_.drop_duplicates(subset=["subject_id", "study_id"], keep="first").copy()
    cxr_meta_["StudyDateTime_study_id"] = cxr_meta_['StudyDateTime'].astype(str) + cxr_meta_['study_id'].astype(str)
    cxr_meta_["StudyDateTime_study_id"] = pd.to_datetime(cxr_meta_['StudyDateTime_study_id'])
    cxr_meta_["StudyOrder"] = cxr_meta_.groupby(["subject_id"])["StudyDateTime_study_id"].rank(method="dense")
    cxr_meta["StudyOrder"] = cxr_meta['study_id'].map(cxr_meta_[["study_id", "StudyOrder"]].set_index('study_id')['StudyOrder'])

    # Assumption: Use only frontal images
    cxr_meta = cxr_meta[cxr_meta["ViewPosition"].isin(["AP", "PA"])].reset_index()

    # Assumption: Given the same study_id, use only one image (studydatetime-first + dicom_id-first)
    cxr_meta = cxr_meta.sort_values(['study_id', 'StudyDateTime', 'dicom_id'], ascending=[True, True, True])
    cxr_meta = cxr_meta[
        cxr_meta['dicom_id'].isin(cxr_meta.groupby(['study_id']).dicom_id.first().values)
    ]
    assert cxr_meta.groupby(['study_id', 'StudyDateTime']).dicom_id.nunique().value_counts().size == 1

    return cxr_meta


def preprocess_sectioned(sectioned):
    # build study ids
    sectioned["study_id"] = sectioned.study.str.replace("s", "").astype(int)
    # filter rows with nan in both sections (impression and findings)
    sectioned = sectioned[~(sectioned.findings.isna() & sectioned.impression.isna())]
    sectioned = sectioned.reset_index(drop=True)
    # remove empty impression or findings
    sectioned["findings"] = sectioned.findings.fillna("")
    sectioned["impression"] = sectioned.impression.fillna("")
    # clean impression and findings
    sectioned["findings"] = sectioned.findings.str.replace(r"[\t\n\r]", " ", regex=True)
    sectioned["impression"] = sectioned.impression.str.replace(r"[\t\n\r]", " ", regex=True)
    return sectioned


def check_valid_image(row):

    row = row[1]

    # load an image
    pid, sid, iid = str(row.subject_id), str(row.study_id), str(row.dicom_id)
    image_path = os.path.join(args.mimic_cxr_image_dir, f"p{pid[:2]}/p{pid}/s{sid}/{iid}.jpg")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # crop an image
    ret, thresh = cv2.threshold(src=img, thresh=0, maxval=255, type=0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]
    x, y, w, h = cv2.boundingRect(cnt)

    if w / h >= 0.8 and w / h <= 1.2:
        return True
    else:
        return False


def _preprocess_reports(dataframe, sectioned):

    instances = []

    for row in dataframe.iterrows():

        flag = True

        if not check_valid_image(row):
            flag = False

        row = row[1]
        pid, sid, iid = str(row.subject_id), str(row.study_id), str(row.dicom_id)
        image_path = os.path.join(args.resize_img_dir, f"p{pid[:2]}/p{pid}/s{sid}/{iid}.jpg")

        sectioned_part = sectioned[sectioned.study_id == int(sid)]

        if len(sectioned_part) == 0:
            flag = False
        else:
            # base information
            findings = sectioned_part.findings.values[0]
            impression = sectioned_part.impression.values[0]
            len_findings = sectioned_part.len_findings.values[0]
            len_impression = sectioned_part.len_impression.values[0]

            if len_impression + len_findings == 0:
                flag = False
            elif len_impression + len_findings <= 253:
                text = findings + " " + impression
            else:
                if len_findings <= 253 and len_findings > len_impression:
                    text = findings
                elif len_impression <= 253 and len_impression >= len_findings:
                    text = impression
                else:
                    flag = False

        if flag:
            instance = {
                "id": f"s{sid}",
                "split": "train",
                "text": text,
                "img": image_path,
            }
        else:
            instance = {
                "id": f"s{sid}",
                "split": "",
                "text": "",
                "img": "",
            }

        instances.append(instance)

    instances = pd.DataFrame(instances)

    return instances


def main(args):

    # load dataset
    cxr_meta = load_mimic_cxr_meta(args.mimic_cxr_db_dir)

    # load report sectioned
    sectioned = pd.read_csv(os.path.join(args.mimic_cxr_db_dir, "mimic-cxr-sections/mimic_cxr_sectioned.csv"))
    sectioned = preprocess_sectioned(sectioned)

    # load chexpert
    chexpert = pd.read_csv(os.path.join(args.mimic_cxr_db_dir, "mimic-cxr-2.0.0-chexpert.csv"))

    if args.debug:
        cxr_meta = cxr_meta[:1000]
        sectioned = sectioned[:1000]

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    # apply tokenizer
    sectioned["len_impression"] = sectioned.impression.apply(lambda x: len(tokenizer.tokenize(x)))
    sectioned["len_findings"] = sectioned.findings.apply(lambda x: len(tokenizer.tokenize(x)))
    print("tokenizer applied...")

    import parmap
    import multiprocessing as mp
    from functools import partial

    # parallelize the process
    non_parallel_args = {"sectioned": sectioned}
    parallel_function = partial(_preprocess_reports, **non_parallel_args)

    # split data into multiple parts
    num_cores = int(mp.cpu_count() / 2)
    splitted_data = np.array_split(cxr_meta, num_cores)

    # run multiprocessing
    with mp.Pool(num_cores) as pool:
        results = parmap.map(parallel_function, splitted_data, pm_pbar=True)
        if isinstance(results, list):
            results = pd.concat(results)
        else:
            raise ValueError()

    # # save results
    # results.to_csv("results.csv", index=False)

    # filter out invalid data
    results = results[~results.text.isna()]
    results = results[results.text != ""]
    results = results[~results.img.isna()]
    results = results[results.img != ""]

    # make test dataset
    CHEST_IMAGENOME_DIR = "../../../physionet.org/files/chest-imagenome/1.0.0"
    gold_dataset = pd.read_csv(os.path.join(CHEST_IMAGENOME_DIR, "gold_dataset/gold_attributes_relations_500pts_500studies1st.txt"), sep="\t")
    gold_pids = gold_dataset["patient_id"].unique()
    gold_sids = cxr_meta[cxr_meta['subject_id'].isin(gold_pids)]["study_id"]  # 2502 studies
    test_studies = [f"s{sid}" for sid in gold_sids]
    results_test = results[results.id.isin(test_studies)]
    results_test["split"] = "test"
    results_test_json = results_test.to_dict("records")
    write_jsonl(results_test_json, os.path.join(args.save_jsonl_dir, "pretrain_test.jsonl"))
    print("pretrain_test.jsonl saved...", len(results_test_json))
    
    # [sig] first, remove test set (= gold patients)
    results = results[~results.id.isin(test_studies)]

    # make valid dataset
    os.makedirs(args.save_jsonl_dir, exist_ok=True)
    mimic_cxr_split = pd.read_csv(os.path.join(args.mimic_cxr_db_dir, "mimic-cxr-2.0.0-split.csv"))
    valid_sids = mimic_cxr_split[mimic_cxr_split.split == "validate"].study_id.unique()
    valid_studies = [f"s{sid}" for sid in valid_sids]
    results_valid = results[results.id.isin(valid_studies)]
    results_valid["split"] = "valid"
    results_valid_json = results_valid.to_dict("records")
    write_jsonl(results_valid_json, os.path.join(args.save_jsonl_dir, "pretrain_valid.jsonl"))
    print("pretrain_valid.jsonl saved...", len(results_valid_json))

    # make train dataset
    results_train = results[~results.id.isin(valid_studies)]
    results_train_json = results_train.to_dict("records")
    write_jsonl(results_train_json, os.path.join(args.save_jsonl_dir, "pretrain_train.jsonl"))
    print("pretrain_train.jsonl saved...", len(results_train_json))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # debug
    parser.add_argument("--debug", action="store_true")

    # directory
    parser.add_argument("--mimic_cxr_db_dir", type=str, default="../../../physionet.org/files/mimic-cxr-jpg/2.0.0")
    parser.add_argument("--mimic_cxr_report_dir", type=str, default="../../../physionet.org/files/mimic-cxr-jpg/2.0.0/files")
    parser.add_argument("--mimic_cxr_image_dir", type=str, default="../../../physionet.org/files/mimic-cxr-jpg/2.0.0/files")
    parser.add_argument("--resize_img_dir", type=str, default="../../../physionet.org/files/mimic-cxr-jpg/2.0.0/re224_3ch_contour_cropped")
    parser.add_argument("--save_jsonl_dir", type=str, default="../../dataset/pretrained_corpus")

    # image

    # text
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--max_length", type=int, default=253)

    args = parser.parse_args()

    main(args)
