import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
import logging


def configure_logging():
    logging.basicConfig(level=logging.INFO)


def load_template_pk_to_template():
    return {
        "base_1": "Are there any ${category} in the ${object}?",
        "base_2": "Is there ${attribute} in the ${object}?",
        "base_abn_1": "Is the ${object} abnormal?",
        "base_4": "Are there any ${category_1} or ${category_2} in the ${object}?",
        "base_5": "Are there both ${attribute_1} and ${attribute_2} in the ${object}?",
        "base_6": "Is there either ${attribute_1} or ${attribute_2} in the ${object}?",
        "base_9": "List all ${category} in the ${object}.",
        "base_abn_2": "List all abnormalities in the ${object}.",
        "base_10": "List all ${category_1} and ${category_2} in the ${object}.",
        "base_11": "Which ${category} is related to the ${object}, ${attribute_1} or ${attribute_2}?",
        "base_abn_3": "Are there any abnormalities in either the ${object_1} or the ${object_2}?",
        "base_abn_4": "Are there any abnormalities in both the ${object_1} and the ${object_2}?",
        "base_25": "List all ${category} in either the ${object_1} or the ${object_2}.",
        "base_26": "List all common ${category} in both the ${object_1} and the ${object_2}.",
        "base_27": "List all ${category} only in the ${object_1} but not in the ${object_2}.",
        "base_abn_5": "List all abnormalities in either the ${object_1} or the ${object_2}.",
        "base_abn_6": "List all common abnormalities in both the ${object_1} and the ${object_2}.",
        "base_abn_7": "List all abnormalities only in the ${object_1} but not in the ${object_2}.",
        "base_30": "Are there any ${category}?",
        "base_abn_8": "Are there any abnormalities?",
        "base_32": "Are there any ${category_1} or ${category_2}?",
        "base_33": "Is there ${attribute}?",
        "base_34": "Are there both ${attribute_1} and ${attribute_2}?",
        "base_35": "Is there either ${attribute_1} or ${attribute_2}?",
        "base_40": "List all ${category}.",
        "base_41": "List all ${category_1} and ${category_2}.",
        "base_abn_9": "List all abnormalities.",
        "base_42": "Which ${category} is related, ${attribute_1} or ${attribute_2}?",
        "base_45": "Are both the ${object_1} and the ${object_2} related to ${attribute}?",
        "base_46": "Is either the ${object_1} or the ${object_2} related to ${attribute}?",
        "base_49": "List all anatomical locations related to ${attribute}.",
        "base_50": "Which anatomical location is related to ${attribute}, the ${object_1} or the ${object_2}?",
        "base_abn_10": "Which anatomical location is abnormal, the ${object_1} or the ${object_2}?",
        "base_58": "List all anatomical locations related to either ${attribute_1} or ${attribute_2}.",
        "base_59": "List all common anatomical locations related to both ${attribute_1} and ${attribute_2}.",
        "base_60": "List all anatomical locations related to ${attribute_1} but not ${attribute_2}.",
        "base_64": "Are there any ${category} related to the ${object_1} and the ${object_2}?",
        "base_65": "Are there any ${category} related to the ${object_1} or the ${object_2}?",
        "base_66": "List all anatomical locations related to any ${category}.",
        "base_73": "List all anatomical locations related to any ${category_1} or ${category_2}.",
        "viewpos_1": "Is this an ${viewpos} view?",
        "viewpos_2": "Which view is in this image, AP or PA?",
        "viewpos_3": "What is the view of this image?",
        "gender_1": "Is this patient ${gender}?",
        "gender_2": "What is the gender of this patient, male or female?",
        "gender_3": "What is the gender of this patient?",
        "ct_ratio_1": "Is the width of the cardiac silhouette wider than 1/2 of the thorax width?",
        "mt_ratio_1": "Is the width of the upper mediastinum wider than 1/3 of the thorax width?",
    }


def get_template_pk_to_program_idx(template_pk_to_template):
    return {template_pk: f"program_{i+1}" for i, template_pk in enumerate(template_pk_to_template.keys())}


def split_string(string):
    if pd.isna(string):
        return []
    return string.split("|")


def process_sample(sample, template_pk_to_program_idx, split, sample_idx):
    try:
        category_type = str(sample["category_type"]).replace("position", "anatomy").replace("condition", "presence")
        semantic_type = str(sample["semantic_type"])

        template = str(sample["template"])
        question = str(sample["question"])
        _answer = split_string(sample["answer"])

        _subject_id = int(sample["subject_id"])
        _study_id = int(sample["study_id"])
        image_id = str(sample["image_id"])
        _image_path = str(sample["jpg_fpath"])

        args_object = {idx: item for idx, item in enumerate(split_string(sample["argcomb_object"]))}
        args_attribute = {idx: item for idx, item in enumerate(split_string(sample["argcomb_attribute"]))}
        args_category = {idx: item for idx, item in enumerate(split_string(sample["argcomb_category"]))}

        args_viewpos = {}
        args_gender = {}
        if pd.notnull(sample["argcomb_extra"]):
            if sample["argcomb_extra"] in {"AP", "PA"}:
                args_viewpos = {0: sample["argcomb_extra"]}
            elif sample["argcomb_extra"] in {"male", "female"}:
                args_gender = {0: "M" if sample["argcomb_extra"] == "male" else "F"}
            else:
                logging.warning("Unexpected value in argcomb_extra: %s", sample["argcomb_extra"])

        new_sample = {
            "split": str(split),
            "idx": sample_idx,
            # "_subject_id": _subject_id,
            # "_study_id": _study_id,
            "image_id": image_id,
            # "_image_path": _image_path,
            "question": question,
            # "_answer": _answer,
            "content_type": category_type,
            "semantic_type": semantic_type,
            "template": template,
            "template_program": str(template_pk_to_program_idx[str(sample["template_pk"])]),
            "template_arguments": {
                "object": args_object,
                "attribute": args_attribute,
                "category": args_category,
                "viewpos": args_viewpos,
                "gender": args_gender,
            },
        }

    except KeyError as e:
        logging.error("KeyError: Missing expected field in sample: %s", e)
        return None

    except Exception as e:
        logging.error("Unexpected error while processing sample: %s", e)
        return None

    return new_sample


def convert_qa_dataset_format_from_csv_to_json(csv_file_path, json_file_path, template_pk_to_program_idx, split):
    try:
        dataset = pd.read_csv(csv_file_path).to_dict("records")
    except FileNotFoundError:
        logging.error("CSV file not found: %s", csv_file_path)
        return
    except pd.errors.EmptyDataError:
        logging.error("CSV file is empty: %s", csv_file_path)
        return

    logging.info("len(dataset): %d", len(dataset))
    logging.debug("dataset[0]: %s", dataset[0])

    new_dataset = []
    for sample_idx, sample in enumerate(tqdm(dataset)):
        processed_sample = process_sample(sample, template_pk_to_program_idx, split, sample_idx)
        if processed_sample is not None:
            new_dataset.append(processed_sample)

    if not new_dataset:
        logging.warning("No valid samples processed. Skipping file write.")
        return

    if split == "train":
        # NOTE: split train into train_part1 and train_part2 due to the memory issue
        # save
        new_dataset_1 = new_dataset[: int(len(new_dataset) / 2)]
        new_dataset_2 = new_dataset[int(len(new_dataset) / 2) :]
        with open(json_file_path.replace(".json", "_part1.json"), "w") as f:
            json.dump(new_dataset_1, f, indent=2)
        with open(json_file_path.replace(".json", "_part2.json"), "w") as f:
            json.dump(new_dataset_2, f, indent=2)

        # check
        with open(json_file_path.replace(".json", "_part1.json"), "r") as f:
            new_dataset_1 = json.load(f)
        with open(json_file_path.replace(".json", "_part2.json"), "r") as f:
            new_dataset_2 = json.load(f)
        new_dataset = new_dataset_1 + new_dataset_2
        logging.info("len(new_dataset): %d", len(new_dataset))
        logging.debug("new_dataset[0]: %s", new_dataset[0])

    else:
        # save
        with open(json_file_path, "w") as f:
            json.dump(new_dataset, f, indent=2)

        # check
        with open(json_file_path, "r") as f:
            new_dataset = json.load(f)
        logging.info("len(new_dataset): %d", len(new_dataset))
        logging.debug("new_dataset[0]: %s", new_dataset[0])


def main(args):
    configure_logging()
    template_pk_to_template = load_template_pk_to_template()
    template_pk_to_program_idx = get_template_pk_to_program_idx(template_pk_to_template)
    logging.info("Template PK to Program IDX: %s", template_pk_to_program_idx)

    for split in ["train", "valid", "test"]:
        csv_file_path = os.path.join(args.csv_file_dir, f"{split}_qa_dataset_all_paraphrase.csv")
        json_file_path = os.path.join(args.json_file_dir, "dataset", f"_{split}.json")
        convert_qa_dataset_format_from_csv_to_json(csv_file_path, json_file_path, template_pk_to_program_idx, split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert QA dataset from CSV to JSON format.")
    parser.add_argument("--csv_file_dir", type=str, required=True, help="Directory containing the CSV files.")
    parser.add_argument("--json_file_dir", type=str, required=True, help="Directory to save the JSON files.")
    args = parser.parse_args()
    main(args)
