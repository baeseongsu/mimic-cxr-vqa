import os
import json
import pandas as pd
import logging
from tqdm import tqdm
from program import PROGRAM_LIST

logging.basicConfig(filename="error.log", filemode="w", format="%(name)s - %(levelname)s - %(message)s")

# Constants
CONFIG = {
    "REMOVED_OBJS": ["left arm", "right arm"],
    "REMOVED_ATTRS": [
        "artifact",
        "bronchiectasis",
        "pigtail catheter",
        "skin fold",
        "aortic graft/repair",
        "diaphragmatic eventration (benign)",
        "sternotomy wires",  # ?
    ],
}

DATA_FIELDS = [
    "split",
    "idx",
    "subject_id",
    "study_id",
    "image_id",
    "image_path",
    "question",
    "semantic_type",
    "content_type",
    "template",
    "template_program",
    "template_arguments",
    "answer",
]


def load_metadata(args):
    metadata = pd.read_csv(os.path.join(args.mimic_cxr_jpg_dir, "mimic-cxr-2.0.0-metadata.csv"))
    metadata = metadata[["subject_id", "study_id", "dicom_id", "ViewPosition"]]
    metadata = metadata.rename(columns={"dicom_id": "image_id"})
    return metadata


def load_bbox_information(args):
    if args.split == "test":
        bbox_info = pd.read_csv(
            os.path.join(args.chest_imagenome_dir, "gold_dataset", "gold_bbox_coordinate_annotations_1000images.csv"),
            usecols=["image_id", "bbox_name", "coord224"],
        )
        bbox_info["image_id"] = bbox_info["image_id"].str.replace(".dcm", "")
        bbox_info = bbox_info.rename(columns={"bbox_name": "object"})
    else:
        bbox_info = pd.read_csv(
            os.path.join(args.chest_imagenome_dir, "silver_dataset", "scene_tabular", "bbox_objects_tabular.txt"),
            sep="\t",
            usecols=["object_id", "bbox_name", "x1", "y1", "x2", "y2"],
        )
        assert sum(bbox_info["object_id"].apply(lambda x: x.split("_")[-1]) != bbox_info["bbox_name"]) == 0
        bbox_info["image_id"] = bbox_info["object_id"].apply(lambda x: x.split("_")[0])
        bbox_info["coord224"] = [str([int(x1), int(y1), int(x2), int(y2)]) for x1, y1, x2, y2 in zip(bbox_info["x1"], bbox_info["y1"], bbox_info["x2"], bbox_info["y2"])]
        bbox_info = bbox_info[["image_id", "bbox_name", "coord224"]]
        bbox_info = bbox_info.rename(columns={"bbox_name": "object"})

    bbox_info["x1"] = bbox_info["coord224"].apply(lambda x: int(x[1:-1].split(",")[0]))
    bbox_info["x2"] = bbox_info["coord224"].apply(lambda x: int(x[1:-1].split(",")[2]))

    return bbox_info


def format_answer(answer):
    if answer is True:
        return ["yes"]
    elif answer is False:
        return ["no"]
    elif answer is None:
        return []
    else:
        return sorted([a.lower() for a in answer])


def process_label_dataset(label_dataset):
    # Perform any necessary preprocessing or filtering on label_dataset
    label_dataset = label_dataset[~label_dataset["object"].isin(CONFIG["REMOVED_OBJS"])]
    label_dataset = label_dataset[~label_dataset["attribute"].isin(CONFIG["REMOVED_ATTRS"])]
    label_dataset = label_dataset.reset_index(drop=True)
    return label_dataset


def verify_data_match(data, subject_id, study_id, image_path, answer):
    try:
        assert data["_subject_id"] == subject_id, f"subject_id is not matched: {data['_subject_id']} vs {subject_id}"
        assert data["_study_id"] == study_id, f"study_id is not matched: {data['_study_id']} vs {study_id}"
        assert data["_image_path"] == image_path, f"image_path is not matched: {data['_image_path']} vs {image_path}"
        assert data["_answer"] == answer, f"answer is not matched: {data['_answer']} vs {answer}"
    except AssertionError:
        logging.error(f"Error at image_id {image_path}: {data['_answer']} vs {answer}")


def main(args):
    # Load label_dataset
    label_dataset = pd.read_csv(args.label_dataset_path)
    label_dataset = label_dataset[["image_id", "bbox", "relation", "label_name", "categoryID", "object_id"]]
    label_dataset = label_dataset.rename(columns={"dicom_id": "image_id", "bbox": "object", "label_name": "attribute", "categoryID": "category"})

    # Load MIMIC-CXR-JPG metadata
    metadata = load_metadata(args)
    metadata = metadata.set_index("image_id")

    # Add subject_id/study_id/viewpos information to label_dataset
    label_dataset = label_dataset.merge(metadata, on="image_id", how="left")
    label_dataset = label_dataset.reset_index(drop=True)

    # Load MIMIC-IV patients table
    tb_patients = pd.read_csv(os.path.join(args.mimic_iv_dir, "hosp/patients.csv"))
    tb_patients = tb_patients[["subject_id", "gender"]]

    # Add gender information to label_dataset
    label_dataset = label_dataset.merge(tb_patients, on="subject_id", how="left")

    # Load bounding box information
    bbox_info = load_bbox_information(args)

    # Add bounding box information to label_dataset
    label_dataset = label_dataset.merge(bbox_info, on=["image_id", "object"], how="left")

    # Process label_dataset
    label_dataset = process_label_dataset(label_dataset)
    label_dataset = label_dataset.set_index("image_id")

    # Read json file
    if args.split == "train":
        dataset_part1 = json.load(open(args.json_file.replace(".json", "_part1.json")))
        dataset_part2 = json.load(open(args.json_file.replace(".json", "_part2.json")))
        dataset = dataset_part1 + dataset_part2
    else:
        dataset = json.load(open(args.json_file))

    answer_error_cnt = 0

    # Dictionary key will be image_id
    grouped = label_dataset.groupby("image_id")
    grouped = sorted(grouped, key=lambda x: len(x[1]), reverse=False)
    grouped = {image_id: group for image_id, group in grouped}

    new_dataset = []
    for data in tqdm(dataset):
        # Load image features
        image_id = data["image_id"]
        image_labels = grouped[image_id]

        # Find patient-related identifiers
        subject_id = metadata.loc[image_id]["subject_id"]
        study_id = metadata.loc[image_id]["study_id"]
        image_path = f"p{str(subject_id)[:2]}/p{str(subject_id)}/s{str(study_id)}/{str(image_id)}.jpg"

        # Retrieve answer
        program = PROGRAM_LIST[data["template_program"]]
        args_object = [v for v in data["template_arguments"]["object"].values()]
        args_attribute = [v for v in data["template_arguments"]["attribute"].values()]
        args_category = [v for v in data["template_arguments"]["category"].values()]
        args_viewpos = [v for v in data["template_arguments"]["viewpos"].values()]
        args_gender = [v for v in data["template_arguments"]["gender"].values()]

        # Run program
        if data["template_program"] in ["program_41", "program_42", "program_43"]:
            answer = program(features=image_labels, viewpos=args_viewpos)
        elif data["template_program"] in ["program_44", "program_45", "program_46"]:
            answer = program(features=image_labels, gender=args_gender)
        elif data["template_program"] in ["program_47", "program_48"]:
            answer = program(features=image_labels, object=args_object, category=args_category, attribute=args_attribute)
        else:
            answer = program(features=image_labels, object=args_object, category=args_category, attribute=args_attribute)

        answer = format_answer(answer)

        new_data = {
            "subject_id": subject_id,
            "study_id": study_id,
            "image_path": image_path,
            "answer": answer,
            **data,
        }

        # re-order keys
        new_data = {k: new_data[k] for k in DATA_FIELDS}

        if args.debug:
            # Verify data match
            verify_data_match(data, subject_id, study_id, image_path, answer)
            new_data.pop("_subject_id")
            new_data.pop("_study_id")
            new_data.pop("_image_path")
            new_data.pop("_answer")
        else:
            assert not any([k.startswith("_") for k in new_data.keys()])

        new_dataset.append(new_data)

    print(f"answer_error_cnt: {answer_error_cnt}")

    # store new dataset
    with open(args.output_path, "w") as f:
        json.dump(new_dataset, f, indent=4, default=str)  # use `default=str` to serialize int64
    print(f"Saved new dataset to {args.output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--mimic_iv_dir", type=str, required=True)
    parser.add_argument("--mimic_cxr_jpg_dir", type=str, required=True)
    parser.add_argument("--chest_imagenome_dir", type=str, required=True)
    parser.add_argument("--label_dataset_path", type=str, default="./preprocessed_data/test_dataset.csv")
    parser.add_argument("--json_file", type=str, default="../mimiccxrvqa/dataset/_test.json")
    parser.add_argument("--output_path", type=str, default="../mimiccxrvqa/dataset/test.json")
    # parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    args.split = os.path.basename(args.output_path).split(".")[0]
    print(args)

    main(args)
