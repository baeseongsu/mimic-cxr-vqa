import os
import json
import argparse
import numpy as np
import pandas as pd

from arguments import CONFIG, TEMPLATE_ARG_DICT, COMB_LIST


def config():
    parser = argparse.ArgumentParser(description="preprocessing upperbound dataset")

    # file directory
    parser.add_argument("--datadir", default="../../../../dataset", type=str)
    parser.add_argument("--chest_imagenome_dir", default="../../../../../physionet.org/files/chest-imagenome/1.0.0", type=str)
    parser.add_argument("--save_dir", default="../../../../dataset", type=str)
    parser.add_argument("--split", default="test", type=str)

    args = parser.parse_args()

    return args


def load_bbox_information(args):
    if args.split == "test":
        bbox_info = pd.read_csv(
            os.path.join(args.chest_imagenome_dir, "gold_dataset", "gold_bbox_coordinate_annotations_1000images.csv"),
        )
        bbox_info = preprocess_gold_bbox_for_mediastinum(bbox_info)
        bbox_info = bbox_info[["image_id", "bbox_name", "coord224"]]
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

    return bbox_info


# Process for gold dataset
def preprocess_gold_bbox_for_mediastinum(df):
    """
    append mediastinum bboxes derived from the gold bbox annotations
    """
    # arrange columns
    df["image_id"] = df["image_id"].replace(".dcm", "", regex=True)
    df["object_id"] = df["image_id"] + "_" + df["bbox_name"]

    # NOTE: Mediastinum consists of the upper mediastinum and cardiac silhouette.
    bbox_names_in_mediastinum = ["upper mediastinum", "cardiac silhouette"]
    med = df[df["bbox_name"].isin(bbox_names_in_mediastinum)].reset_index(drop=True).copy()

    # set base columns
    lbase_coord_cols = ["x1", "y1", "original_x1", "original_y1"]
    rbase_coord_cols = ["x2", "y2", "original_x2", "original_y2"]
    base_cols = ["image_id"] + lbase_coord_cols + rbase_coord_cols
    med = med[base_cols].copy()

    # merge bbox coordinates
    assert all(med[lbase_coord_cols + rbase_coord_cols].dtypes == "float64") == True  # check whether base coord columns are float64 dtype or not
    for c in lbase_coord_cols:
        med[c] = med.groupby(["image_id"])[c].transform(lambda x: min(x))
    for c in rbase_coord_cols:
        med[c] = med.groupby(["image_id"])[c].transform(lambda x: max(x))

    # build columns from base columns
    med["width"] = med["x2"] - med["x1"]
    med["height"] = med["y2"] - med["y1"]
    med["original_width"] = med["original_x2"] - med["original_x1"]
    med["original_height"] = med["original_y2"] - med["original_y1"]
    med["bbox_name"] = "mediastinum"
    med["object_id"] = med["image_id"] + "_" + med["bbox_name"]
    med["annot_id"] = med["image_id"] + "|" + med["bbox_name"]
    med["coord224"] = [str([int(x1), int(y1), int(x2), int(y2)]) for x1, y1, x2, y2 in zip(med["x1"], med["y1"], med["x2"], med["y2"])]
    med["coord_original"] = [str([int(x1), int(y1), int(x2), int(y2)]) for x1, y1, x2, y2 in zip(med["original_x1"], med["original_y1"], med["original_x2"], med["original_y2"])]
    med = med.drop_duplicates(subset=["object_id"]).reset_index(drop=True).copy()  # originally, there are two mediastinum bboxes
    assert len(med) == 1000

    # append mediastinum bboxes
    df = pd.concat([df, med], ignore_index=True).sort_values(["image_id", "bbox_name"]).reset_index(drop=True).copy()
    return df


def process_label_dataset(label_dataset):
    # Perform any necessary preprocessing or filtering on label_dataset
    label_dataset = label_dataset[~label_dataset["object"].isin(CONFIG["REMOVED_OBJS"])]
    label_dataset = label_dataset[~label_dataset["attribute"].isin(CONFIG["REMOVED_ATTRS"])]
    label_dataset = label_dataset.reset_index(drop=True)
    return label_dataset


def convert_qa_to_upperbound_df(label_dataset, dataset):
    template_arg_dict = TEMPLATE_ARG_DICT
    template_dfs = []
    data_keys = ["image_id"]
    for k, v in template_arg_dict.items():
        template_keys = k.split("_")
        template_df = dataset[dataset.template_program.isin(v)]
        template_df = template_df[data_keys + template_keys + ["image_path"]]
        for template_key in template_keys:
            template_df = template_df.explode(template_key)
        template_df = pd.merge(label_dataset, template_df, how="inner", on=data_keys + template_keys).copy()
        template_dfs.append(template_df)
    template_dfs = pd.concat(template_dfs)
    return template_dfs


def main(args):
    # Load dataset
    dataset = json.load(open(os.path.join(args.datadir, f"{args.split}.json")))
    dataset = pd.DataFrame(dataset)
    dataset = dataset.rename({"argcomb_object": "object", "argcomb_attribute": "attribute", "argcomb_category": "category"}, axis=1)
    dataset = pd.concat([dataset.drop(["template_arguments"], axis=1), dataset["template_arguments"].apply(pd.Series)], axis=1)
    dataset["object"] = dataset["object"].apply(lambda x: list(x.values()))
    dataset["attribute"] = dataset["attribute"].apply(lambda x: list(x.values()))
    dataset["category"] = dataset["category"].apply(lambda x: list(x.values()))
    print(f"Load {len(dataset)} qa samples for {args.split} set ({dataset.image_id.nunique()} images)")

    # Load labeled dataset
    label_dataset_path = os.path.join(args.datadir, f"preprocessed_data/{args.split}_dataset.csv")
    label_dataset = pd.read_csv(label_dataset_path)
    label_dataset = label_dataset[["image_id", "bbox", "relation", "label_name", "categoryID", "object_id"]]
    label_dataset = label_dataset.rename(columns={"dicom_id": "image_id", "bbox": "object", "label_name": "attribute", "categoryID": "category"})
    print(label_dataset.shape)

    # Load bounding box information
    bbox_info = load_bbox_information(args)

    # Add bounding box information to label_dataset
    label_dataset = label_dataset.merge(bbox_info, on=["image_id", "object"], how="left")

    # Process label_dataset
    label_dataset = process_label_dataset(label_dataset)
    label_dataset = label_dataset[(~label_dataset.coord224.isna()) & (label_dataset.coord224 != "[0, 0, 0, 0]")]

    label_dataset = convert_qa_to_upperbound_df(label_dataset, dataset)
    label_dataset = label_dataset.sort_values(by=["image_id", "object", "attribute", "relation"], ascending=False)
    label_dataset = label_dataset.drop_duplicates(["image_id", "object", "attribute"], keep="first")

    label_dataset["coord224"] = label_dataset["coord224"].str.replace("[", "").str.replace("]", "").str.split(",")
    label_dataset["coord224"] = label_dataset["coord224"].apply(lambda x: [int(_x) for _x in x])
    label_dataset["height"] = label_dataset.apply(lambda x: abs(x.coord224[3] - x.coord224[1]), axis=1)
    label_dataset["width"] = label_dataset.apply(lambda x: abs(x.coord224[2] - x.coord224[0]), axis=1)
    label_dataset["comb"] = label_dataset["object"].astype("str") + "_" + label_dataset["attribute"].astype("str")
    label_dataset = label_dataset[(label_dataset.height > 0) & (label_dataset.width > 0)]
    label_dataset = label_dataset[label_dataset.comb.isin(COMB_LIST)]
    print(label_dataset.comb.nunique())
    print(label_dataset.shape)

    # To evaluate qa model
    label_dataset["template"] = "Is there ${attribute} in the ${object}?"
    label_dataset["question"] = label_dataset["template"]
    label_dataset["question"] = label_dataset.apply(lambda x: x.question.replace("${object}", x.object), axis=1)
    label_dataset["question"] = label_dataset.apply(lambda x: x.question.replace("${attribute}", x.attribute), axis=1)
    label_dataset["answer"] = np.where(label_dataset["relation"] == 0, ["no"], ["yes"])
    label_dataset["content_type"] = "condition"
    label_dataset["semantic_type"] = "verify"

    label_dataset = label_dataset.to_dict("records")

    # store new dataset
    args.save_path = os.path.join(args.save_dir, f"{args.split}_ref.json")

    with open(args.save_path, "w") as f:
        json.dump(label_dataset, f, indent=4, default=str)  # use `default=str` to serialize int64

    print(f"Saved new dataset to {args.save_path}")


if __name__ == "__main__":
    args = config()
    main(args)
    print("Done")
