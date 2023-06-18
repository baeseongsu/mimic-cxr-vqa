import os
import argparse
import numpy as np
import pandas as pd

# from tqdm import tqdm


def config():
    parser = argparse.ArgumentParser(description="preprocessing cohorts")

    # debug
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--debug_nrows", default=100000, type=int, help="debug mode - nrows")

    # file directory
    parser.add_argument("--mimic_cxr_jpg_dir", default="../mimic-cxr-jpg/", type=str)
    parser.add_argument("--chest_imagenome_dir", default="../chest-imagenome/", type=str)
    parser.add_argument("--save_dir", default="./preprocessed_data", type=str)

    parser.add_argument("--max_study_order", default=20, type=int)

    args = parser.parse_args()

    return args


class CohortPreprocessor:
    def __init__(
        self,
        args,
    ):
        self.args = args

        # check debug
        self.nrows = args.debug_nrows if args.debug else None

        # load dataset
        self._load_mimic_cxr_metadata()
        self._load_bbox_objects_tabular()  # silver bbox
        self._load_attribute_relations_tabular()  # silver attribute
        self._load_gold_patient_ids()
        self._load_gold_1st_image_ids()

    def _load_mimic_cxr_metadata(self):
        # read
        cxr_meta = pd.read_csv(
            os.path.join(self.args.mimic_cxr_jpg_dir, "mimic-cxr-2.0.0-metadata.csv"),
            usecols=["dicom_id", "subject_id", "study_id", "ViewPosition", "Rows", "Columns", "StudyDate", "StudyTime"],
        )
        print(cxr_meta.shape)

        # rename columns
        cxr_meta = cxr_meta.rename(columns={"dicom_id": "image_id"})

        # build a new column: StudyDateTime
        cxr_meta["StudyDateTime"] = pd.to_datetime(cxr_meta.StudyDate.astype(str).apply(lambda x: f"{x[:4]}-{x[4:6]}-{x[6:]}") + " " + cxr_meta.StudyTime.apply(lambda x: "%010.3f" % x))

        # build a new column: StudyOrder
        cxr_meta_ = cxr_meta.copy()
        cxr_meta_ = cxr_meta_.sort_values(by=["subject_id", "study_id", "StudyDateTime"])
        cxr_meta_ = cxr_meta_.drop_duplicates(subset=["subject_id", "study_id"], keep="first").copy()
        cxr_meta_["StudyDateTime_study_id"] = cxr_meta_["StudyDateTime"].astype(str) + cxr_meta_["study_id"].astype(str)
        cxr_meta_["StudyDateTime_study_id"] = pd.to_datetime(cxr_meta_["StudyDateTime_study_id"])
        cxr_meta_["StudyOrder"] = cxr_meta_.groupby(["subject_id"])["StudyDateTime_study_id"].rank(method="dense")
        cxr_meta["StudyOrder"] = cxr_meta["study_id"].map(cxr_meta_[["study_id", "StudyOrder"]].set_index("study_id")["StudyOrder"])

        # remove overlapped columns
        del cxr_meta["StudyDate"]
        del cxr_meta["StudyTime"]

        # after base preprocessing, keep all data
        self.mimic_cxr_metadata = cxr_meta.copy()

        # Assumption: Use only frontal images
        cxr_meta = cxr_meta[cxr_meta["ViewPosition"].isin(["AP", "PA"])].reset_index(drop=True)
        print(cxr_meta.shape)

        # Assumption: Given the same study_id, use only one image (studydatetime-first + dicom_id-first)
        cxr_meta = cxr_meta.sort_values(["study_id", "StudyDateTime", "image_id"], ascending=[True, True, True])
        cxr_meta = cxr_meta[cxr_meta["image_id"].isin(cxr_meta.groupby(["study_id"])["image_id"].first().values)]
        print(cxr_meta.shape)

        assert cxr_meta.groupby(["study_id", "StudyDateTime"])["image_id"].nunique().value_counts().size == 1

        self.meta_data = cxr_meta.copy()

    def _load_bbox_objects_tabular(self):
        bbox_objects_tabular = pd.read_csv(
            os.path.join(self.args.chest_imagenome_dir, "silver_dataset/scene_tabular/bbox_objects_tabular.txt"),
            sep="\t",
            nrows=self.nrows,
            usecols=[
                "object_id",
                "x1",
                "y1",
                "x2",
                "y2",
                "width",
                "height",
                "bbox_name",
                # 'synsets', 'name',
                # 'original_x1', 'original_y1', 'original_x2','original_y2', 'original_width', 'original_height',
            ],
        )
        bbox_objects_tabular["image_id"] = bbox_objects_tabular["object_id"].apply(lambda x: x.split("_")[0])  # add column
        self.bbox_objects_tabular = bbox_objects_tabular

    def _load_attribute_relations_tabular(self):
        attribute_relations_tabular = pd.read_csv(
            os.path.join(self.args.chest_imagenome_dir, "silver_dataset/scene_tabular/attribute_relations_tabular.txt"),
            sep="\t",
            nrows=self.nrows,
            usecols=["study_id", "image_id", "sent_loc", "row_id", "bbox", "categoryID", "label_name", "relation"],
        )
        self.attribute_relations_tabular = attribute_relations_tabular

    def _load_gold_patient_ids(self):
        gold_dataset = pd.read_csv(
            os.path.join(self.args.chest_imagenome_dir, "gold_dataset/gold_attributes_relations_500pts_500studies1st.txt"),
            sep="\t",
        )
        gold_dataset = gold_dataset.rename(columns={"patient_id": "subject_id"})
        gold_dataset["image_id"] = gold_dataset["image_id"].str.replace(".dcm", "")
        assert gold_dataset["subject_id"].nunique() == 500
        self.gold_pids = gold_dataset["subject_id"].unique()

    def _load_gold_1st_image_ids(self):
        gold_dataset = pd.read_csv(
            os.path.join(self.args.chest_imagenome_dir, "gold_dataset/gold_attributes_relations_500pts_500studies1st.txt"),
            sep="\t",
        )
        gold_dataset = gold_dataset.rename(columns={"patient_id": "subject_id"})
        gold_dataset["image_id"] = gold_dataset["image_id"].str.replace(".dcm", "")
        assert gold_dataset["subject_id"].nunique() == 500
        self.gold_1st_iids = gold_dataset["image_id"].unique()

    def preprocessImage_bounding_box(self):
        """
        1) Remove frontal images where the number of the bounding box in each image less than 36
        2) Remove frontal images whose width is more than 3 standard deviations. (in 224*224 image size)
        """
        # 0
        meta_data = self.meta_data.copy()
        bbox_objects_tabular = self.bbox_objects_tabular.copy()
        # 1
        num_of_unique_bbox = bbox_objects_tabular.groupby(["image_id"])["bbox_name"].nunique()
        remove_iids = num_of_unique_bbox[num_of_unique_bbox != 36].index
        meta_data = meta_data[~meta_data["image_id"].isin(remove_iids)]

        # 2
        def get_outlier_image_ids(bbox_name, tgt_data, src_data, measure_of_unit="width", n_std=3):
            tgt_array = tgt_data[tgt_data["bbox_name"] == bbox_name][measure_of_unit].copy()
            src_array = src_data[src_data["bbox_name"] == bbox_name][measure_of_unit].copy()

            mean, std = src_array.mean(), src_array.std()
            threshold_min = mean - n_std * std
            threshold_max = mean + n_std * std
            tgt_array_refined = tgt_array[(tgt_array < threshold_min) | (tgt_array > threshold_max)]

            outlier_image_ids = tgt_data.loc[tgt_array_refined.index]["image_id"].values
            return outlier_image_ids

        meta_bbox = bbox_objects_tabular[bbox_objects_tabular["image_id"].isin(meta_data.image_id.unique())].copy()
        assert len(bbox_objects_tabular.bbox_name.unique()) == 36
        for bbox_name in bbox_objects_tabular.bbox_name.unique():
            outlier_image_ids = get_outlier_image_ids(bbox_name=bbox_name, tgt_data=meta_bbox, src_data=bbox_objects_tabular)
            meta_data = meta_data[~meta_data["image_id"].isin(outlier_image_ids)]
        # -1
        self.meta_data = meta_data.reset_index(drop=True)
        print("preprocessImage_bounding_box: {}".format(self.meta_data.shape))

    def preprocessStudy_study_order(self, max_study_order=20):
        """
        we retain studies with study order <= max_study_order
        """
        # 0
        meta_data = self.meta_data.copy()
        # 1
        meta_data = meta_data[meta_data["StudyOrder"] <= max_study_order]
        # -1
        self.meta_data = meta_data.reset_index(drop=True)
        print("preprocessStudy_study_order: {}".format(self.meta_data.shape))

    def preprocessPatient_gold_pids(self, flag="silver"):
        """
        remove gold pids
        """
        # 0
        meta_data = self.meta_data.copy()
        # 1
        if flag == "silver":
            meta_data = meta_data[~meta_data["subject_id"].isin(self.gold_pids)]
        elif flag == "gold":
            meta_data = meta_data[meta_data["subject_id"].isin(self.gold_pids)]
        else:
            raise ValueError("flag must be either 'silver' or 'gold'")
        # -1
        self.meta_data = meta_data.reset_index(drop=True)
        print("remove_gold_pids: {}".format(self.meta_data.shape))

    def save_cohort_silver(self):
        # load
        meta_data = self.meta_data.copy()

        # arrange
        meta_data = meta_data[
            [
                "subject_id",
                "study_id",
                "image_id",
                "ViewPosition",
                "StudyDateTime",
                "StudyOrder",
            ]
        ]
        meta_data = meta_data.sort_values(by=["subject_id", "StudyOrder"])

        # save
        os.makedirs(self.args.save_dir, exist_ok=True)
        meta_data = meta_data.reset_index(drop=True)
        meta_data.to_csv(os.path.join(self.args.save_dir, f"cohort_silver.csv"), index=False)

    def reset_meta_data(self):
        self._load_mimic_cxr_metadata()

    def preprocessStudy_gold_1st(self):
        meta_data_raw = self.mimic_cxr_metadata.copy()
        meta_data_gold_1st = meta_data_raw[meta_data_raw["image_id"].isin(self.gold_1st_iids)]
        assert len(meta_data_gold_1st) == 500
        meta_data = self.meta_data.copy()
        meta_data = pd.concat(
            [
                meta_data[meta_data["StudyOrder"] != 1],
                meta_data_gold_1st,
            ]
        )
        assert len(self.meta_data) == len(meta_data)
        self.meta_data = meta_data.reset_index(drop=True)

    def save_cohort_gold(self):
        # load
        meta_data = self.meta_data.copy()

        # arrange
        meta_data = meta_data[
            [
                "subject_id",
                "study_id",
                "image_id",
                "ViewPosition",
                "StudyDateTime",
                "StudyOrder",
            ]
        ]
        meta_data = meta_data.sort_values(by=["subject_id", "StudyOrder"])

        # save
        os.makedirs(self.args.save_dir, exist_ok=True)
        meta_data = meta_data.reset_index(drop=True)
        meta_data.to_csv(os.path.join(self.args.save_dir, f"cohort_gold.csv"), index=False)


def main(args):
    # load preprocessor
    cohort_preproc = CohortPreprocessor(args)

    # NOTE: preprocessing for silver dataset
    cohort_preproc.preprocessImage_bounding_box()
    cohort_preproc.preprocessStudy_study_order(max_study_order=args.max_study_order)
    cohort_preproc.preprocessPatient_gold_pids(flag="silver")
    cohort_preproc.save_cohort_silver()

    # NOTE: preprocessing for gold dataset
    cohort_preproc.reset_meta_data()
    cohort_preproc.preprocessStudy_study_order(max_study_order=args.max_study_order)
    cohort_preproc.preprocessPatient_gold_pids(flag="gold")
    cohort_preproc.preprocessStudy_gold_1st()  # NOTE: specialized pre-processing for gold dataset
    cohort_preproc.save_cohort_gold()


if __name__ == "__main__":
    args = config()
    main(args)
    print("Done")
