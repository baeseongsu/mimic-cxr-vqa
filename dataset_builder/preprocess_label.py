import os
import json
import argparse
import numpy as np
import pandas as pd

# from tqdm import tqdm


def config():
    parser = argparse.ArgumentParser(description="preprocessing label information")

    # debug
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--debug_nrows", default=100000, type=int, help="debug mode - nrows")

    # file directory
    parser.add_argument("--save_dir", default="./preprocessed_data", type=str)
    parser.add_argument("--mimic_cxr_jpg_dir", default="../mimic-cxr-jpg/", type=str)
    parser.add_argument("--chest_imagenome_dir", default="../chest-imagenome/", type=str)

    args = parser.parse_args()

    return args


class LabelPreprocessor:
    def __init__(
        self,
        args,
    ):
        self.args = args

        # check debug
        self.nrows = args.debug_nrows if args.debug else None

        # load
        self._load_silver_attributes_relations()
        self._load_gold_attributes_relations()
        self._load_and_modify_chest_imagenome_ontology()

    def _load_silver_attributes_relations(self):
        # read
        silver_attributes_relations = pd.read_csv(
            os.path.join(self.args.chest_imagenome_dir, "silver_dataset/scene_tabular/attribute_relations_tabular.txt"),
            sep="\t",
            # usecols=None,
        )
        self.silver_attributes_relations = silver_attributes_relations

        # arrange
        silver_dataset = silver_attributes_relations.copy()
        silver_dataset["image_id"] = silver_dataset["image_id"].str.replace(".dcm", "")
        silver_dataset["object_id"] = silver_dataset["image_id"] + "_" + silver_dataset["bbox"]
        silver_dataset["sent_loc"] = silver_dataset["row_id"].apply(lambda x: float(x.split("|")[-1]))
        silver_dataset["annot_id"] = (
            silver_dataset["study_id"].astype(str) + "|" + silver_dataset["bbox"] + "|" + silver_dataset["relation"].astype(str) + "|" + silver_dataset["label_name"]
        )  # erase sent_loc
        silver_dataset = silver_dataset[["study_id", "image_id", "sent_loc", "bbox", "relation", "label_name", "categoryID", "annot_id", "object_id"]]

        self.silver_dataset = silver_dataset.reset_index(drop=True)

    def _load_gold_attributes_relations(self):
        # read
        gold_attributes_relations = pd.read_csv(
            os.path.join(self.args.chest_imagenome_dir, "gold_dataset/gold_attributes_relations_500pts_500studies1st.txt"),
            sep="\t",
            # usecols=None,
        )
        self.gold_attributes_relations = gold_attributes_relations

        # arrange
        gold_dataset = gold_attributes_relations.copy()
        gold_dataset["image_id"] = gold_dataset["image_id"].str.replace(".dcm", "")
        gold_dataset["object_id"] = gold_dataset["image_id"] + "_" + gold_dataset["bbox"]
        gold_dataset["sent_loc"] = gold_dataset["row_id"].apply(lambda x: float(x.split("|")[-1]))
        gold_dataset["annot_id"] = gold_dataset["study_id"].astype(str) + "|" + gold_dataset["bbox"] + "|" + gold_dataset["relation"].astype(str) + "|" + gold_dataset["label_name"]  # erase sent_loc
        gold_dataset = gold_dataset[["study_id", "image_id", "sent_loc", "bbox", "relation", "label_name", "categoryID", "annot_id", "object_id"]]

        self.gold_dataset = gold_dataset.reset_index(drop=True)

    def _load_and_modify_chest_imagenome_ontology(self):
        target_categories = ["anatomicalfinding", "technicalassessment", "disease", "tubesandlines", "device"]  # nlp

        # object list: 38 objects in total
        with open(os.path.join(self.args.chest_imagenome_dir, "semantics/objects_extracted_from_reports_v1.txt"), "r") as f:
            obj_v1 = [line.strip().replace(",", "") for line in f.readlines()]
        print(f"chest imagenome ontology - {len(obj_v1)} objects loaded: {obj_v1}")

        # attribute list: 76 attributes in total
        # anatomicalfinding(43), technicalassessment(5), disease(10), tubesandlines (12), device (5)
        with open(os.path.join(self.args.chest_imagenome_dir, "semantics/attribute_relations_v1.txt"), "r") as f:
            cat_attr_v1 = {}
            for line in f.readlines():
                cat, rel, attr = line.strip().replace(",", "").split("|")
                if cat in target_categories:
                    if cat not in cat_attr_v1:
                        cat_attr_v1[cat] = []
                    cat_attr_v1[cat] += [attr]
            attr_v1 = [vv for v in cat_attr_v1.values() for vv in v]
            attr_v1.remove("sternotomy wires")
        print(f"chest imagenome ontology - {len(attr_v1)} attributes loaded: {attr_v1}")

        attr_cat_v1 = {}
        for k, v in cat_attr_v1.items():
            for vv in v:
                attr_cat_v1[vv] = k

        # load ontology
        with open(os.path.join(self.args.chest_imagenome_dir, "semantics/label_to_UMLS_mapping.json"), "r") as f:
            ont = json.load(f)

            # NOTE: modify ontology
            # added; new obj-obj p2c ontology
            ont["all_children"]["lungs"] += [
                "hilar structures",
                "right hilar structures",
                "left hilar structures",
                # "main stem bronchus",
                # "left main stem bronchus",
                # "right main stem bronchus",
            ]
            ont["all_children"]["left lung"] += [
                "left hilar structures",
            ]
            ont["all_children"]["right lung"] += [
                "right hilar structures",
            ]
            ont["all_children"]["hilar structures"] = [
                "right hilar structures",
                "left hilar structures",
            ]
            ont["all_children"]["mediastinum"] += [
                "aortic arch",
                "svc",
            ]

            # build attr-attr p2c ontology
            attr_p2c_map = {}
            for parent, childs in ont["all_children"].items():
                if parent in attr_v1:
                    attr_p2c_map[parent] = [child for child in childs if child in attr_v1]

            # build attr-attr c2p ontology (reversed)
            attr_c2p_map = {}
            for parent, childs in attr_p2c_map.items():
                for child in childs:
                    if child not in attr_c2p_map:
                        attr_c2p_map[child] = []
                    attr_c2p_map[child].append(parent)

            # build obj-obj p2c ontology
            obj_p2c_map = {}
            for parent, childs in ont["all_children"].items():
                if parent in obj_v1:
                    new_childs = [child for child in childs if child in obj_v1]
                    if len(new_childs) > 0:
                        obj_p2c_map[parent] = new_childs

            # build obj-obj c2p ontology
            obj_c2p_map = {}
            for parent, childs in obj_p2c_map.items():
                for child in childs:
                    if child in obj_c2p_map.keys():
                        obj_c2p_map[child].append(parent)
                    else:
                        obj_c2p_map[child] = [parent]

            # NOTE: modify ontology
            # added: obj-attr possible relationship w/ obj-obj p2c ontology
            for k, v in ont["possible_attribute_of"].items():
                for child in obj_c2p_map:
                    parents = obj_c2p_map[child]
                    for parent in parents:
                        if (child in v) and (k in attr_v1):
                            if parent not in v:
                                ont["possible_attribute_of"][k] += [parent]
                                print(f"{k}-{child} \t => {k}-{parent}")

            # NOTE: modify ontology
            # added: obj-attr possible relationship w/ attr-attr p2c ontology
            for kc in ont["possible_attribute_of"].keys():
                if kc in attr_c2p_map:
                    kps = attr_c2p_map[kc]
                    for kp in kps:
                        vc = ont["possible_attribute_of"][kc]
                        vp = ont["possible_attribute_of"][kp]
                        vc_diff_vp = set(vc) - set(vp)
                        # vp_diff_vc = set(vp) - set(vc)
                        if len(vc_diff_vp) > 0:
                            ont["possible_attribute_of"][kp] += list(vc_diff_vp)
                            print(f"{kc}:{vc_diff_vp} \t => {kp}+={vc_diff_vp}")

        # list
        self.obj_v1 = obj_v1
        self.attr_v1 = attr_v1
        # cat-attr
        self.cat_attr_v1 = cat_attr_v1
        self.attr_cat_v1 = attr_cat_v1
        # p2c
        self.attr_p2c_map = attr_p2c_map
        self.attr_c2p_map = attr_c2p_map
        self.obj_p2c_map = obj_p2c_map
        self.obj_c2p_map = obj_c2p_map
        # full ont
        self.ont = ont

    def preprocessLabel_category(self, flag="silver"):
        if flag == "silver":
            silver_dataset = self.silver_dataset.copy()
            silver_dataset = silver_dataset[silver_dataset["categoryID"] != "nlp"]
            assert "normal" not in silver_dataset.label_name.unique()
            assert "abnormal" not in silver_dataset.label_name.unique()
            self.silver_dataset = silver_dataset.reset_index(drop=True)
        elif flag in ["gold", "gold+"]:
            gold_dataset = self.gold_dataset
            gold_dataset = gold_dataset[gold_dataset["categoryID"] != "nlp"]
            assert "normal" not in gold_dataset.label_name.unique()
            assert "abnormal" not in gold_dataset.label_name.unique()
            self.gold_dataset = gold_dataset.reset_index(drop=True)
        else:
            raise ValueError("flag must be either 'silver' or 'gold'")

    def preprocessLabel_bbox(self, flag="silver"):
        if flag == "silver":
            silver_dataset = self.silver_dataset.copy()
            silver_dataset = silver_dataset[silver_dataset["bbox"] != "unknown"]
            self.silver_dataset = silver_dataset.reset_index(drop=True)
        elif flag in ["gold", "gold+"]:
            gold_dataset = self.gold_dataset.copy()
            gold_dataset = gold_dataset[gold_dataset["bbox"] != "unknown"]
            self.gold_dataset = gold_dataset.reset_index(drop=True)
        else:
            raise ValueError("flag must be either 'silver' or 'gold'")

    def preprocessLabel_cohort(self, flag="silver"):
        if flag == "silver":
            silver_dataset = self.silver_dataset.copy()
            silver_cohort = pd.read_csv(
                os.path.join(self.args.save_dir, f"cohort_silver.csv"),
            )
            silver_iids = silver_cohort["image_id"].unique()
            silver_dataset = silver_dataset[silver_dataset.image_id.isin(silver_iids)]
            self.silver_dataset = silver_dataset.reset_index(drop=True)

        elif flag == "gold":
            gold_dataset = self.gold_dataset.copy()
            gold_cohort = pd.read_csv(
                os.path.join(self.args.save_dir, f"cohort_gold.csv"),
            )
            gold_iids = gold_cohort["image_id"].unique()
            gold_dataset = gold_dataset[gold_dataset["image_id"].isin(gold_iids)]
            assert len(gold_dataset["image_id"].unique()) == 500

            self.gold_dataset = gold_dataset.reset_index(drop=True)

        elif flag == "gold+":
            gold_dataset = self.gold_dataset.copy()
            gold_cohort = pd.read_csv(
                os.path.join(self.args.save_dir, f"cohort_gold.csv"),
            )
            gold_iids = gold_cohort["image_id"].unique()  # 2338 images
            gold_dataset = gold_dataset[gold_dataset["image_id"].isin(gold_iids)]  # 500 images
            assert len(gold_dataset["image_id"].unique()) == 500

            # NOTE: For studies where order >= 2, we use the silver labels
            # the pre-processing code of silver dataset (until here) should be run again
            self._load_silver_attributes_relations()
            self.preprocessLabel_category(flag="silver")
            self.preprocessLabel_bbox(flag="silver")
            silver_dataset = self.silver_dataset.copy()
            silver_dataset = silver_dataset[(~silver_dataset["image_id"].isin(gold_dataset.image_id.unique())) & (silver_dataset["image_id"].isin(gold_iids))]

            # The number of images of gold patients w/ 1st study (in cohort_gold.csv)
            print("# of images for gold patients w/ 1st study", gold_dataset["image_id"].nunique())
            # The number of images of gold patients w/ >=2nd study (in cohort_gold.csv)
            print("# of images for gold patients w/ >=2nd study", gold_cohort[gold_cohort["StudyOrder"] >= 2]["image_id"].nunique())
            # The number of images of gold patients w/ >=2nd study (in silver_dataset.csv ~ attribute_relations_tabular.txt)
            print("# of images for gold patients w/ >=2nd study with silver labels", silver_dataset["image_id"].nunique())
            print("NOTE: Some gold studies (>=2nd) are missing in the silver dataset!!!")
            gold_dataset = pd.concat([gold_dataset, silver_dataset], axis=0)

            self.gold_dataset = gold_dataset.reset_index(drop=True)

        else:
            raise ValueError("flag must be either 'silver' or 'gold'")

    def aggregate_labels_by_report_level(self, flag="silver", agg_option="last"):
        if flag == "silver":
            silver_dataset = self.silver_dataset
            silver_dataset = silver_dataset.sort_values(by=["study_id", "image_id", "sent_loc", "bbox", "label_name"])
            silver_dataset = silver_dataset.drop_duplicates(subset=["study_id", "image_id", "bbox", "label_name"], keep=agg_option)
            self.silver_dataset = silver_dataset.reset_index(drop=True)
        elif flag in ["gold", "gold+"]:
            gold_dataset = self.gold_dataset
            gold_dataset = gold_dataset.sort_values(by=["study_id", "image_id", "sent_loc", "bbox", "label_name"])
            gold_dataset = gold_dataset.drop_duplicates(subset=["study_id", "image_id", "bbox", "label_name"], keep=agg_option)
            self.gold_dataset = gold_dataset.reset_index(drop=True)
        else:
            raise ValueError("flag must be either 'silver' or 'gold'")

    def apply_attribute_p2c_ontology_to_dataset(self, flag="silver"):
        def _apply_attribute_p2c_ontology_to_dataset(dataset, attr_c2p_map, attr_cat_v1):
            dataset_parents = None
            for child, parents in attr_c2p_map.items():
                for parent in parents:
                    dataset_parent = dataset[dataset.label_name == child].copy()
                    dataset_parent["label_name"] = parent
                    dataset_parent["categoryID"] = attr_cat_v1[parent]
                    dataset_parent["annot_id"] = (
                        dataset_parent["study_id"].astype(str) + "|" + dataset_parent["bbox"] + "|" + dataset_parent["relation"].astype(str) + "|" + dataset_parent["label_name"]
                    )
                    dataset_parents = pd.concat([dataset_parents, dataset_parent], axis=0) if dataset_parents is not None else dataset_parent
            dataset = pd.concat([dataset, dataset_parents], axis=0)
            dataset = dataset.sort_values(by=["study_id", "image_id", "sent_loc", "bbox", "label_name"])
            dataset = dataset.drop_duplicates(subset=["study_id", "image_id", "bbox", "label_name"], keep="last")
            return dataset

        if flag == "silver":
            silver_dataset = self.silver_dataset
            silver_dataset = _apply_attribute_p2c_ontology_to_dataset(dataset=silver_dataset, attr_c2p_map=self.attr_c2p_map, attr_cat_v1=self.attr_cat_v1)
            self.silver_dataset = silver_dataset.reset_index(drop=True)
        elif flag in ["gold", "gold+"]:
            gold_dataset = self.gold_dataset
            gold_dataset = _apply_attribute_p2c_ontology_to_dataset(dataset=gold_dataset, attr_c2p_map=self.attr_c2p_map, attr_cat_v1=self.attr_cat_v1)
            self.gold_dataset = gold_dataset.reset_index(drop=True)
        else:
            raise ValueError("flag must be either 'silver' or 'gold'")

    def apply_object_p2c_ontology_to_dataset(self, flag="silver"):
        def _apply_object_p2c_ontology_to_dataset(dataset, obj_c2p_map):
            dataset_parents = None
            for child, parents in obj_c2p_map.items():
                for parent in parents:
                    dataset_parent = dataset[dataset.bbox == child].copy()
                    dataset_parent["bbox"] = parent
                    dataset_parent["annot_id"] = (
                        dataset_parent["study_id"].astype(str) + "|" + dataset_parent["bbox"] + "|" + dataset_parent["relation"].astype(str) + "|" + dataset_parent["label_name"]
                    )
                    dataset_parent["object_id"] = dataset_parent["image_id"] + "_" + dataset_parent["bbox"]
                    dataset_parents = pd.concat([dataset_parents, dataset_parent], axis=0) if dataset_parents is not None else dataset_parent
            dataset = pd.concat([dataset, dataset_parents], axis=0)
            dataset = dataset.sort_values(by=["study_id", "image_id", "sent_loc", "bbox", "label_name"])
            dataset = dataset.drop_duplicates(subset=["study_id", "image_id", "bbox", "label_name"], keep="last")
            return dataset

        if flag == "silver":
            silver_dataset = self.silver_dataset
            silver_dataset = _apply_object_p2c_ontology_to_dataset(dataset=silver_dataset, obj_c2p_map=self.obj_c2p_map)
            self.silver_dataset = silver_dataset.reset_index(drop=True)
        elif flag in ["gold", "gold+"]:
            gold_dataset = self.gold_dataset
            gold_dataset = _apply_object_p2c_ontology_to_dataset(dataset=gold_dataset, obj_c2p_map=self.obj_c2p_map)
            self.gold_dataset = gold_dataset.reset_index(drop=True)
        else:
            raise ValueError("flag must be either 'silver' or 'gold'")

    def apply_object_attribute_possible_relationship_to_dataset(self, flag="silver"):
        def _apply_object_attribute_possible_relationship_to_dataset(dataset, ont):
            removed_annot_ids = []
            obj_attr_combs = dataset[["bbox", "label_name"]].drop_duplicates().values
            for obj_attr_comb in obj_attr_combs:
                obj, attr = obj_attr_comb
                if attr not in ["abnormal", "normal", "artifact"]:
                    possible_objs = ont["possible_attribute_of"][attr]
                    if obj not in possible_objs:
                        removed_annot_ids += dataset[(dataset["bbox"] == obj) & (dataset["label_name"] == attr)].annot_id.tolist()

            dataset = dataset[~dataset.annot_id.isin(removed_annot_ids)]
            return dataset

        if flag == "silver":
            silver_dataset = self.silver_dataset
            silver_dataset = _apply_object_attribute_possible_relationship_to_dataset(dataset=silver_dataset, ont=self.ont)
            self.silver_dataset = silver_dataset.reset_index(drop=True)
        elif flag in ["gold", "gold+"]:
            gold_dataset = self.gold_dataset
            gold_dataset = _apply_object_attribute_possible_relationship_to_dataset(dataset=gold_dataset, ont=self.ont)
            self.gold_dataset = gold_dataset.reset_index(drop=True)
        else:
            raise ValueError("flag must be either 'silver' or 'gold'")

    def sanity_check(self, flag="silver"):
        if flag == "silver":
            dataset = self.silver_dataset
        elif flag in ["gold", "gold+"]:
            dataset = self.gold_dataset
        else:
            raise ValueError("flag must be either 'silver' or 'gold'")

        assert (dataset.object_id.apply(lambda x: x.split("_")[-1]) != dataset.bbox).sum() == 0
        assert (dataset.object_id.apply(lambda x: x.split("_")[0]) != dataset.image_id).sum() == 0
        assert (dataset.annot_id.apply(lambda x: x.split("|")[0]) != dataset.study_id.astype(str)).sum() == 0
        assert (dataset.annot_id.apply(lambda x: x.split("|")[1]) != dataset.bbox).sum() == 0
        assert (dataset.annot_id.apply(lambda x: x.split("|")[2]) != dataset.relation.astype(str)).sum() == 0
        assert (dataset.annot_id.apply(lambda x: x.split("|")[-1]) != dataset.label_name).sum() == 0

    def remove_minority_label(self, flag="silver"):
        def _remove_minority_label(dataset, removed_objs, removed_attrs):
            dataset = dataset[(~dataset["bbox"].isin(removed_objs)) & (~dataset["label_name"].isin(removed_attrs))]
            return dataset

        REMOVED_OBJS = ["left arm", "right arm"]
        REMOVED_ATTRS = [
            "artifact",
            "bronchiectasis",
            "pigtail catheter",
            "skin fold",
            "aortic graft/repair",
            "diaphragmatic eventration (benign)",
            "sternotomy wires",
        ]

        if flag == "silver":
            silver_dataset = self.silver_dataset
            silver_dataset = _remove_minority_label(dataset=silver_dataset, removed_objs=REMOVED_OBJS, removed_attrs=REMOVED_ATTRS)
            self.silver_dataset = silver_dataset.reset_index(drop=True)
        elif flag in ["gold", "gold+"]:
            gold_dataset = self.gold_dataset
            gold_dataset = _remove_minority_label(dataset=gold_dataset, removed_objs=REMOVED_OBJS, removed_attrs=REMOVED_ATTRS)
            self.gold_dataset = gold_dataset.reset_index(drop=True)
        else:
            raise ValueError("flag must be either 'silver' or 'gold'")

    def restore_normal_relation(self, flag="silver"):
        # compute possible object-attribute combinations (in here, a total of 609 combinations)
        objattr_combs = []
        for k, v in self.ont["possible_attribute_of"].items():
            if k in self.attr_v1:
                objattr_combs += [(vv, 0, k, self.attr_cat_v1[k]) for vv in v if vv in self.obj_v1]
        objattr_combs = pd.DataFrame(objattr_combs, columns=["bbox", "relation", "label_name", "categoryID"])

        if flag == "silver":
            silver_dataset = self.silver_dataset

            silver_all = pd.merge(silver_dataset[["study_id", "image_id"]].drop_duplicates().assign(key=1), objattr_combs.assign(key=1), on="key").drop("key", axis=1)
            silver_all["annot_id"] = silver_all["study_id"].astype(str) + "|" + silver_all["bbox"] + "|" + silver_all["relation"].astype(str) + "|" + silver_all["label_name"]
            silver_all["object_id"] = silver_all["image_id"] + "_" + silver_all["bbox"]

            silver_dataset = pd.concat([silver_dataset, silver_all])
            silver_dataset = silver_dataset.sort_values(by=["study_id", "image_id", "bbox", "label_name", "relation"], ascending=True)
            silver_dataset = silver_dataset.drop_duplicates(
                subset=[
                    "study_id",
                    "image_id",
                    "bbox",
                    "label_name",
                ],
                keep="last",
            )

            self.silver_dataset = silver_dataset.reset_index(drop=True)

        elif flag in ["gold", "gold+"]:
            gold_dataset = self.gold_dataset

            gold_all = pd.merge(gold_dataset[["study_id", "image_id"]].drop_duplicates().assign(key=1), objattr_combs.assign(key=1), on="key").drop("key", axis=1)
            gold_all["annot_id"] = gold_all["study_id"].astype(str) + "|" + gold_all["bbox"] + "|" + gold_all["relation"].astype(str) + "|" + gold_all["label_name"]
            gold_all["object_id"] = gold_all["image_id"] + "_" + gold_all["bbox"]

            gold_dataset = pd.concat([gold_dataset, gold_all])
            gold_dataset = gold_dataset.sort_values(by=["study_id", "image_id", "bbox", "label_name", "relation"], ascending=True)
            gold_dataset = gold_dataset.drop_duplicates(
                subset=[
                    "study_id",
                    "image_id",
                    "bbox",
                    "label_name",
                ],
                keep="last",
            )

            self.gold_dataset = gold_dataset.reset_index(drop=True)
        else:
            raise ValueError("flag must be either 'silver' or 'gold'")

    def save_label_dataset(self, flag="silver"):
        if flag == "silver":
            silver_dataset = self.silver_dataset
            silver_dataset.to_csv(os.path.join(self.args.save_dir, "silver_dataset.csv"), index=False)
            print("silver dataset saved")
        elif flag == "gold":
            gold_dataset = self.gold_dataset
            gold_dataset.to_csv(os.path.join(self.args.save_dir, "gold_dataset.csv"), index=False)
            print("gold dataset saved")
        elif flag == "gold+":
            gold_dataset = self.gold_dataset
            gold_dataset.to_csv(os.path.join(self.args.save_dir, "gold+_dataset.csv"), index=False)
            print("gold+ dataset saved")
        else:
            raise ValueError("flag must be either 'silver' or 'gold'")

    def split_and_save_dataset(self):
        from sklearn.model_selection import train_test_split

        # 0
        # gold_dataset = self.gold_dataset
        # silver_dataset = self.silver_dataset
        # To avoid the problem of the order of the dataset, we load the dataset from the saved csv file.
        gold_dataset = pd.read_csv(os.path.join(self.args.save_dir, "gold_dataset.csv"))
        silver_dataset = pd.read_csv(os.path.join(self.args.save_dir, "silver_dataset.csv"))

        # 1: divide by relation type
        silver_dataset_abn = silver_dataset[silver_dataset["relation"] == 1]
        silver_dataset_nm = silver_dataset[~silver_dataset.study_id.isin(silver_dataset_abn.study_id.unique())]

        # 2: split sids (train, valid)
        SEED = 103
        TEST_SIZE = 0.05

        train_study_ids_abn = silver_dataset_abn.study_id.unique()
        train_study_ids_abn, valid_study_ids_abn = train_test_split(train_study_ids_abn, test_size=TEST_SIZE, random_state=SEED)
        train_study_ids_nm = silver_dataset_nm.study_id.unique()
        train_study_ids_nm, valid_study_ids_nm = train_test_split(train_study_ids_nm, test_size=TEST_SIZE, random_state=SEED)

        train_sids = train_study_ids_abn.tolist() + train_study_ids_nm.tolist()
        valid_sids = valid_study_ids_abn.tolist() + valid_study_ids_nm.tolist()
        # test_sids = gold_dataset.study_id.unique().tolist()

        train_dataset = silver_dataset.loc[silver_dataset["study_id"].isin(train_sids)]
        valid_dataset = silver_dataset.loc[silver_dataset["study_id"].isin(valid_sids)]
        # test_dataset = gold_dataset.loc[gold_dataset["study_id"].isin(test_sids)]
        test_dataset = gold_dataset.copy()

        # 3: save
        train_dataset = train_dataset.reset_index(drop=True)
        valid_dataset = valid_dataset.reset_index(drop=True)
        test_dataset = test_dataset.reset_index(drop=True)

        train_dataset.to_csv(os.path.join(self.args.save_dir, "train_dataset.csv"), index=False)  # silver
        valid_dataset.to_csv(os.path.join(self.args.save_dir, "valid_dataset.csv"), index=False)  # silver
        test_dataset.to_csv(os.path.join(self.args.save_dir, "test_dataset.csv"), index=False)  # gold
        print("silver, gold dataset -> train, valid, test dataset saved")


def main(args):
    # load preprocessor
    label_preproc = LabelPreprocessor(args)

    FLAGS = ["silver", "gold", "gold+"]
    # FLAGS = ["gold+"]

    for flag in FLAGS:
        # preprocessing - labels
        label_preproc.preprocessLabel_category(flag=flag)
        label_preproc.preprocessLabel_bbox(flag=flag)
        label_preproc.preprocessLabel_cohort(flag=flag)
        label_preproc.aggregate_labels_by_report_level(flag=flag, agg_option="last")

        # preprocessing - ontology
        label_preproc.apply_attribute_p2c_ontology_to_dataset(flag=flag)
        label_preproc.apply_object_p2c_ontology_to_dataset(flag=flag)
        label_preproc.apply_object_attribute_possible_relationship_to_dataset(flag=flag)

        label_preproc.sanity_check(flag=flag)

        # preprocessing
        label_preproc.remove_minority_label(flag=flag)
        label_preproc.restore_normal_relation(flag=flag)

        # save
        label_preproc.save_label_dataset(flag=flag)

    if "silver" in FLAGS and "gold" in FLAGS:
        label_preproc.split_and_save_dataset()


if __name__ == "__main__":
    args = config()
    main(args)
    print("Done")
