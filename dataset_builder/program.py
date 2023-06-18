import random
from typing import List
import pandas as pd


def program_1(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    assert len(object) == 1 and len(category) == 1 and len(attribute) == 0
    relation = 1 if relation else 0
    features = features[(features["object"] == object[0]) & (features["relation"] == relation)]
    features = features[features["category"] == category[0]]
    result = len(features) > 0
    return result


def program_2(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    assert len(object) == 1 and len(category) == 0 and len(attribute) == 1
    relation = 1 if relation else 0
    features = features[(features["object"] == object[0]) & (features["relation"] == relation)]
    features = features[features["attribute"] == attribute[0]]
    result = len(features) > 0
    return result


def program_3(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    # define abnormality
    category = ["anatomicalfinding", "disease", "device", "tubesandlines"]
    assert len(object) == 1 and len(category) == 4 and len(attribute) == 0, f"{len(object)}/{len(category)}/{len(attribute)}"
    # verify
    features = features[(features["object"] == object[0]) & (features["relation"] == relation)]
    features = features[features["category"].isin(category)]
    result = len(features) > 0
    return result


def program_4(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    assert len(object) == 1 and len(category) == 2 and len(attribute) == 0
    features = features[(features["object"] == object[0]) & (features["relation"] == 1)]
    # verify
    features_1 = features[features["category"] == category[0]]
    # verify
    features_2 = features[features["category"] == category[1]]
    # logical or
    result = (len(features_1) > 0) or (len(features_2) > 0)
    return result


def program_5(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    assert len(object) == 1 and len(category) == 0 and len(attribute) == 2
    features = features[(features["object"] == object[0]) & (features["relation"] == 1)]
    # verify
    features_1 = features[features["attribute"] == attribute[0]]
    # verify
    features_2 = features[features["attribute"] == attribute[1]]
    # logical and
    result = (len(features_1) > 0) and (len(features_2) > 0)
    return result


def program_6(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    assert len(object) == 1 and len(category) == 0 and len(attribute) == 2
    features = features[(features["object"] == object[0]) & (features["relation"] == 1)]
    # verify
    features_1 = features[features["attribute"] == attribute[0]]
    # verify
    features_2 = features[features["attribute"] == attribute[1]]
    # logical or
    result = (len(features_1) > 0) or (len(features_2) > 0)
    return result


def program_7(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    assert len(object) == 1 and len(category) == 1 and len(attribute) == 0
    features = features[(features["object"] == object[0]) & (features["relation"] == 1)]
    features = features[features["category"] == category[0]]
    result = sorted(features["attribute"].unique())
    return result


def program_8(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    # define abnormality
    category = ["anatomicalfinding", "disease", "device", "tubesandlines"]
    assert len(object) == 1 and len(category) == 4 and len(attribute) == 0, f"{len(object)}/{len(category)}/{len(attribute)}"
    # query
    features = features[(features["object"] == object[0]) & (features["relation"] == relation)]
    features = features[features["category"].isin(category)]
    result = sorted(features["attribute"].unique())
    return result


def program_9(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    assert len(object) == 1 and len(category) == 2 and len(attribute) == 0
    # query_union
    features = features[(features["object"] == object[0]) & (features["relation"] == 1)]
    features = features[features["category"].isin(category)]
    result = sorted(features["attribute"].unique())
    return result


def program_10(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    assert len(object) == 1 and len(category) == 1 and len(attribute) == 2
    features = features[(features["object"] == object[0]) & (features["relation"] == 1)]
    features = features[features["attribute"].isin(attribute)]
    result = sorted(features["attribute"].unique())
    return result


def program_11(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    # define abnormality
    category = ["anatomicalfinding", "disease", "device", "tubesandlines"]
    assert len(object) == 2 and len(category) == 4 and len(attribute) == 0, f"{len(object)}/{len(category)}/{len(attribute)}"
    # query - object1
    features_1 = features[(features["object"] == object[0]) & (features["relation"] == relation)]
    features_1 = features_1[features_1["category"].isin(category)]
    # query - object2
    features_2 = features[(features["object"] == object[1]) & (features["relation"] == relation)]
    features_2 = features_2[features_2["category"].isin(category)]
    # verify_disjunctive
    result = (len(features_1) > 0) or (len(features_2) > 0)
    return result


def program_12(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    # define abnormality
    category = ["anatomicalfinding", "disease", "device", "tubesandlines"]
    assert len(object) == 2 and len(category) == 4 and len(attribute) == 0, f"{len(object)}/{len(category)}/{len(attribute)}"
    # query - object1
    features_1 = features[(features["object"] == object[0]) & (features["relation"] == relation)]
    features_1 = features_1[features_1["category"].isin(category)]
    # query - object2
    features_2 = features[(features["object"] == object[1]) & (features["relation"] == relation)]
    features_2 = features_2[features_2["category"].isin(category)]
    # verify_conjunctive
    result = (len(features_1) > 0) and (len(features_2) > 0)
    return result


def program_13(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    assert len(object) == 2 and len(category) == 1 and len(attribute) == 0
    # query
    features_1 = features[(features["object"] == object[0]) & (features["relation"] == 1)]
    features_1 = features_1[features_1["category"] == category[0]]
    features_1 = features_1["attribute"].unique()
    # query
    features_2 = features[(features["object"] == object[1]) & (features["relation"] == 1)]
    features_2 = features_2[features_2["category"] == category[0]]
    features_2 = features_2["attribute"].unique()
    # union
    result = sorted(list(set(features_1).union(set(features_2))))
    return result


def program_14(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    assert len(object) == 2 and len(category) == 1 and len(attribute) == 0
    # query
    features_1 = features[(features["object"] == object[0]) & (features["relation"] == 1)]
    features_1 = features_1[features_1["category"] == category[0]]
    features_1 = features_1["attribute"].unique()
    # query
    features_2 = features[(features["object"] == object[1]) & (features["relation"] == 1)]
    features_2 = features_2[features_2["category"] == category[0]]
    features_2 = features_2["attribute"].unique()
    # intersection
    result = sorted(list(set(features_1).intersection(set(features_2))))
    return result


def program_15(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    assert len(object) == 2 and len(category) == 1 and len(attribute) == 0
    # query
    features_1 = features[(features["object"] == object[0]) & (features["relation"] == 1)]
    features_1 = features_1[features_1["category"] == category[0]]
    features_1 = features_1["attribute"].unique()
    # query
    features_2 = features[(features["object"] == object[1]) & (features["relation"] == 1)]
    features_2 = features_2[features_2["category"] == category[0]]
    features_2 = features_2["attribute"].unique()
    # difference
    result = sorted(list(set(features_1).difference(set(features_2))))
    return result


def program_16(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    # define abnormality
    category = ["anatomicalfinding", "disease", "device", "tubesandlines"]
    assert len(object) == 2 and len(category) == 4 and len(attribute) == 0, f"{len(object)}/{len(category)}/{len(attribute)}"
    # query - object1
    features_1 = features[(features["object"] == object[0]) & (features["relation"] == relation)]
    features_1 = features_1[features_1["category"].isin(category)]
    features_1 = features_1["attribute"].unique()
    # query - object2
    features_2 = features[(features["object"] == object[1]) & (features["relation"] == relation)]
    features_2 = features_2[features_2["category"].isin(category)]
    features_2 = features_2["attribute"].unique()
    # union
    result = sorted(list(set(features_1).union(set(features_2))))
    return result


def program_17(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    # define abnormality
    category = ["anatomicalfinding", "disease", "device", "tubesandlines"]
    assert len(object) == 2 and len(category) == 4 and len(attribute) == 0, f"{len(object)}/{len(category)}/{len(attribute)}"
    # query - object1
    features_1 = features[(features["object"] == object[0]) & (features["relation"] == relation)]
    features_1 = features_1[features_1["category"].isin(category)]
    features_1 = features_1["attribute"].unique()
    # query - object2
    features_2 = features[(features["object"] == object[1]) & (features["relation"] == relation)]
    features_2 = features_2[features_2["category"].isin(category)]
    features_2 = features_2["attribute"].unique()
    # intersection
    result = sorted(list(set(features_1).intersection(set(features_2))))
    return result


def program_18(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    # define abnormality
    category = ["anatomicalfinding", "disease", "device", "tubesandlines"]
    assert len(object) == 2 and len(category) == 4 and len(attribute) == 0, f"{len(object)}/{len(category)}/{len(attribute)}"
    # query - object1
    features_1 = features[(features["object"] == object[0]) & (features["relation"] == relation)]
    features_1 = features_1[features_1["category"].isin(category)]
    features_1 = features_1["attribute"].unique()
    # query - object2
    features_2 = features[(features["object"] == object[1]) & (features["relation"] == relation)]
    features_2 = features_2[features_2["category"].isin(category)]
    features_2 = features_2["attribute"].unique()
    # difference
    result = sorted(list(set(features_1).difference(set(features_2))))
    return result


def program_19(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    assert len(object) == 0 and len(category) == 1 and len(attribute) == 0
    features = features[(features["relation"] == 1)]
    features = features[features["category"] == category[0]]
    result = len(features) > 0
    return result


def program_20(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    # define abnormality
    category = ["anatomicalfinding", "disease", "device", "tubesandlines"]
    assert len(object) == 0 and len(category) == 4 and len(attribute) == 0, f"{len(object)}/{len(category)}/{len(attribute)}"
    features = features[features["relation"] == relation]
    features = features[features["category"].isin(category)]
    result = len(features) > 0
    return result


def program_21(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    assert len(object) == 0 and len(category) == 2 and len(attribute) == 0
    features = features[(features["relation"] == 1)]
    features_1 = features[features["category"] == category[0]]
    features_2 = features[features["category"] == category[1]]
    result = (len(features_1) > 0) or (len(features_2) > 0)
    return result


def program_22(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    assert len(object) == 0 and len(category) == 0 and len(attribute) == 1
    features = features[(features["relation"] == 1)]
    features = features[features["attribute"] == attribute[0]]
    result = len(features) > 0
    return result


def program_23(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    assert len(object) == 0 and len(category) == 0 and len(attribute) == 2
    features = features[(features["relation"] == 1)]
    features_1 = features[features["attribute"] == attribute[0]]
    features_2 = features[features["attribute"] == attribute[1]]
    result = (len(features_1) > 0) and (len(features_2) > 0)
    return result


def program_24(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    assert len(object) == 0 and len(category) == 0 and len(attribute) == 2
    features = features[(features["relation"] == 1)]
    features_1 = features[features["attribute"] == attribute[0]]
    features_2 = features[features["attribute"] == attribute[1]]
    result = (len(features_1) > 0) or (len(features_2) > 0)
    return result


def program_25(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    assert len(object) == 0 and len(category) == 1 and len(attribute) == 0
    features = features[(features["relation"] == 1)]
    features = features[features["category"] == category[0]]
    result = sorted(features["attribute"].unique())
    return result


def program_26(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    assert len(object) == 0 and len(category) == 2 and len(attribute) == 0
    features = features[(features["relation"] == 1)]
    features = features[features["category"].isin(category)]
    result = sorted(features["attribute"].unique())
    return result


def program_27(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    # define abnormality
    category = ["anatomicalfinding", "disease", "device", "tubesandlines"]
    assert len(object) == 0 and len(category) == 4 and len(attribute) == 0, f"{len(object)}/{len(category)}/{len(attribute)}"

    features = features[features["relation"] == relation]
    features = features[features["category"].isin(category)]
    result = sorted(features["attribute"].unique())
    return result


def program_28(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    assert len(object) == 0 and len(category) == 1 and len(attribute) == 2
    features = features[features["relation"] == 1]
    features = features[features["attribute"].isin(attribute)]
    result = sorted(features["attribute"].unique())
    return result


def program_29(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    assert len(object) == 2 and len(category) == 0 and len(attribute) == 1
    relation = 1 if relation else 0
    features = features[(features["attribute"] == attribute[0]) & (features["relation"] == relation)]
    features_1 = features[features["object"] == object[0]]
    features_2 = features[features["object"] == object[1]]
    result = (len(features_1) > 0) and (len(features_2) > 0)
    return result


def program_30(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    assert len(object) == 2 and len(category) == 0 and len(attribute) == 1
    relation = 1 if relation else 0
    features = features[(features["attribute"] == attribute[0]) & (features["relation"] == relation)]
    features_1 = features[features["object"] == object[0]]
    features_2 = features[features["object"] == object[1]]
    result = (len(features_1) > 0) or (len(features_2) > 0)
    return result


def program_31(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    assert len(object) == 0 and len(category) == 0 and len(attribute) == 1
    features = features[(features["attribute"] == attribute[0]) & (features["relation"] == 1)]
    result = sorted(features["object"].unique())
    return result


def program_32(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    assert len(object) == 2 and len(category) == 0 and len(attribute) == 1
    features = features[(features["attribute"] == attribute[0]) & (features["relation"] == 1)]
    features = features[features["object"].isin(object)]
    result = sorted(features["object"].unique())
    return result


def program_33(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True, debug=False):
    # define abnormality
    category = ["anatomicalfinding", "disease", "device", "tubesandlines"]
    assert len(object) == 2 and len(category) == 4 and len(attribute) == 0, f"{len(object)}/{len(category)}/{len(attribute)}"
    features = features[(features["category"].isin(category)) & (features["relation"] == relation)]
    features = features[features["object"].isin(object)]
    result = sorted(features["object"].unique())
    return result


def program_34(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    assert len(object) == 0 and len(category) == 0 and len(attribute) == 2
    # query
    features_1 = features[(features["attribute"] == attribute[0]) & (features["relation"] == 1)]
    features_1 = features_1["object"].unique()
    # query
    features_2 = features[(features["attribute"] == attribute[1]) & (features["relation"] == 1)]
    features_2 = features_2["object"].unique()
    # union
    result = sorted(list(set(features_1).union(set(features_2))))
    return result


def program_35(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    assert len(object) == 0 and len(category) == 0 and len(attribute) == 2
    # query
    features_1 = features[(features["attribute"] == attribute[0]) & (features["relation"] == 1)]
    features_1 = features_1["object"].unique()
    # query
    features_2 = features[(features["attribute"] == attribute[1]) & (features["relation"] == 1)]
    features_2 = features_2["object"].unique()
    # intersection
    result = sorted(list(set(features_1).intersection(set(features_2))))
    return result


def program_36(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    assert len(object) == 0 and len(category) == 0 and len(attribute) == 2
    # query
    features_1 = features[(features["attribute"] == attribute[0]) & (features["relation"] == 1)]
    features_1 = features_1["object"].unique()
    # query
    features_2 = features[(features["attribute"] == attribute[1]) & (features["relation"] == 1)]
    features_2 = features_2["object"].unique()
    # difference
    result = sorted(list(set(features_1).difference(set(features_2))))
    return result


def program_37(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    assert len(object) == 2 and len(category) == 1 and len(attribute) == 0
    relation = 1 if relation else 0
    features = features[(features["category"] == category[0]) & (features["relation"] == relation)]
    features_1 = features[features["object"] == object[0]]
    features_2 = features[features["object"] == object[1]]
    result = (len(features_1) > 0) and (len(features_2) > 0)
    return result


def program_38(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    assert len(object) == 2 and len(category) == 1 and len(attribute) == 0
    relation = 1 if relation else 0
    features = features[(features["category"] == category[0]) & (features["relation"] == relation)]
    features_1 = features[features["object"] == object[0]]
    features_2 = features[features["object"] == object[1]]
    result = (len(features_1) > 0) or (len(features_2) > 0)
    return result


def program_39(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    assert len(object) == 0 and len(category) == 1 and len(attribute) == 0
    features = features[(features["category"] == category[0]) & (features["relation"] == 1)]
    result = sorted(features["object"].unique())
    return result


def program_40(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    assert len(object) == 0 and len(category) == 2 and len(attribute) == 0
    features_1 = features[(features["category"] == category[0]) & (features["relation"] == 1)]
    features_1 = features_1["object"].unique()
    features_2 = features[(features["category"] == category[1]) & (features["relation"] == 1)]
    features_2 = features_2["object"].unique()
    result = sorted(list(set(features_1).union(set(features_2))))
    return result


def program_41(features: pd.DataFrame, viewpos: List, relation=True):
    assert "ViewPosition" in features.columns
    assert len(viewpos) == 1

    features = features.ViewPosition.unique()
    assert len(features) == 1
    result = features[0] == viewpos[0]

    return result


def program_42(features: pd.DataFrame, viewpos: List, relation=True):
    assert "ViewPosition" in features.columns

    features = features.ViewPosition.unique()
    assert len(features) == 1
    result = features

    return result


def program_43(features: pd.DataFrame, viewpos: List, relation=True):
    assert "ViewPosition" in features.columns

    features = features.ViewPosition.unique()
    assert len(features) == 1
    result = features

    return result


def program_44(features: pd.DataFrame, gender: List, relation=True):
    assert "gender" in features.columns
    assert len(gender) == 1

    features = features.gender.unique()
    assert len(features) == 1
    result = features[0] == gender[0]

    return result


def program_45(features: pd.DataFrame, gender: List, relation=True):
    assert "gender" in features.columns
    assert len(gender) == 0

    features = features.gender.unique()
    assert len(features) == 1
    result = features

    return result


def program_46(features: pd.DataFrame, gender: List, relation=True):
    assert "gender" in features.columns
    assert len(gender) == 0
    features = features.gender.unique()
    assert len(features) == 1
    result = features

    return result


def program_47(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    assert len(category) == len(attribute) == 0
    assert "x1" in features.columns or "x2" in features.columns

    features_ll = features[features["object"] == "left lung"]
    features_rl = features[features["object"] == "right lung"]
    features_cd = features[features["object"] == "cardiac silhouette"]
    thorax_width = features_ll["x2"].values[0] - features_rl["x1"].values[0]
    cardiac_width = features_cd["x2"].values[0] - features_cd["x1"].values[0]
    ct_ratio = int(cardiac_width) / int(thorax_width)
    result = ct_ratio > 0.5

    return result


def program_48(features: pd.DataFrame, object: List, category: List, attribute: List, relation=True):
    assert len(category) == len(attribute) == 0
    assert "x1" in features.columns or "x2" in features.columns

    features_ll = features[features["object"] == "left lung"]
    features_rl = features[features["object"] == "right lung"]
    features_cd = features[features["object"] == "upper mediastinum"]
    thorax_width = features_ll["x2"].values[0] - features_rl["x1"].values[0]
    cardiac_width = features_cd["x2"].values[0] - features_cd["x1"].values[0]
    mt_ratio = int(cardiac_width) / int(thorax_width)
    result = mt_ratio > 1 / 3

    return result


PROGRAM_LIST = {
    "program_1": program_1,
    "program_2": program_2,
    "program_3": program_3,
    "program_4": program_4,
    "program_5": program_5,
    "program_6": program_6,
    "program_7": program_7,
    "program_8": program_8,
    "program_9": program_9,
    "program_10": program_10,
    "program_11": program_11,
    "program_12": program_12,
    "program_13": program_13,
    "program_14": program_14,
    "program_15": program_15,
    "program_16": program_16,
    "program_17": program_17,
    "program_18": program_18,
    "program_19": program_19,
    "program_20": program_20,
    "program_21": program_21,
    "program_22": program_22,
    "program_23": program_23,
    "program_24": program_24,
    "program_25": program_25,
    "program_26": program_26,
    "program_27": program_27,
    "program_28": program_28,
    "program_29": program_29,
    "program_30": program_30,
    "program_31": program_31,
    "program_32": program_32,
    "program_33": program_33,
    "program_34": program_34,
    "program_35": program_35,
    "program_36": program_36,
    "program_37": program_37,
    "program_38": program_38,
    "program_39": program_39,
    "program_40": program_40,
    "program_41": program_41,
    "program_42": program_42,
    "program_43": program_43,
    "program_44": program_44,
    "program_45": program_45,
    "program_46": program_46,
    "program_47": program_47,
    "program_48": program_48,
}
