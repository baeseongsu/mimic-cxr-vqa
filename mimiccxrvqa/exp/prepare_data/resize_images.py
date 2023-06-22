import os
import cv2
import parmap
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp


def load_mimic_cxr_meta(mimic_cxr_jpg_dir):
    # load raw data
    cxr_meta = pd.read_csv(os.path.join(mimic_cxr_jpg_dir, "mimic-cxr-2.0.0-metadata.csv"))

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

    # Assumption: Use only frontal images
    cxr_meta = cxr_meta[cxr_meta["ViewPosition"].isin(["AP", "PA"])].reset_index()

    # Assumption: Given the same study_id, use only one image (studydatetime-first + dicom_id-first)
    cxr_meta = cxr_meta.sort_values(["study_id", "StudyDateTime", "dicom_id"], ascending=[True, True, True])
    cxr_meta = cxr_meta[cxr_meta["dicom_id"].isin(cxr_meta.groupby(["study_id"]).dicom_id.first().values)]
    assert cxr_meta.groupby(["study_id", "StudyDateTime"]).dicom_id.nunique().value_counts().size == 1

    return cxr_meta


def _preprocess_image(rows, cropped, resolution_):
    """
    main process:
    - load
    - crop
    - resize
    - save

    reference:
    - https://stackoverflow.com/questions/52979965/crop-images-with-different-black-margins
    - https://til.songyunseop.com/python/tqdm-with-multiprocessing.html
    """

    for row in rows.iterrows():
        # get a row information
        row = row[1]

        # load an image
        pid, sid, iid = str(row.subject_id), str(row.study_id), str(row.dicom_id)
        image_path = os.path.join(args.mimic_cxr_jpg_dir, "files", f"p{pid[:2]}/p{pid}/s{sid}/{iid}.jpg")
        assert os.path.exists(image_path), f"image_path: {image_path}"
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # crop an image
        if cropped:
            ret, thresh = cv2.threshold(gray, 0, 255, 0)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            cnt = contours[max_index]
            x, y, w, h = cv2.boundingRect(cnt)
            crop_img = img[y : y + h, x : x + w]
        else:
            crop_img = img

        # resize image resolution
        resolution = (resolution_, resolution_)
        resize_img = cv2.resize(crop_img, resolution, interpolation=cv2.INTER_CUBIC)

        save_image_dir = os.path.join(args.save_img_dir, f"p{pid[:2]}/p{pid}/s{sid}")
        os.makedirs(save_image_dir, exist_ok=True)

        save_image_path = os.path.join(args.save_img_dir, f"p{pid[:2]}/p{pid}/s{sid}/{iid}.jpg")
        cv2.imwrite(save_image_path, resize_img)


def main(args):
    # load dataset
    meta_data = load_mimic_cxr_meta(mimic_cxr_jpg_dir=args.mimic_cxr_jpg_dir)

    # run multiprocessing
    num_cores = max(int(mp.cpu_count() / 16), 1)
    splitted_data = np.array_split(meta_data, num_cores)
    with mp.Pool(num_cores) as pool:
        parmap.map(_preprocess_image, splitted_data, args.cropped, args.resolution, pm_pbar=True)

    # # NOTE: [issue handling] image pre-processing for gold dataset
    gold_data = pd.read_csv(
        os.path.join(args.chest_imagenome_dir, "gold_dataset/gold_attributes_relations_500pts_500studies1st.txt"),
        sep="\t",
    )
    gold_data["image_id"] = gold_data["image_id"].str.replace(".dcm", "")
    gold_data = gold_data[["patient_id", "study_id", "image_id"]].drop_duplicates()
    gold_data = gold_data.rename(columns={"patient_id": "subject_id", "image_id": "dicom_id"})

    # run multiprocessing
    num_cores = int(mp.cpu_count() / 16)
    splitted_data = np.array_split(gold_data, num_cores)
    with mp.Pool(num_cores) as pool:
        parmap.map(_preprocess_image, splitted_data, args.cropped, args.resolution, pm_pbar=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # directory
    parser.add_argument("--mimic_cxr_jpg_dir", type=str, default="../../../physionet.org/files/mimic-cxr-jpg/2.0.0")
    parser.add_argument("--chest_imagenome_dir", type=str, default="../../../physionet.org/files/chest-imagenome/1.0.0/")
    parser.add_argument("--save_img_dir", type=str, default="../../../physionet.org/files/mimic-cxr-jpg/2.0.0/re512_3ch_contour_cropped")

    # image preprocessing
    parser.add_argument("--resolution", type=int, choices=[224, 512], required=True)
    parser.add_argument("--cropped", action="store_true")

    args = parser.parse_args()

    assert str(args.resolution) in os.path.basename(args.save_img_dir)
    assert ("contour_cropped" if args.cropped else "") in os.path.basename(args.save_img_dir)

    main(args)
