#!/bin/bash

# # text preprocessing
# python prepare_data/create_section_files.py \
# --reports_path ../../physionet.org/files/mimic-cxr/2.0.0/files \
# --output_path ../../physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-sections

# image preprocessing
python prepare_data/resize_images.py \
--mimic_cxr_jpg_dir "../../physionet.org/files/mimic-cxr-jpg/2.0.0" \
--chest_imagenome_dir "../../physionet.org/files/chest-imagenome/1.0.0/" \
--save_img_dir "../../physionet.org/files/mimic-cxr-jpg/2.0.0/re512_3ch_contour_cropped" \
--resolution 512 \
--cropped

# python prepare_data/resize_images.py \
# --mimic_cxr_jpg_dir "../../physionet.org/files/mimic-cxr-jpg/2.0.0" \
# --chest_imagenome_dir "../../physionet.org/files/chest-imagenome/1.0.0/" \
# --save_img_dir "../../physionet.org/files/mimic-cxr-jpg/2.0.0/re224_3ch_contour_cropped" \
# --resolution 224 \
# --cropped

# # Build pretrained corpus
# python prepare_data/build_pretrain_corpus.py
