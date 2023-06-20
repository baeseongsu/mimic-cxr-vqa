#!/bin/bash

# Txt preprocessing
# python txt/create_section_files.py \
# --reports_path ../../../physionet.org/files/mimic-cxr/2.0.0/files \
# --output_path ../../../physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-sections

# Img preprocessing
# python img/resize_images.py --resolution 512 --saved_dir "../../../physionet.org/files/mimic-cxr-jpg/2.0.0/re512_3ch_contour_cropped"
# python img/resize_images.py --resolution 224 --saved_dir "../../../physionet.org/files/mimic-cxr-jpg/2.0.0/re224_3ch_contour_cropped"

# Build pretrained corpus
python collect_dataset.py