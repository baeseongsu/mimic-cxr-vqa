#!/bin/bash

# Capture the start time
start_time=$(date +%s)

# Read username and password (for PhysioNet)
read -p "Username: " USERNAME
read -s -p "Password: " PASSWORD
printf "\n"

# Define directories and file names
MIMIC_CXR="https://physionet.org/files/mimic-cxr-jpg/2.0.0"

# Define wget parameters for readability
WGET_PARAMS="-r -N -c -np --user $USERNAME --password $PASSWORD"

# Helper function to download and extract files
download() {
    local file_url=$1
    local file_name=$(basename "$file_url")

    # Download the file
    wget $WGET_PARAMS "$file_url"
    if [ $? -ne 0 ]; then
        printf "Error: Failed to download $file_url\n"
        exit 1
    fi
}

# Function to read image paths from JSON using Python
get_image_paths() {
    local json_file=$1
    python -c "import json; f=open('$json_file'); data=json.load(f); print('\n'.join([item['image_path'] for item in data]))"
}

# Read JSON file and gather image paths using Python
image_paths_train=$(get_image_paths 'mimiccxrvqa/dataset/train.json')
image_paths_valid=$(get_image_paths 'mimiccxrvqa/dataset/valid.json')
image_paths_test=$(get_image_paths 'mimiccxrvqa/dataset/test.json')

image_paths=$(printf "%s\n%s\n%s\n" "$image_paths_train" "$image_paths_valid" "$image_paths_test")

# Convert string to array
IFS=$'\n' read -rd '' -a arr <<<"$image_paths"

# Make unique
arr=($(echo "${arr[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '))

# Count
printf "Total number of images: %d\n" "${#arr[@]}"

# Download MIMIC-CXR images
printf "Downloading images...\n"
for image_path in "${arr[@]}"
do
    printf "Downloading $image_path\n"
    download "$MIMIC_CXR/files/$image_path"
done
printf "All images have been successfully downloaded.\n"