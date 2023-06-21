#!/bin/bash

# Capture the start time
start_time=$(date +%s)

# Read username and password (for PhysioNet)
read -p "Username: " USERNAME
read -s -p "Password: " PASSWORD
printf "\n"

# Define directories and file names
MIMIC_CXR="https://physionet.org/files/mimic-cxr-jpg/2.0.0/"
MIMIC_CXR_REPORT="https://physionet.org/files/mimic-cxr/2.0.0/"

# Define wget parameters for readability
WGET_PARAMS="-r -N -c -np --user $USERNAME --password $PASSWORD"

# Helper function to download and extract files
download() {
    local file_url=$1
    local destination_dir=$2
    local file_name=$(basename "$file_url")

    # Download the file
    wget $WGET_PARAMS "$file_url" "$destination_dir"

    # Extract if it's a zip file
    if [[ "$file_name" == *.zip ]]; then
        unzip -o "$destination_dir/$file_name" -d "$destination_dir" # -o: overwrite
    fi

    # Extract if it's a gzip file
    if [[ "$file_name" == *.gz ]]; then
        gzip -d "$destination_dir/$file_name"
    fi
}

# Download MIMIC-CXR images
printf "Downloading reports...\n"
download "$MIMIC_CXR_REPORT/mimic-cxr-reports.zip" "physionet.org/files/mimic-cxr/2.0.0/"

printf "Downloading images...\n"
download "$MIMIC_CXR" "physionet.org/files/mimic-cxr-jpg/2.0.0/"

printf "All images have been successfully downloaded.\n"