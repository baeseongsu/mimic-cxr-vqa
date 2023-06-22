#!/bin/bash

# Function to display error message and usage instructions
display_usage() {
    echo "Error: <save_dir> argument is missing."
    echo "Usage: bash download_mimic_cxr_jpg.sh <save_dir>"
}

# Check if save_dir argument is provided
if [ -z "$1" ]; then
    display_usage
    exit 1
fi

# Assign save_dir from argument
save_dir="$1"

# Capture the start time
start_time=$(date +%s)

# Read username and password (for PhysioNet)
read -p "Username: " USERNAME
read -s -p "Password: " PASSWORD
printf "\n"

# Define directories and file names
MIMIC_CXR_JPG="https://physionet.org/files/mimic-cxr-jpg/2.0.0/"

# Define wget parameters for readability
WGET_PARAMS="-r -N -c -np --user $USERNAME --password $PASSWORD"

# Helper function to download and extract files
download() {
    local file_url="$1"
    local destination_dir="$2"
    local file_name=$(basename "$file_url")

    # Download the file
    wget $WGET_PARAMS "$file_url" -P "$destination_dir"

    # Extract if it's a zip file
    if [[ "$file_name" == *.zip ]]; then
        unzip -o "$destination_dir/$file_name" -d "$destination_dir" # -o: overwrite
    fi

    # Extract if it's a gzip file
    if [[ "$file_name" == *.gz ]]; then
        gzip -d "$destination_dir/$file_name"
    fi
}

# Download MIMIC-CXR-JPG images
printf "Downloading images...\n"
download "$MIMIC_CXR_JPG" "$save_dir"

printf "All files downloaded.\n"