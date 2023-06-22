#!/bin/bash

# Check if save_dir argument is provided
if [ -z "$1" ]; then
    echo "Error: <save_dir> argument is missing."
    echo "Usage: bash download_mimic_cxr_report.sh <save_dir>"
    exit 1
fi

# Define file save directory
save_dir="$1"

# Capture the start time
start_time=$(date +%s)

# Read username and password (for PhysioNet)
read -p "Username: " USERNAME
read -s -p "Password: " PASSWORD
printf "\n"

# Define directories
MIMIC_CXR="https://physionet.org/files/mimic-cxr/2.0.0"

# Download MIMIC-CXR reports
printf "Downloading reports...\n"
wget -c --user $USERNAME --password $PASSWORD "$MIMIC_CXR/mimic-cxr-reports.zip" -P "$save_dir"
unzip "$save_dir/mimic-cxr-reports.zip" -d "$save_dir"

printf "All files downloaded.\n"
