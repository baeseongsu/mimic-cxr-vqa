#!/bin/bash

# Capture the start time
start_time=$(date +%s)

# NOTE: Define your own paths to the MIMIC-IV, MIMIC-CXR, and Chest ImaGenome datasets
MIMIC_IV_BASE_DIR="physionet.org/files/mimiciv/2.2"
MIMIC_CXR_BASE_DIR="physionet.org/files/mimic-cxr-jpg/2.0.0"
CHEST_IMAGENOME_BASE_DIR="physionet.org/files/chest-imagenome/1.0.0"

# Save current directory
orig_dir=$(pwd)

# Change directory and run python script
if [ ! -f "${CHEST_IMAGENOME_BASE_DIR}/silver_dataset/scene_tabular/attribute_relations_tabular.txt" ] || [ ! -f "${CHEST_IMAGENOME_BASE_DIR}/silver_dataset/scene_tabular/bbox_objects_tabular.txt" ]; then
    if [ ! -d "${CHEST_IMAGENOME_BASE_DIR}/utils/scene_postprocessing" ]; then
        echo "Error: Directory ${CHEST_IMAGENOME_BASE_DIR}/utils/scene_postprocessing does not exist."
        exit 1
    fi
    cd "${CHEST_IMAGENOME_BASE_DIR}/utils/scene_postprocessing"

    echo '{
        "SCENE_DIR": "../../silver_dataset/scene_graph",
        "OUTPUT_DIR": "../../silver_dataset/scene_tabular",
        "OUTPUT_TYPE": ["attributes", "objects"],
        "RDF_LEVEL": "study_id",
        "RESOURCE": "../../semantics/label_to_UMLS_mapping.json",
        "AGGREGATION": "last",
        "INCLUDE_SECTIONS": "all"
    }' > scenegraph_postprocessing_settings.json
    python scenegraph_postprocessing.py
    echo "Done with scene postprocessing"
fi

# Return to the original directory
cd "$orig_dir"

# Preprocessing and generate dataset
SAVE_DIR="dataset_builder/preprocessed_data/"
PREPROCESS_SCRIPTS=("preprocess_cohort.py" "preprocess_label.py")
SPLITS=("train" "valid" "test")

mkdir -p "$SAVE_DIR"

for split in "${SPLITS[@]}"; do
    if [ ! -f "${SAVE_DIR}/${split}_dataset.csv" ]; then
        for script in "${PREPROCESS_SCRIPTS[@]}"; do
            python "dataset_builder/${script}" \
                --mimic_cxr_jpg_dir $MIMIC_CXR_BASE_DIR \
                --chest_imagenome_dir $CHEST_IMAGENOME_BASE_DIR \
                --save_dir "$SAVE_DIR"
        done
    fi
done

for split in "${SPLITS[@]}"; do
    python dataset_builder/generate_answer.py \
        --mimic_iv_dir $MIMIC_IV_BASE_DIR \
        --mimic_cxr_jpg_dir $MIMIC_CXR_BASE_DIR \
        --chest_imagenome_dir $CHEST_IMAGENOME_BASE_DIR \
        --label_dataset_path "dataset_builder/preprocessed_data/${split}_dataset.csv" \
        --json_file "mimiccxrvqa/dataset/_${split}.json" \
        --output_path "mimiccxrvqa/dataset/${split}.json"
done

# Capture the end time and calculate runtime
end_time=$(date +%s)
runtime=$((end_time - start_time))

# Display the script runtime
echo "Script runtime: $runtime seconds"