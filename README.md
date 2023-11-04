
# MIMIC-CXR-VQA

*A new collection of medical visual question answering dataset on MIMIC-CXR database*


## Overview

The MIMIC-CXR-VQA dataset is a complex (involving set and logical operations), diverse (with 48 templates), and large-scale (approximately 377K) resource, designed specifically for Visual Question Answering (VQA) tasks in the medical domain. Primarily focusing on chest radiographs, this dataset was mainly derived from the MIMIC-CXR-JPG and Chest ImaGenome datasets, both of which were sourced from Physionet.

The goal of the MIMIC-CXR-VQA dataset is to serve as a benchmark for evaluating the effectiveness of current medical VQA approaches. It not only functions as a tool for traditional medical VQA tasks but also has the unique quality of being an image-based Electronic Health Records (EHRs) Question Answering dataset resource. Therefore, we utilize question templates from the MIMIC-CXR-VQA dataset as seed question templates for image modality, to construct a multi-modal EHR QA dataset, [EHRXQA](https://github.com/baeseongsu/ehrxqa).

## Features

- [x] Provide a script to download source datasets (MIMIC-CXR-JPG, Chest ImaGenome, and MIMIC-IV) from Physionet.
- [x] Provide a script to preprocess the source datasets.
- [x] Provide a script to generate the MIMIC-CXR-VQA dataset (with answer information).

## Installation

### For Linux:

Ensure that you have Python 3.8.5 or higher installed on your machine. Set up the environment and install the required packages using the commands below:

```
# Set up the environment
conda create --name mimiccxrvqa python=3.8.5

# Activate the environment
conda activate mimiccxrvqa

# Install required packages
pip install pandas==1.1.3 tqdm==4.65.0 scikit-learn==0.23.2
```

## Setup

Clone this repository and navigate into it:

```
git clone https://github.com/baeseongsu/mimic-cxr-vqa.git
cd mimic-cxr-vqa
```

## Usage

### Privacy

We take data privacy very seriously. All of the data you access through this repository has been carefully prepared to prevent any privacy breaches or data leakage. You can use this data with confidence, knowing that all necessary precautions have been taken.

### Access Requirements

The MIMIC-CXR-VQA dataset is constructed from the MIMIC-CXR-JPG (v2.0.0), Chest ImaGenome (v1.0.0), and MIMIC-IV (v2.2). All these source datasets require a credentialed Physionet license. Due to these requirements and in adherence to the Data Use Agreement (DUA), only credentialed users can access the MIMIC-CXR-VQA dataset files (see Access Policy). To access the source datasets, you must fulfill all of the following requirements:

1. Be a [credentialed user](https://physionet.org/settings/credentialing/)
    - If you do not have a PhysioNet account, register for one [here](https://physionet.org/register/).
    - Follow these [instructions](https://physionet.org/credential-application/) for credentialing on PhysioNet.
    - Complete the "CITI Data or Specimens Only Research" [training course](https://physionet.org/about/citi-course/).
2. Sign the data use agreement (DUA) for each project
    - https://physionet.org/sign-dua/mimic-cxr-jpg/2.0.0/
    - https://physionet.org/sign-dua/chest-imagenome/1.0.0/
    - https://physionet.org/sign-dua/mimiciv/2.2/

### Accessing the MIMIC-CXR-VQA Dataset

While the complete MIMIC-CXR-VQA dataset is being prepared for publication on the Physionet platform, we provide partial access to the dataset via this repository for credentialed users. The MIMIC-CXR-VQA dataset mainly comprises three components: an image (I), a question (Q), and an answer (A). In this partial release, we omit the answer (A) and certain metadata, thereby maintaining privacy by preventing any instance-level information leakage. Moreover, during the creation of the dataset, we carefully implemented an unbiased sampling strategy for images, questions, and answers. This ensures no distribution-level leakage, such as the image-question distribution.

To access the MIMIC-CXR-VQA dataset, you can run the provided main script (which requires your unique Physionet credentials) in this repository as follows:

```
bash build_dataset.sh
```

During script execution, enter your PhysioNet credentials when prompted:

- Username: Enter your PhysioNet username and press `Enter`.
- Password: Enter your PhysioNet password and press `Enter`. The password characters won't appear on screen.

This script performs several actions: 1) it downloads the source datasets from Physionet, 2) preprocesses these datasets, and 3) generates the complete MIMIC-CXR-VQA dataset by creating ground-truth answer information.

Ensure you keep your credentials secure. If you encounter any issues, please ensure that you have the necessary permissions, a stable internet connection, and all prerequisite tools installed.

### Downloading MIMIC-CXR-JPG Images

To enhance user convenience, we will provide a script that allows you to download only the CXR images relevant to the MIMIC-CXR-VQA dataset, rather than downloading all the MIMIC-CXR-JPG images.

```
bash download_images.sh
```

During script execution, enter your PhysioNet credentials when prompted:

- Username: Enter your PhysioNet username and press `Enter`.
- Password: Enter your PhysioNet password and press `Enter`. The password characters won't appear on screen.

This script performs several actions: 1) it reads the image paths from the JSON files of the MIMIC-CXR-VQA dataset; 2) uses these paths to download the corresponding images from the MIMIC-CXR-JPG dataset hosted on Physionet; and 3) saves these images locally in the corresponding directories as per their paths.

### Dataset Structure

The dataset is structured as follows:

```
mimiccxrvqa
└── dataset
    ├── ans2idx.json
    ├── _train_part1.json
    ├── _train_part2.json
    ├── _valid.json
    ├── _test.json
    ├── train.json (available post-script execution)
    ├── valid.json (available post-script execution)
    └── test.json  (available post-script execution)
```

- The `mimiccxrvqa` is the root directory. Within this, the `dataset` directory contains various JSON files that are part of the MIMIC-CXR-VQA dataset.
- The `ans2idx.json` file is a dictionary mapping from answers to their corresponding indices.
- `_train_part1.json`, `_train_part2.json`, `_valid.json`, and `_test.json` are pre-release versions of the dataset files corresponding to the training, validation, and testing sets respectively. These versions are intentionally incomplete to safeguard privacy and prevent the leakage of sensitive information; they do not include certain crucial information, such as the answers.
- Once the main script is executed with valid Physionet credentials, the full versions of these files - `train.json`, `valid.json`, and `test.json` - will be generated. These files contain the complete information, including images, questions, and the corresponding answers for each entry in the respective sets.

### Dataset Description

The QA samples in the MIMIC-CXR-VQA dataset are stored in individual `.json` files. Each file contains a list of Python dictionaries with keys that indicate:

- `split`: a string indicating its split.
- `idx`: a number indicating its instance index.
- `image_id`: a string indicating the associated image ID.
- `question`: a question string.
- `content_type`: a string indicating its content type, which can be one of this list:
    - `anatomy`
    - `attribute`
    - `presence`
    - `abnormality`
    - `plane`
    - `gender`
    - `size`
- `semantic_type`: a string indicating its semantic type, which can be one of this list:
    - `verify`
    - `choose`
    - `query`
- `template`: a template string.
- `template_program`: a string indicating its template program. Each template has a unique program to get its answer from the database.
- `template_arguments`: a dictionary specifying its template arguments, consisting of five sub-dictionaries that represent the sampled values for arguments in the template. When an argument needs to appear multiple times in a question template, an index is appended to the dictionary.
    - `object`
    - `attribute`
    - `category`
    - `viewpos`
    - `gender`

Note that these details can be open-sourced without safety concerns and without revealing the dataset's distribution information (including image, question, and answer distributions), thanks to our uniform sampling strategy.

After validating the PhysioNet credentials, the `create_answer.py` script generates the following items:

- `answer`: an answer string.
- `subject_id`: a string indicating the corresponding subject ID (patient ID).
- `study_id`: a string indicating the corresponding study ID.
- `image_path`: a string indicating the corresponding image path.

To be specific, here is the example instance:

```
{
    "split": "train",
    "idx": 13280,
    "image_id": "34c81443-5a19ccad-7b5e431c-4e1dbb28-42a325c0",
    "question": "Are there signs of both pleural effusion and lung cancer in the left lower lung zone?",
    "content_type": "attribute",
    "semantic_type": "verify",
    "template": "Are there signs of both ${attribute_1} and ${attribute_2} in the ${object}?",
    "template_program": "program_5",
    "template_arguments": {
      "object": {
        "0": "left lower lung zone"
      },
      "attribute": {
        "0": "pleural effusion",
        "1": "lung cancer"
      },
      "category": {},
      "viewpos": {},
      "gender": {}
    },
	"answer": "Will be generated by dataset_builder/generate_answer.py"
	"subject_id": "Will be generated by dataset_builder/generate_answer.py"
	"study_id": "Will be generated by dataset_builder/generate_answer.py"
	"image_path": "Will be generated by dataset_builder/generate_answer.py"
}
```

## Versioning

We employ semantic versioning for our dataset, with the current version being v1.0.0. Generally, we will maintain and provide updates only for the latest version of the dataset. However, in cases where significant updates occur or when older versions are required for validating previous research, we may exceptionally retain previous dataset versions for a period of up to one year. For a detailed list of changes made in each version, check out our CHANGELOG.

## Contributing

Contributions to enhance the usability and functionality of this dataset are always welcomed. If you're interested in contributing, feel free to fork this repository, make your changes, and then submit a pull request. For significant changes, please first open an issue to discuss the proposed alterations.

## Contact

For any questions or concerns regarding this dataset, please feel free to reach out to us ([seongsu@kaist.ac.kr](mailto:seongsu@kaist.ac.kr) or [kyungdaeun@kaist.ac.kr](mailto:kyungdaeun@kaist.ac.kr)). We appreciate your interest and are eager to assist.

## Acknowledgements

More details will be provided soon.

## Citation

When you use the MIMIC-CXR-VQA dataset, we would appreciate it if you cite the following:
```
@misc{bae2023ehrxqa,
      title={EHRXQA: A Multi-Modal Question Answering Dataset for Electronic Health Records with Chest X-ray Images}, 
      author={Seongsu Bae and Daeun Kyung and Jaehee Ryu and Eunbyeol Cho and Gyubok Lee and Sunjun Kweon and Jungwoo Oh and Lei Ji and Eric I-Chao Chang and Tackeun Kim and Edward Choi},
      year={2023},
      eprint={2310.18652},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License

The code in this repository is provided under the terms of the MIT License. The final output of the dataset created using this code, the MIMIC-CXR-VQA, is subject to the terms and conditions of the original datasets from Physionet: [MIMIC-CXR-JPG License](https://physionet.org/content/mimic-cxr/view-license/2.0.0/), [Chest ImaGenome License](https://physionet.org/content/chest-imagenome/view-license/1.0.0/), and [MIMIC-IV License](https://physionet.org/content/mimiciv/view-license/2.2/).
