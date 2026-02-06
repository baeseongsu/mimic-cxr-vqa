# MIMIC-CXR-VQA

[![License:Physionet](https://img.shields.io/badge/License-Physionet-red.svg)](https://physionet.org/content/mimic-ext-mimic-cxr-vqa/1.0.0/)
![GitHub release](https://img.shields.io/github/release/baeseongsu/mimic-cxr-vqa.svg)
![GitHub last commit](https://img.shields.io/github/last-commit/baeseongsu/mimic-cxr-vqa.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

*A new collection of medical visual question answering dataset on MIMIC-CXR database*

## Overview

The MIMIC-CXR-VQA dataset is a complex (involving set and logical operations), diverse (with 48 templates), and large-scale (approximately 377K) resource, designed specifically for Visual Question Answering (VQA) tasks in the medical domain. Primarily focusing on chest radiographs, this dataset was mainly derived from the MIMIC-CXR-JPG and Chest ImaGenome datasets, both of which were sourced from Physionet.

The goal of the MIMIC-CXR-VQA dataset is to serve as a benchmark for evaluating the effectiveness of current medical VQA approaches. It not only functions as a tool for traditional medical VQA tasks but also has the unique quality of being an image-based Electronic Health Records (EHRs) Question Answering dataset resource. Therefore, we utilize question templates from the MIMIC-CXR-VQA dataset as seed question templates for image modality, to construct a multi-modal EHR QA dataset, [EHRXQA](https://github.com/baeseongsu/ehrxqa).

## Updates

- [07/20/2024] We released [MIMIC-CXR-VQA dataset](https://physionet.org/content/mimic-ext-mimic-cxr-vqa/1.0.0/) on Physionet.
- [12/12/2023] We presented our research work at NeurIPS 2023 Datasets and Benchmarks Track as a [poster](https://neurips.cc/virtual/2023/poster/73600).
- [10/28/2023] We released our research paper on [arXiv](https://arxiv.org/abs/2310.18652).

## Reproducing the Dataset

### Prerequisites

- **Python 3.12+**
- **PhysioNet credentialed account** with signed DUAs for:
  - [MIMIC-CXR-JPG v2.0.0](https://physionet.org/sign-dua/mimic-cxr-jpg/2.0.0/)
  - [Chest ImaGenome v1.0.0](https://physionet.org/sign-dua/chest-imagenome/1.0.0/)
  - [MIMIC-IV v2.2](https://physionet.org/sign-dua/mimiciv/2.2/)

<details>
<summary><b>New to PhysioNet?</b> Click to see credentialing instructions</summary>

1. [Register for a PhysioNet account](https://physionet.org/register/)
2. Follow the [credentialing instructions](https://physionet.org/credential-application/)
3. Complete the [CITI Data or Specimens Only Research training course](https://physionet.org/about/citi-course/)
4. Sign the DUA for each required dataset (links above)

</details>

### Environment Setup

<details>
<summary><b>Using UV (Recommended)</b></summary>

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/baeseongsu/mimic-cxr-vqa.git
cd mimic-cxr-vqa

# Create environment and install dependencies
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install pandas tqdm scikit-learn
```

</details>

<details>
<summary><b>Using Conda</b></summary>

```bash
# Clone repository
git clone https://github.com/baeseongsu/mimic-cxr-vqa.git
cd mimic-cxr-vqa

# Create environment and install dependencies
conda create -n mimiccxrvqa python=3.12
conda activate mimiccxrvqa
pip install pandas tqdm scikit-learn
```

</details>

### Running the Reproduction Script

**Option 1: Build from pre-downloaded datasets**

If you have already downloaded MIMIC-CXR, MIMIC-IV, and Chest ImaGenome, ensure the directory paths in the script match your local setup, then run:

```bash
bash build_dataset.sh
```

**Option 2: Download and build**

To download source datasets from PhysioNet and generate the dataset:

```bash
bash download_and_build_dataset.sh
```

When prompted, enter your PhysioNet credentials (password will not be displayed).

**What these scripts do:**
1. Download source datasets from PhysioNet (MIMIC-CXR-JPG, Chest ImaGenome, MIMIC-IV)
2. Preprocess the datasets
3. Generate complete MIMIC-CXR-VQA dataset with ground-truth answers and metadata

<details>
<summary><b>Downloading Images Only</b></summary>

To download only the CXR images relevant to MIMIC-CXR-VQA (rather than all MIMIC-CXR-JPG images):

```bash
bash download_images.sh
```

This script reads image paths from the dataset JSON files and downloads only the required images from PhysioNet.

</details>

<details>
<summary><b>Dataset Structure</b></summary>

```
mimiccxrvqa/
└── dataset/
    ├── ans2idx.json          # Answer to index mapping
    ├── _train_part1.json     # Pre-release (without answers)
    ├── _train_part2.json     # Pre-release (without answers)
    ├── _valid.json           # Pre-release (without answers)
    ├── _test.json            # Pre-release (without answers)
    ├── train.json            # Generated after running script
    ├── valid.json            # Generated after running script
    └── test.json             # Generated after running script
```

Pre-release files (`_*.json`) are intentionally incomplete to safeguard privacy. Complete files with answers and metadata are generated after running the reproduction script with valid PhysioNet credentials.

</details>

<details>
<summary><b>Dataset Schema</b></summary>

Each QA sample is a JSON object with the following fields:

**Core Fields:**
- `split`: Dataset split (train/valid/test)
- `idx`: Instance index
- `image_id`: Associated image ID
- `question`: Natural language question
- `answer`: Answer string (generated by script)

**Template Fields:**
- `content_type`: Content category (anatomy, attribute, presence, abnormality, plane, gender, size)
- `semantic_type`: Question type (verify, choose, query)
- `template`: Question template
- `template_program`: Program to generate answer from database
- `template_arguments`: Template argument values (object, attribute, category, viewpos, gender)

**Metadata (generated by script):**
- `subject_id`: Patient ID
- `study_id`: Study ID
- `image_path`: Image file path

**Example:**
```json
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
        "object": {"0": "left lower lung zone"},
        "attribute": {"0": "pleural effusion", "1": "lung cancer"}
    },
    "answer": "no",
    "subject_id": "10000032",
    "study_id": "50414267",
    "image_path": "files/p10/p10000032/s50414267/34c81443-5a19ccad-7b5e431c-4e1dbb28-42a325c0.jpg"
}
```

</details>

## Version

**Current:** v1.0.0

This project uses semantic versioning. For detailed changes, see [CHANGELOG](CHANGELOG.md).

## Citation

When you use the MIMIC-CXR-VQA dataset, we would appreciate it if you cite the following:

```bibtex
@article{bae2024ehrxqa,
  title={EHRXQA: A multi-modal question answering dataset for electronic health records with chest x-ray images},
  author={Bae, Seongsu and Kyung, Daeun and Ryu, Jaehee and Cho, Eunbyeol and Lee, Gyubok and Kweon, Sunjun and Oh, Jungwoo and Ji, Lei and Chang, Eric and Kim, Tackeun and others},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```

## License

The code in this repository is provided under the terms of the [MIT License](LICENSE). The final output dataset (MIMIC-CXR-VQA) is subject to the terms and conditions of the original datasets from Physionet: [MIMIC-CXR-JPG License](https://physionet.org/content/mimic-cxr/view-license/2.0.0/), [Chest ImaGenome License](https://physionet.org/content/chest-imagenome/view-license/1.0.0/), and [MIMIC-IV License](https://physionet.org/content/mimiciv/view-license/2.2/).

## Contact

For questions or concerns regarding this dataset, please contact:
- Seongsu Bae ([seongsu@kaist.ac.kr](mailto:seongsu@kaist.ac.kr))
- Daeun Kyung ([kyungdaeun@kaist.ac.kr](mailto:kyungdaeun@kaist.ac.kr))
