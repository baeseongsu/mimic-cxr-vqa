# M3AE 
This is the implementation of Multi-Modal Masked Autoencoders for Medical Vision-and-Language Pre-Training, presented at MICCAI 2022, specifically designed for the MIMIC-CXR-VQA dataset.

## Requirements
Install the required packages by running:
```
pip install -r requirements.txt
```

## Dataset Preparation
To construct the MIMIC-CXR-VQA dataset, follow the instructions provided in the main [README.md](../../../README.md) of this project.

## Data Pre-processing
### Pre-training Data Preprocessing
Prepare the pre-training data with the following command:
```
python prepro/prepro_pretraining_data.py
```

### Fine-tuning Data Preprocessing
Prepare the fine-tuning data with this command:

```
python prepro/prepro_finetuning_data.py --data_root <mimiccxrvqa_dir> --image_root <mimiccxrjpg_dir> --save_root <finetune_arrow_save_root>
```

## Pre-training Model
Execute the pre-training phase using the below command, configured with our experimental hyperparameters:

```
python main.py \
 with data_root=<pretraining_data_path> \
 num_gpus=8 num_nodes=1 \
 task_pretrain_m3ae \
 per_gpu_batchsize=32 \
 clip16 text_roberta \
 image_size=288 \
 max_text_len=64 \
 tokenizer=roberta-base
```


## Fine-tuning Model
For fine-tuning, use the following command, set with our experimental hyperparameters:

```
python main.py with data_root=<fintuning_data_path> \
 num_gpus=4 num_nodes=1 \
 task_finetune_vqa_mmehr \
 per_gpu_batchsize=16 \
 clip16 text_roberta \
 image_size=512 \
 tokenizer=roberta-base \
 load_path=<pretraining_ckpt_path> \
 seed=0
```

## Acknowledgements
This implementation uses code from following repository:
- [Official M3AE implementation](https://github.com/zhjohnchan/M3AE/tree/master)

We thank the authors for their open-sourced code. <br /><br />
