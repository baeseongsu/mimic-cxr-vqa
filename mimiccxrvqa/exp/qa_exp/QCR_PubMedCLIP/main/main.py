# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         main
# Description:  the entrance of procedure
#-------------------------------------------------------------------------------

import os
import sys
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
import torch
import random
import argparse
import numpy as np

from train import train
from test import test

from torch.utils.data import DataLoader
from lib.config import cfg, update_config
from lib.dataset import *
from lib.utils.create_dictionary import Dictionary
from lib.BAN.multi_level_model import BAN_Model
from lib.language.classify_question import classify_model
torch.autograd.set_detect_anomaly(True)

def parse_args():
    parser = argparse.ArgumentParser(description="Med VQA")
    # cfg
    parser.add_argument(
            "--cfg",
            help="decide which cfg to use",
            required=False,
            default="/home/test.yaml",
            type=str,
            )
    # GPU config
    parser.add_argument('--gpu', type=int, default=0,
                        help='use gpu device. default:0')
    parser.add_argument('--test', type=bool, default=False, help='Test or train.')
    parser.add_argument("--wandb_project", type=str, default="phase1-qa-dataset")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    torch.cuda.empty_cache()
    print(os.getcwd())
    args = parse_args()
    device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")
    args.device = device
    update_config(cfg, args)
    data_dir = cfg.DATASET.DATA_DIR
    args.data_dir = data_dir

    # Fixed random seed
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.autograd.set_detect_anomaly(True)

    # Logging
    if not args.test:
        print("Wandb init...")
        wandb_username = "ehr-vqg"
        wandb.init(
            project=args.wandb_project,
            entity=wandb_username,
            name=cfg.NAME,
        )
        wandb.config.update(cfg)

    # load the model
    d = Dictionary.load_from_file(data_dir + '/dictionary.pkl')
    question_classify = classify_model(d.ntoken, data_dir + '/glove6b_init_300d.npy')
    ckpt = os.path.join(os.path.dirname(os.path.dirname(__file__)), './checkpoints/type_classifier_mimiccxr.pth')
    pretrained_model = torch.load(ckpt, map_location='cpu')['model_state']
    del pretrained_model
    
    # training phase
    # create VQA model and question classify model
    if args.test:
        val_dataset = VQAMIMICCXRFeatureDataset('valid', cfg, d, dataroot=data_dir)
        test_dataset = VQAMIMICCXRFeatureDataset('test', cfg, d, dataroot=data_dir)

        drop_last = False
        drop_last_val = False 
        val_loader = DataLoader(val_dataset, cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=2,drop_last=drop_last_val, pin_memory=True)
        test_loader = DataLoader(test_dataset, cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=2,drop_last=drop_last_val, pin_memory=True)
        model = BAN_Model(val_dataset, cfg, device)
        model_data = torch.load(cfg.TEST.MODEL_FILE)
        model.load_state_dict(model_data.get('model_state', model_data), strict=False)
        test(cfg, model, question_classify, val_loader, val_dataset.num_close_candidates, args.device, "valid")
        test(cfg, model, question_classify, test_loader, val_dataset.num_close_candidates, args.device, "test")
    else:
        train_dataset = VQAMIMICCXRFeatureDataset('train', cfg, d, dataroot=data_dir)
        val_dataset = VQAMIMICCXRFeatureDataset('valid', cfg, d, dataroot=data_dir)

        drop_last = False
        drop_last_val = False 
        train_loader = DataLoader(train_dataset, cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=2,drop_last=drop_last, pin_memory=True)
        val_loader = DataLoader(val_dataset, cfg.TEST.BATCH_SIZE, shuffle=True, num_workers=2,drop_last=drop_last_val, pin_memory=True)
        model = BAN_Model(train_dataset, cfg, device)
        train(cfg, model, question_classify, train_loader, val_loader, train_dataset.num_close_candidates, args.device)