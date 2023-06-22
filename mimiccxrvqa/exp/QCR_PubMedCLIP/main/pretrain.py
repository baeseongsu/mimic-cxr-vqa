import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import clip
import json
import wandb
import logging
import argparse
import numpy as np

from train import train
from lib.dataset import *
from lib.utils import utils
from lib.utils.utils import get_optimizer
from lib.core.function import valid_model
from lib.core.evaluate import AverageMeter
from lib.pretrain_config import cfg, update_config

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

def create_logger(cfg):
    dataset = cfg.DATASET.DATASET
    log_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    time_str = time.strftime("%Y-%m-%d-%H-%M")
    log_name = f"{dataset}_{time_str}.log"
    log_file = os.path.join(log_dir, log_name)
    # set up logger
    print("=> creating log {}".format(log_file))
    head = "%(asctime)-15s %(message)s"
    logging.basicConfig(filename=str(log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger("").addHandler(console)

    logger.info("---------------------Cfg is set as follow--------------------")
    logger.info(cfg)
    logger.info("-------------------------------------------------------------")
    return logger, log_file

# Train phase
def train(cfg, train_loader, eval_loader, device):
    tblog_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "tensorboardlogs")
    if not os.path.exists(tblog_dir):
        os.makedirs(tblog_dir)

    # create logger
    logger, _ = create_logger(cfg)
    model_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    logger.info(f"-------Loading CLIP with vision encoder {cfg.TRAIN.VISION_ENCODER} -------")
    model, preprocess = clip.load(cfg.TRAIN.VISION_ENCODER, device=device, jit=False)
    if device == "cpu":
          model.float()
    else :
        clip.model.convert_weights(model)

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    optim = get_optimizer(cfg, model)

    best_loss, best_epoch, best_model = 10000, 0, ""

    logger.info("-------Training started-------")

    for epoch in range(cfg.TRAIN.N_EPOCH):
        train_all_loss = AverageMeter()
        model.train()
        model_save_path = os.path.join(
                model_dir,
                f"epoch_{epoch}.pth")
        number_batch = len(train_loader)

        # Predicting and computing score
        for i, (image, caption) in enumerate(train_loader):
            optim.zero_grad()
            images = torch.stack([img for img in image], dim=0).to(device)
            captions = clip.tokenize(caption, context_length=cfg.TRAIN.MAX_SEQ_LENGTH, truncate = True).to(device)

            logits_per_image, logits_per_text = model(images, captions)
            logits_per_image *= (np.exp(0.01) / np.exp(0.07))
            logits_per_text *= (np.exp(0.01) / np.exp(0.07))

            ground_truth = torch.arange(cfg.TRAIN.BATCH_SIZE, dtype=torch.long, device=device)
            lambdaa = 0.5
            train_total_loss = lambdaa*(loss_img(logits_per_image, ground_truth)) + (1-lambdaa)* (loss_txt(logits_per_text, ground_truth))
            train_total_loss.backward()
            if device == "cpu":
                optim.step()
            else : 
                _convert_models_to_fp32(model)
                optim.step()
                clip.model.convert_weights(model)
            if i % cfg.SHOW_STEP == 0:
                pbar_str = "Epoch:{:>3d}  Batch:{:>3d}/{}  Batch_Loss:{:>5.3f}  ".format(epoch, i, number_batch, train_total_loss)
                logger.info(pbar_str)
                wandb.log({
                'Loss/Batch' : train_total_loss,
                })

            cnt = len(caption)
            train_all_loss.update(train_total_loss.data.item(), cnt)

        train_all_loss = train_all_loss.avg
        pbar_str = f"---Epoch:{epoch}  Epoch_Loss:{train_all_loss}"
        logger.info(pbar_str)
        wandb.log({
                'Loss/train' : train_all_loss,
                'epoch' : epoch,
                })
        # Eval
        if eval_loader is not None:
            eval_all_loss = valid_model(eval_loader, model, loss_img, loss_txt, cfg, device)
            if eval_all_loss < best_loss:
                best_loss = eval_all_loss
                best_epoch = epoch
                torch.save({
                    'best_epoch': best_epoch,
                    'state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'best_loss': best_loss,
                    }, os.path.join(model_dir, "best_model.pth")
                    )
            logger.info(
                    f"--------------Epoch:{epoch}    Eval_Loss:{eval_all_loss}%--------------")
            logger.info(
                    f"--------------Best_Epoch:{best_epoch}    Best_Eval_Loss:{best_loss}%--------------")
            wandb.log({
                'Loss/val' : eval_all_loss,
                'epoch' : epoch,
                })
        else:
            if train_all_loss < best_loss:
                best_loss = train_all_loss
                best_epoch = epoch
                torch.save({
                    'best_epoch': best_epoch,
                    'state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'best_loss': best_loss,
                    }, os.path.join(model_dir, "best_model.pth")
                    )
            logger.info(
                    f"--------------Best_Epoch:{best_epoch}    Best_Train_Loss:{best_loss}%--------------")
            wandb.log({
                'Loss/train-as-val' : train_all_loss,
                'epoch' : epoch,
                })

        if not os.path.exists(cfg.RESULTS_DIR):
            os.makedirs(cfg.RESULTS_DIR)

        with open(os.path.join(cfg.RESULTS_DIR, "best.json"), "w") as f:
            json.dump({"best_epoch": best_epoch, "best_loss": best_loss}, f)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune CLIP with a medical dataset.")
    # cfg
    parser.add_argument(
            "--cfg",
            help="decide which cfg to use",
            required=False,
            default="/home/test.yaml",
            type=str,
            )
    # GPU config
    parser.add_argument('--seed', type=int, default=5
                        , help='random seed for gpu.default:5')
    parser.add_argument('--gpu', type=int, default=0,
                        help='use gpu device. default:0')


    args = parser.parse_args()
    return args


if __name__ == '__main__':

    data = cfg.DATASET.DATA_DIR
    wandb.init(project='MIMIV-VQA-QCR', entity='ehr-vqg', name='MIMIC-CLIP-pretraining')
    wandb.config.update(cfg)
    args = parse_args()
    args.data_dir = data
    # set GPU device
    device = torch.device("cuda" if args.gpu >= 0 else "cpu")
    update_config(cfg, args)
    # Fixed random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    # prepare the dataloader
#     print(cfg)
    
    train_dataset = ImageTextDataset("train", cfg)
    train_loader = DataLoader(train_dataset, cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)
    val_dataset = ImageTextDataset("val", cfg)
    val_loader = DataLoader(val_dataset, cfg.TEST.BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)

    # training phase
    train(cfg, train_loader, val_loader, device)

