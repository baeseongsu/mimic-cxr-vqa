# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         train
# Description:  
# Author:       Boliu.Kelvin, Sedigheh Eslami
#-------------------------------------------------------------------------------
import os
import sys
import time
import wandb
import torch
import torch.nn as nn

# from torch.utils.tensorboard import SummaryWriter
from lib.utils import utils
from tqdm import tqdm

def compute_score_with_logits(logits, labels):
    if labels.shape[0] == 0:     # sometimes, all samples in the batch are either open or close
                                 # hence, the labels and logits is empty
        scores = torch.zeros(*labels.size()).to(logits.device)
        return scores
    func = torch.nn.Softmax(dim=1)
    logits = func(logits)
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).to(logits.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def compute_score_with_logits_multilabel(logits, labels, return_logit=False):
    logits = torch.sigmoid(logits)
    one_hots = torch.zeros(*labels.size()).to(logits.device)
    one_hots[logits > 0.5] = 1
    # scores = accuracy_score(y_true=labels.cpu(), y_pred=one_hots.cpu())
    scores = ((labels != one_hots).sum(axis=1) == 0)
    # return scores
    if return_logit:
        return scores, logits
    else:
        return scores


def get_time_stamp():
    ct = time.time()
    local_time = time.localtime(ct)
    data_head = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    data_secs = (ct - int(ct)) * 1000
    time_stamp = "%s.%03d" % (data_head, data_secs)
    return time_stamp


# Train phase
def train(cfg, model, question_model, train_loader, eval_loader, n_unique_close, device, s_opt=None, s_epoch=0):
    tblog_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "tensorboardlogs")
    if not os.path.exists(tblog_dir):
        os.makedirs(tblog_dir)
    # writer = SummaryWriter(log_dir=tblog_dir)
    base_lr = cfg.TRAIN.OPTIMIZER.BASE_LR
    momentum = cfg.TRAIN.OPTIMIZER.MOMENTUM_CNN
    model = model.to(device)
    question_model = question_model.to(device)

    # create packet for output
    utils.create_dir(cfg.OUTPUT_DIR)
    
    # for every train, create a packet for saving .pth and .log
    ckpt_path = os.path.join(cfg.OUTPUT_DIR, cfg.NAME)
    utils.create_dir(ckpt_path)
    
    # create logger
    logger = utils.Logger(os.path.join(ckpt_path, 'medVQA.log')).get_logger()
    logger.info(">>>The net is:")
    logger.info(model)

    # Adamax optimizer
    optim = torch.optim.Adamax(params=model.parameters(), lr=base_lr)

    # Loss function
    if cfg.LOSS.LOSS_TYPE == "BCELogits":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif cfg.LOSS.LOSS_TYPE == "CrossEntropy":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"{cfg.LOSS.LOSS_TYPE} loss not supported!")

    ae_criterion = torch.nn.MSELoss()

    best_eval_score = 0
    best_epoch = 0
    # Epoch passing in training phase
    print(len(train_loader))
    for epoch in range(s_epoch, cfg.TRAIN.N_EPOCH):
        total_loss = 0
        total_open_loss = 0
        total_close_loss = 0
        train_score = 0
        number=0
        open_cnt = 0
        close_cnt = 0
        model.train()
        
        # Predicting and computing score
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            v = data['image']
            q = data['question_logit']
            a = data['target']
            answer_target = data['answer_target']
            optim.zero_grad()
            if cfg.TRAIN.VISION.MAML:
                v[0] = v[0].reshape(v[0].shape[0], 84, 84).unsqueeze(1)
                v[0] = v[0].to(device)
            if cfg.TRAIN.VISION.AUTOENCODER:
                v[1] = v[1].reshape(v[1].shape[0], 128, 128).unsqueeze(1)
                v[1] = v[1].to(device)
            if cfg.TRAIN.VISION.CLIP:
                if cfg.TRAIN.VISION.CLIP_VISION_ENCODER == "RN50x4":
                    v[2] = v[2].reshape(v[2].shape[0], 3, 288, 288)
                else:
                    v[2] = v[2].reshape(v[2].shape[0], 3, 250, 250)
                v[2] = v[2].to(device)
            if cfg.TRAIN.VISION.OTHER_MODEL:
                v = v.to(device)

            q = q.to(device)
            a = a.to(device)
            if cfg.TRAIN.VISION.AUTOENCODER:
                last_output_close, last_output_open, a_close, a_open, decoder = model(v, q,a, answer_target)
            else:
                last_output_close, last_output_open, a_close, a_open = model(v, q,a, answer_target)

            preds_close, preds_open = model.classify(last_output_close, last_output_open)

            #loss
            if cfg.LOSS.LOSS_TYPE == "BCELogits":
                loss_close = criterion(preds_close.float(), a_close)
                loss_open = criterion(preds_open.float(), a_open)
            elif cfg.LOSS.LOSS_TYPE == "CrossEntropy":
                loss_close = criterion(preds_close.float(), torch.max(a_close, 1)[1])
                loss_open = criterion(preds_open.float(), torch.max(a_open, 1)[1])
            if torch.isnan(loss_open):
                assert a_open.shape[0] == 0
                loss_open = torch.tensor([0.0]).to(device)
            if torch.isnan(loss_close):
                assert a_close.shape[0] == 0
                loss_close = torch.tensor([0.0]).to(device)
            loss = loss_close + loss_open
            if cfg.TRAIN.VISION.AUTOENCODER:
                loss_ae = ae_criterion(v[1], decoder)
                loss = loss + (loss_ae * cfg.TRAIN.VISION.AE_ALPHA)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optim.step()
            
            #compute the acc for open and close
            batch_close_score = compute_score_with_logits(preds_close, a_close.data).sum()
            batch_open_score = compute_score_with_logits_multilabel(preds_open, a_open.data).sum()
            total_open_loss += loss_open.item()
            total_close_loss += loss_close.item()
            total_loss += loss.item()
            train_score += batch_close_score + batch_open_score
            number+= q.shape[0]
            open_cnt += preds_open.shape[0]
            close_cnt += preds_close.shape[0]

            if i % 2000 == 0:
                log_dict = {
                    "train/Loss": total_loss / number,
                    "train/Accuracy": 100 * train_score / number,
                }
                wandb.log(log_dict)
                # time_per_iter = (time.time() - start_time) / (epoch*len(train_loader) + i)
                # total_eta = (cfg.TRAIN.N_EPOCH*len(train_loader) - (epoch*len(train_loader) + i)) * time_per_iter
                # logger.info(f"[Igit ter]:{str(i):10s} | [eta]:{total_eta} | [Train] Loss:{total_loss / number:.3f} , Train_Acc:{100 * train_score / number:.3f}%")
       
        total_loss /= len(train_loader)
        if open_cnt != 0:
            total_open_loss /= open_cnt 
        if close_cnt != 0:
            total_close_loss /= close_cnt
        train_score = 100 * train_score / number
        logger.info('-------[Epoch]:{}-------'.format(epoch))
        logger.info('[Train] Loss:{:.6f} , Train_Acc:{:.6f}%'.format(total_loss, train_score))
        logger.info('[Train] Loss_Open:{:.6f} , Loss_Close:{:.6f}%'.format(total_open_loss, total_close_loss))
        log_dict = {
            "train/Loss": total_loss,
            "train/Loss_Open": total_open_loss, 
            "train/Loss_Close": total_close_loss,
            "train/Accuracy": train_score,
            "train/epoch": epoch
        }
        wandb.log(log_dict)

        # Evaluation
        if eval_loader is not None:
            eval_score, open_score, close_score = evaluate_classifier(model,question_model, eval_loader, cfg, n_unique_close, device, logger)
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                best_epoch = epoch
                # Save the best acc epoch
                model_path = os.path.join(ckpt_path, '{}_best.pth'.format(best_epoch))
                utils.save_model(model_path, model, best_epoch, eval_score, open_score, close_score, optim)
            logger.info('[Result] The best acc is {:.6f}% at epoch {}'.format(best_eval_score, best_epoch))
            log_dict = {
                "val/Accuracy": eval_score,
                "val/Accuracy_open": open_score, 
                "val/Accuracy_close": close_score,
                "val/epoch": epoch
            }
            wandb.log(log_dict)

        
# Evaluation
def evaluate_classifier(model,pretrained_model, dataloader, cfg, n_unique_close, device, logger):
    score = 0
    total = 0
    open_ended = 0. #'OPEN'
    score_open = 0.

    closed_ended = 0. #'CLOSED'
    score_close = 0.
    model.eval()
    
    with torch.no_grad():
        for i,data in enumerate(dataloader):
            v = data['image']
            q = data['question_logit']
            a = data['target']
            if cfg.TRAIN.VISION.MAML:
                v[0] = v[0].reshape(v[0].shape[0], 84, 84).unsqueeze(1)
                v[0] = v[0].to(device)
            if cfg.TRAIN.VISION.AUTOENCODER:
                v[1] = v[1].reshape(v[1].shape[0], 128, 128).unsqueeze(1)
                v[1] = v[1].to(device)
            if cfg.TRAIN.VISION.CLIP:
                if cfg.TRAIN.VISION.CLIP_VISION_ENCODER == "RN50x4":
                    v[2] = v[2].reshape(v[2].shape[0], 3, 288, 288)
                else:
                    v[2] = v[2].reshape(v[2].shape[0], 3, 250, 250)
                v[2] = v[2].to(device)
            if cfg.TRAIN.VISION.OTHER_MODEL:
                v = v.to(device)
            
            q = q.to(device)
            if cfg.TRAIN.QUESTION.CLIP:
                q = q.to(device)
            a = a.to(device)

            if cfg.TRAIN.VISION.AUTOENCODER:
                last_output_close, last_output_open, a_close, a_open, decoder, _, _ = model.forward_classify(v, q, a, pretrained_model, n_unique_close)
            else:
                last_output_close, last_output_open, a_close, a_open, _, _ = model.forward_classify(v, q, a, pretrained_model, n_unique_close)

            preds_close, preds_open = model.classify(last_output_close, last_output_open)
            
            batch_close_score = 0.
            batch_open_score = 0.
            if preds_close.shape[0] != 0:
                batch_close_score = compute_score_with_logits(preds_close, a_close.data).sum()
            if preds_open.shape[0] != 0: 
                batch_open_score = compute_score_with_logits_multilabel(preds_open, a_open.data).sum()

            score += batch_close_score + batch_open_score

            size = q.shape[0]
            total += size  # batch number
            
            open_ended += preds_open.shape[0]
            score_open += batch_open_score

            closed_ended += preds_close.shape[0]
            score_close += batch_close_score

    try:
        score = 100* score / total
    except ZeroDivisionError:
        score = 0
    try:
        open_score = 100* score_open/ open_ended
    except ZeroDivisionError:
        open_score = 0
    try:
        close_score = 100* score_close/ closed_ended
    except ZeroDivisionError:
        close_score = 0
    print(total, open_ended, closed_ended)
    logger.info('[Validate] Val_Acc:{:.3f}%  |  Open_ACC:{:.3f}%   |  Close_ACC:{:.3f}%' .format(score,open_score,close_score))
    return score, open_score, close_score
