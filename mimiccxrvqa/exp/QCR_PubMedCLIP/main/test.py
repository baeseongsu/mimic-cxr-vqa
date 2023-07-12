# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         train
# Description:  
# Author:       Boliu.Kelvin, Sedigheh Eslami
#-------------------------------------------------------------------------------
import os
import time
import torch
from lib.utils import utils
import pandas as pd
from sklearn.metrics import accuracy_score


def compute_score_with_logits(logits, labels):
    func = torch.nn.Softmax(dim=1)
    logits = func(logits)
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).to(logits.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores, logits

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
def test(cfg, model, question_model, eval_loader, n_unique_close, device, phase=".", s_opt=None, s_epoch=0):
    model = model.to(device)
    question_model = question_model.to(device)
    utils.create_dir(os.path.join(cfg.TEST.RESULT_DIR, phase))

    # Evaluation
    eval_score, open_score, close_score = evaluate_classifier(model,question_model, eval_loader, cfg, n_unique_close, device, cfg.TEST.RESULT_DIR)

        
# Evaluation
def evaluate_classifier(model,pretrained_model, dataloader, cfg, n_unique_close, device, result_dir):
    score = 0
    total = 0
    open_ended = 0. #'OPEN'
    score_open = 0.

    closed_ended = 0. #'CLOSED'
    score_close = 0.
    model.eval()
    
    correct_results = {"image_name": [], "question": [], "answer": [], "answer_type": [], "predicted_answer_type": [], "content_type": []}
    incorrect_results = {"image_name": [], "question": [], "answer": [], "answer_type": [], "predicted_answer": [], "predicted_answer_type": [], "content_type": []}

    with torch.no_grad():
    # (v, q, a,answer_type, question_type, phrase_type, answer_target, image_name, question_text, answer_text)
        for i, data in enumerate(dataloader):
            v = data['image']
            q = data['question_logit']
            a = data['target']
            image_name = data["image_name"]
            question_text = data["question_text"]
            answer_type = data["answer_type"]
            content_type = data["content_type"]

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
                last_output_close, last_output_open, a_close, a_open, decoder, indexs_open, indexs_close = model.forward_classify(v, q, a, pretrained_model, n_unique_close)
            else:
                last_output_close, last_output_open, a_close, a_open, indexs_open, indexs_close = model.forward_classify(v, q, a, pretrained_model, n_unique_close)

            preds_close, preds_open = model.classify(last_output_close, last_output_open)
            
            batch_close_score = 0.
            batch_open_score = 0.
            if preds_close.shape[0] != 0:
                batch_close_score_temp, close_logits = compute_score_with_logits(preds_close, a_close.data)
                close_correct = (batch_close_score_temp == 1).nonzero(as_tuple=True)[0].tolist()
                batch_close_score = batch_close_score_temp.sum()
            else:
                close_correct = None

            if preds_open.shape[0] != 0: 
                batch_open_score_temp, open_logits = compute_score_with_logits_multilabel(preds_open, a_open.data, return_logit=True)
                open_correct = (batch_open_score_temp == 1).nonzero(as_tuple=True)[0].tolist()
                batch_open_score = batch_open_score_temp.sum()
            else:
                open_correct = None

            score += batch_close_score + batch_open_score

            size = q.shape[0]
            total += size  # batch number
            
            open_ended += preds_open.shape[0]
            score_open += batch_open_score

            closed_ended += preds_close.shape[0]
            score_close += batch_close_score

            # assert len(indexs_close) + len(indexs_open) == len(image_name)
            # assert len(close_correct) + len(open_correct) <= len(image_name)  # batch size
            # preds_close_list.append(preds_close)
            # preds_open_list.append(preds_open)
            
            close_incorrect = [i for i in range(len(indexs_close)) if i not in close_correct]
            open_incorrect = [i for i in range(len(indexs_open)) if i not in open_correct]
            if close_correct is not None:
                for i in close_correct:
                    ind = indexs_close[i]
                    correct_results["image_name"].append(image_name[ind])
                    correct_results["question"].append(question_text[ind])
                    correct_results["predicted_answer_type"].append("CLOSED")
                    correct_results["answer_type"].append(answer_type[ind])
                    correct_results["content_type"].append(content_type[ind])
                for ind in close_incorrect:
                    incorrect_results["image_name"].append(image_name[ind])
                    incorrect_results["question"].append(question_text[ind])
                    incorrect_results["predicted_answer"].append(close_logits[ind].cpu())
                    incorrect_results["predicted_answer_type"].append("CLOSED")
                    incorrect_results["answer_type"].append(answer_type[ind])
                    incorrect_results["content_type"].append(content_type[ind])
            if open_correct is not None:
                for i in open_correct:
                    ind = indexs_open[i]
                    correct_results["image_name"].append(image_name[ind])
                    correct_results["question"].append(question_text[ind])
                    correct_results["predicted_answer_type"].append("OPEN")
                    correct_results["answer_type"].append(answer_type[ind])
                    correct_results["content_type"].append(content_type[ind])
                for ind in open_incorrect:
                    incorrect_results["image_name"].append(image_name[ind])
                    incorrect_results["question"].append(question_text[ind])
                    incorrect_results["predicted_answer"].append(open_logits[ind].cpu())
                    incorrect_results["predicted_answer_type"].append("OPEN")
                    incorrect_results["answer_type"].append(answer_type[ind])
                    incorrect_results["content_type"].append(content_type[ind])

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
    print('[Validate] Val_Acc:{:.6f}%  |  Open_ACC:{:.6f}%   |  Close_ACC:{:.6f}%' .format(score,open_score,close_score))
    df = pd.DataFrame(correct_results)
    df.to_csv(f"{result_dir}/correct_predictions.csv", index=False)
    inc_df = pd.DataFrame(incorrect_results)
    inc_df.to_csv(f"{result_dir}/incorrect_predictions.csv", index=False)
    return score, open_score, close_score
