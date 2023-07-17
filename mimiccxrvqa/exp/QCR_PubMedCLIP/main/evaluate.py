import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import torch
import random
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch.utils.data import DataLoader
from lib.config import cfg, update_config
from lib.dataset import *
from lib.utils import utils
from lib.utils.create_dictionary import Dictionary
from lib.BAN.multi_level_model import BAN_Model
from lib.language.classify_question import classify_model
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
torch.autograd.set_detect_anomaly(True)

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
def test(cfg, model, question_model, eval_loader, n_unique_close, device, phase="", s_opt=None, s_epoch=0):
    model = model.to(device)
    question_model = question_model.to(device)
    utils.create_dir(os.path.join(cfg.TEST.RESULT_DIR, phase))

    # Evaluation
    evaluate_classifier(model,question_model, eval_loader, cfg, n_unique_close, device, 
                        os.path.join(cfg.TEST.RESULT_DIR, phase))

        
# Evaluation
def evaluate_classifier(model,pretrained_model, dataloader, cfg, n_unique_close, device, result_dir):
    score = 0
    total = 0
    open_ended = 0. #'OPEN'
    score_open = 0.

    closed_ended = 0. #'CLOSED'
    score_close = 0.
    model.eval()
    
    # 1. type wise accuracy
    # 2. category wise accuracy
    # 3. micro acc / fq
    # 4. macro acc
    
    gts_list = []
    preds_list = []
    results = []

    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
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
                batch_close_score = batch_close_score_temp.sum()

            if preds_open.shape[0] != 0: 
                batch_open_score_temp, open_logits = compute_score_with_logits_multilabel(preds_open, a_open.data, return_logit=True)
                batch_open_score = batch_open_score_temp.sum()

            score += batch_close_score + batch_open_score

            size = q.shape[0]
            total += size  # batch number
            
            open_ended += preds_open.shape[0]
            score_open += batch_open_score

            closed_ended += preds_close.shape[0]
            score_close += batch_close_score
            
            predictions = torch.zeros(*a.size()).to(a.device)
            if preds_open.shape[0] != 0: 
                open_logits = torch.where(open_logits > 0.5, 1, 0)
                
            for idx in range(size):
                results.append({
                    "image_name": image_name[idx],
                    "question": question_text[idx],
                    "answer_type": answer_type[idx],
                    "content_type": content_type[idx],
                })
                if idx in indexs_close:
                    predictions[idx][close_logits[indexs_close.index(idx)]] = 1
                elif idx in indexs_open:
                    predictions[idx][dataloader.dataset.num_close_candidates:] = open_logits[indexs_open.index(idx)]
            preds_list.append(predictions.data)
            gts_list.append(a.data)    

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
        
    total_preds = torch.cat(preds_list).cpu()
    total_gts = torch.cat(gts_list).cpu()
    results = pd.DataFrame(results)
    torch.save({
        "total_preds": total_preds,
        "total_gts": total_gts
    }, os.path.join(result_dir, f"{cfg.DATASET.DATASET_TYPE}{dataloader.dataset.phase}.pth"))
    results.to_csv(os.path.join(result_dir, f"{cfg.DATASET.DATASET_TYPE}{dataloader.dataset.phase}.csv"))

    if cfg.DATASET.DATASET_TYPE == "ref":
        pairwise_result_df = {}
        eval_df = json.load(open(os.path.join(os.path.dirname(cfg.DATASET.DATA_DIR), f"{dataloader.dataset.phase}_ref.json")))
        eval_df = pd.DataFrame(eval_df)
        
        obj_att_pairs = sorted(set(eval_df.groupby(["object", "attribute"]).image_id.count().index.unique()))
        closed_ans_idx = dataloader.dataset.ans2label['yes']
        total_gts = total_gts[:, closed_ans_idx]
        total_preds = total_preds[:, closed_ans_idx]

        for (obj, att) in tqdm(obj_att_pairs):
            _key = f"{obj}_{att}"
            pairwise_result_df[_key] = {}
            _y_true = total_gts[(eval_df.object == obj) & (eval_df.attribute == att)]
            _y_pred = total_preds[(eval_df.object == obj) & (eval_df.attribute == att)]
            
            if len(np.unique(_y_true)) == 1:
                pairwise_result_df[_key]["acc"] = -1
                pairwise_result_df[_key]["f1"] = -1
                pairwise_result_df[_key]["auroc"] = -1
            else:
                acc = accuracy_score(np.array(_y_true), np.array(_y_pred)>=0.5)
                f1 = f1_score(np.array(_y_true), np.array(_y_pred)>=0.5)
                auroc = roc_auc_score(_y_true, _y_pred)
                pairwise_result_df[_key]["acc"] = acc
                pairwise_result_df[_key]["f1"] = f1
                pairwise_result_df[_key]["auroc"] = auroc
            pairwise_result_df[_key]["support"] = len(_y_true)
            
        pairwise_result_df = pd.DataFrame(pairwise_result_df).T
        pd.set_option('display.max_rows', None)
        with open(os.path.join(result_dir, f"{cfg.DATASET.DATASET_TYPE}{dataloader.dataset.phase}_ref_log.txt"), 'a') as f:
            datetime_string = get_time_stamp()
            f.write("-" * 100 + "\n")
            f.write('%s : logging start' % (datetime_string) + "\n")
            f.write(f"{str(pairwise_result_df)}" + "\n")
            f.write(f"ACC (pairwise acc)    : {pairwise_result_df[pairwise_result_df.acc != -1].acc.mean() * 100}")
            f.write(f"AUROC (pairwise auroc): {pairwise_result_df[pairwise_result_df.auroc != -1].auroc.mean()}")
            f.write(f"F1 (pairwise f1)      : {pairwise_result_df[pairwise_result_df.f1 != -1].f1.mean()}")
    else:
        reports = classification_report(total_gts, total_preds, target_names=dataloader.dataset.ans2label.keys())
        acc = accuracy_score(total_gts, total_preds)
        micro_f1 = f1_score(total_gts, total_preds, average='micro')
        macro_f1 = f1_score(total_gts, total_preds, average='weighted')

        with open(os.path.join(result_dir, f"{cfg.DATASET.DATASET_TYPE}{dataloader.dataset.phase}_log.txt"), 'a') as f:
            datetime_string = get_time_stamp()
            f.write("-" * 100 + "\n")
            f.write('%s : logging start' % (datetime_string) + "\n")
            f.write(f"classification report\n")
            f.write(f"{reports}\n")
            f.write(f"score_best_epoch (micro)   | " + "{:.4f}".format(acc * 100) + "\n")
            f.write(f"micro_f1_score_best_epoch  | " + "{:.4f}".format(micro_f1) + "\n")
            f.write(f"macro_f1_score_best_epoch  | " + "{:.4f}".format(macro_f1) + "\n\n")

            cat_result_df = {_cat: {} for _cat in results.content_type.unique()}
            for _cat in sorted(results.content_type.unique()):
                _y_true = total_gts[(results.content_type == _cat)]
                _y_pred = total_preds[(results.content_type == _cat)]
                try:
                    acc = accuracy_score(_y_true, _y_pred >= 0.5)
                    f1 = f1_score(np.array(_y_true), np.array(_y_pred) >= 0.5, average="micro")
                except:
                    import pdb; pdb.set_trace()

                cat_result_df[_cat]["acc"] = acc * 100
                cat_result_df[_cat]["f1"] = f1
            cat_result_df = pd.DataFrame(cat_result_df).T
            f.write(f"category-wise acc" + "\n")
            f.write(str(cat_result_df) + "\n")
            f.write('[{}] Acc:{:.4f}%  |  Open_ACC:{:.4f}%  |  Close_ACC:{:.4f}%' .format(dataloader.dataset.phase, score,open_score,close_score))

    print(total, open_ended, closed_ended)
    print('[Validate] Val_Acc:{:.4f}%  |  Open_ACC:{:.4f}%   |  Close_ACC:{:.4f}%' .format(score, open_score, close_score))
    return score, open_score, close_score


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
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    torch.cuda.empty_cache()

    args = parse_args()
    device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")
    args.device = device
    update_config(cfg, args)
    data_dir = cfg.DATASET.DATA_DIR
    img_root = cfg.DATASET.IMG_DIR
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

    # load the model
    d = Dictionary.load_from_file(data_dir + '/dictionary.pkl')
    question_classify = classify_model(d.ntoken, data_dir + '/glove6b_init_300d.npy')
    ckpt = os.path.join(os.path.dirname(os.path.dirname(__file__)), './checkpoints/type_classifier_mimiccxr.pth')
    pretrained_model = torch.load(ckpt, map_location='cpu')['model_state']
    question_classify.load_state_dict(pretrained_model)
    del pretrained_model
    
    # create VQA model and question classify model
    val_dataset = VQAMIMICCXRFeatureDataset('valid', cfg, d, dataroot=data_dir, imgroot=img_root)
    test_dataset = VQAMIMICCXRFeatureDataset('test', cfg, d, dataroot=data_dir, imgroot=img_root)
    
    drop_last = False
    drop_last_val = False 
    val_loader = DataLoader(val_dataset, cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=2,drop_last=drop_last_val, pin_memory=True)
    test_loader = DataLoader(test_dataset, cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=2,drop_last=drop_last_val, pin_memory=True)
    model = BAN_Model(val_dataset, cfg, device)
    model_data = torch.load(cfg.TEST.MODEL_FILE)
    model.load_state_dict(model_data.get('model_state', model_data), strict=False)
    test(cfg, model, question_classify, val_loader, val_dataset.num_close_candidates, args.device, "valid")
    test(cfg, model, question_classify, test_loader, val_dataset.num_close_candidates, args.device, "test")