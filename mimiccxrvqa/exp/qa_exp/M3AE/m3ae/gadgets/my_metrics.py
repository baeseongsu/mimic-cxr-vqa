import os
import torch
import pickle
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import sklearn.metrics as sklm

from pytorch_lightning.metrics import Metric
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, f1_score

warnings.filterwarnings('ignore') 


class Accuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )
        preds = logits.argmax(dim=-1)
        preds = preds[target != -100]
        target = target[target != -100]
        if target.numel() == 0:
            return 1

        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct / self.total


class Scalar(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)
        self.scalar += scalar
        self.total += 1

    def compute(self):
        return self.scalar / self.total


class VQAScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().float().to(self.score.device),
            target.detach().float().to(self.score.device),
        )
        logits = torch.max(logits, 1)[1]
        one_hots = torch.zeros(*target.size()).to(target)
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        scores = one_hots * target

        self.score += scores.sum()
        self.total += len(logits)

    def compute(self):
        return self.score / self.total


class MMEHRScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("y_trues", default=[], dist_reduce_fx="cat")
        self.add_state("y_preds", default=[], dist_reduce_fx="cat")
        self.add_state("iid", default=[], dist_reduce_fx="cat")
        self.add_state("qid", default=[], dist_reduce_fx="cat")

        # category_wise score
        self.add_state("condition_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("position_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("attribute_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("abnormality_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("size_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("plane_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("gender_score", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.add_state("condition_total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("position_total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("attribute_total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("abnormality_total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("size_total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("plane_total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("gender_total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.categories = ["condition", "position", "attribute", "abnormality", "size", "plane", "gender"]
        self.best_score = 0
        self.best_micro_f1_score = 0
        self.best_macro_f1_score = 0
        self.best_score_per_category = {k:0 for k in self.categories}

    def update(self, logits, target, types=None, iid=None, qid=None):
        logits, target = (
            logits.detach().float().to(self.score.device),
            target.detach().float().to(self.score.device),
        )

        logits = torch.sigmoid(logits)
        one_hots = torch.zeros(*target.size()).to(target.device)
        one_hots[logits > 0.5] = 1

        # Hard scores
        scores = torch.all(target.cpu() == one_hots.cpu(), axis=1)
        # # Soft scores
        # one_hots.scatter_(1, logits.view(-1, 1), 1)
        # scores = one_hots * target
        
        # score per question_type
        if types is not None:
            types = np.array(types)
            for _category in self.categories:
                _score = getattr(self, f"{_category}_score")
                _score += scores[types == _category].sum()
                _total = getattr(self, f"{_category}_total")
                _total += len(scores[types == _category])

        self.score += scores.sum()
        self.total += len(scores)
        self.y_trues.append(target)
        self.y_preds.append(logits)
        self.iid.append(iid)
        self.qid.append(qid)

    def compute(self):
        score = self.score / self.total
        return score

    def compute_f1(self):
        if len(self.y_trues.shape) == 1: # Binary
            f1 = f1_score(np.array(self.y_trues.cpu()), np.array(self.y_preds.cpu()) >= 0.5)
            return f1, 0.0
        else:
            micro_f1 = f1_score(np.array(self.y_trues.cpu()), np.array(self.y_preds.cpu()) >= 0.5, average='micro')
            macro_f1 = f1_score(np.array(self.y_trues.cpu()), np.array(self.y_preds.cpu()) >= 0.5, average='weighted')
            return micro_f1, macro_f1

    def get_best_score(self):
        self.sync()
        score = self.score / self.total
        if score > self.best_score:
            self.best_score = score
            self.best_micro_f1_score, self.best_macro_f1_score = self.compute_f1()
            for _cat in self.categories:
                self.best_score_per_category[_cat] = getattr(self, f"{_cat}_score") / getattr(self, f"{_cat}_total") if getattr(self, f"{_cat}_total") != 0 else 0
        self.unsync()
        return self.best_score


class VQARADScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.add_state("close_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("close_total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.add_state("open_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("open_total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.best_score = 0
        self.best_close_score = 0
        self.best_open_score = 0

    def update(self, logits, target, types=None):
        logits, target = (
            logits.detach().float().to(self.score.device),
            target.detach().float().to(self.score.device),
        )
        logits = torch.max(logits, 1)[1]
        one_hots = torch.zeros(*target.size()).to(target)
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        scores = one_hots * target

        close_scores = scores[types == 0]
        open_scores = scores[types == 1]

        self.close_score += close_scores.sum()
        self.close_total += len(close_scores)
        self.open_score += open_scores.sum()
        self.open_total += len(open_scores)

        self.score += scores.sum()
        self.total += len(scores)

    def compute(self):
        score = self.score / self.total
        return score

    def get_best_score(self):
        self.sync()
        score = self.score / self.total
        if score > self.best_score:
            self.best_score = score
            self.best_close_score = self.close_score / self.close_total if self.close_total != 0 else 0
            self.best_open_score = self.open_score / self.open_total if self.open_total != 0 else 0
        self.unsync()
        return self.best_score

    def get_best_close_score(self):
        return self.best_close_score

    def get_best_open_score(self):
        return self.best_open_score


class ROCScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("y_trues", default=[], dist_reduce_fx="cat")
        self.add_state("y_scores", default=[], dist_reduce_fx="cat")
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="mean")

    def update(self, logits, target):
        logits, target = (
            logits.detach().float(),
            target.detach().float(),
        )

        y_true = target
        y_score = 1 / (1 + torch.exp(-logits))
        self.y_trues.append(y_true)
        self.y_scores.append(y_score)

    def compute(self):
        try:
            score = sklm.roc_auc_score(np.concatenate([y_true.cpu().numpy() for y_true in self.y_trues], axis=0),
                                       np.concatenate([y_score.cpu().numpy() for y_score in self.y_scores], axis=0))
            self.score = torch.tensor(score).to(self.score)
        except ValueError:
            self.score = torch.tensor(0).to(self.score)
        return self.score


class F1Score(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("y_trues", default=[], dist_reduce_fx="cat")
        self.add_state("y_preds", default=[], dist_reduce_fx="cat")
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="mean")

    def update(self, logits, target):
        logits, target = (
            logits.detach().float(),
            target.detach().float(),
        )

        y_true = target
        y_score = 1 / (1 + torch.exp(-logits)) > 0.5
        self.y_trues.append(y_true)
        self.y_preds.append(y_score)

    def compute(self):
        try:
            score = sklm.f1_score(np.concatenate([y_true.cpu().numpy() for y_true in self.y_trues], axis=0),
                                  np.concatenate([y_pred.cpu().numpy() for y_pred in self.y_preds], axis=0))
            self.score = torch.tensor(score).to(self.score)
        except ValueError:
            self.score = torch.tensor(0).to(self.score)
        return self.score


class PairwiseMetrics(nn.Module):
    def __init__(self, pl_module, phase):
        super().__init__()
        data_dir = pl_module.hparams.config["data_root"]
        assert len(pl_module.hparams.config["datasets"]) == 1
        
        dataset_name = pl_module.hparams.config["datasets"][0]
        suffix = pl_module.hparams.config["exp_name"].split("task_finetune_vqa_mmehr_")[-1]

        # hard coding
        data_root = "../../../dataset/" 
        ans2label = json.load(open(os.path.join(data_root, "ans2idx.json"), "rb"))
        self.true_idx = ans2label['yes']
        assert suffix == "ub"

        self.test_df = pd.read_csv(f"{data_dir}{dataset_name}_{phase}_{suffix}.csv", low_memory=False)
        self.obj_att_pairs = sorted(set(self.test_df.groupby(["object", "attribute"]).image_id.count().index.unique()))
        self.result_df = {}

    def forward(self, y_true, y_pred, logging_path):
        y_true = torch.cat(y_true)[:, self.true_idx].cpu()
        y_pred = torch.cat(y_pred)[:, self.true_idx].cpu()

        for (obj, att) in tqdm(self.obj_att_pairs):
            _key = f"{obj}_{att}"
            self.result_df[_key] = {}

            _y_true = y_true[
                (self.test_df.object == obj) & (self.test_df.attribute == att)
                ]
            _y_pred = y_pred[
                (self.test_df.object == obj) & (self.test_df.attribute == att)
                ]
            
            if len(np.unique(_y_true)) == 1:
                self.result_df[_key]["acc"] = -1
                self.result_df[_key]["f1"] = -1
                self.result_df[_key]["auroc"] = -1
            else:
                acc = accuracy_score(np.array(_y_true), np.array(_y_pred)>=0.5)
                f1 = f1_score(np.array(_y_true), np.array(_y_pred)>=0.5)
                auroc = roc_auc_score(_y_true, _y_pred)
                self.result_df[_key]["acc"] = acc
                self.result_df[_key]["f1"] = f1
                self.result_df[_key]["auroc"] = auroc

            self.result_df[_key]["support"] = len(_y_true)
            
        self.result_df = pd.DataFrame(self.result_df).T
        pd.set_option('display.max_rows', None)
        print(self.result_df)
        print(self.result_df.shape)

        with open(logging_path, 'a') as f:
            import datetime
            datetime_string = datetime.datetime.now().strftime("%d-%m-%y %H:%M:%S")
            f.write("-" * 100 + "\n")
            f.write('%s : logging start' % (datetime_string) + "\n")
            f.write(f"{str(self.result_df)}" + "\n")
            f.write(f"ACC (pairwise acc)    : {self.result_df[self.result_df.acc != -1].acc.mean() * 100}")
            f.write(f"AUROC (pairwise auroc): {self.result_df[self.result_df.auroc != -1].auroc.mean()}")
            f.write(f"F1 (pairwise f1)      : {self.result_df[self.result_df.f1 != -1].f1.mean()}")
        
        print(f"ACC (pairwise acc): {self.result_df[self.result_df.acc != -1].acc.mean() * 100}")
        print(f"AUROC (pairwise auroc): {self.result_df[self.result_df.auroc != -1].auroc.mean()}")
        print(f"F1 (pairwise f1): {self.result_df[self.result_df.f1 != -1].f1.mean()}")