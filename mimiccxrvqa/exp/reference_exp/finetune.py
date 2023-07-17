# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import json
import wandb
import random
import argparse
import numpy as np
from pathlib import Path

import torch
import torchvision
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torch import nn
from torchmetrics import F1Score, AUROC, Accuracy
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import utils.utils as utils
from modules.projection_module import DINOMIMICClassification
from datasets.MIMICMultiheadClassificationDataset import EHRMultiheadallClassification


def cal_sample_acc(pred, label):
    return (pred == label).sum() / len(label)


def evaluate(args, model=None):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    assert os.path.isdir(args.output_dir)
    cudnn.benchmark = True
    
    validset = EHRMultiheadallClassification(args=args, phase="valid", data_aug=None, debug=args.debug)
    testset = EHRMultiheadallClassification(args=args, phase="test", data_aug=None, debug=args.debug)
    
    valid_loader = torch.utils.data.DataLoader(
        validset,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    model = DINOMIMICClassification(args, testset.attr_pool)
    model = model.cuda()
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    checkpoint = torch.load(os.path.join(args.output_dir, "best_val_checkpoint.pth.tar"), map_location="cpu")
    best_epoch = checkpoint["epoch"]
    if args.full_finetune:
        model.module.model.load_state_dict(checkpoint["state_dict"]["model"], strict=False)
    model.module.mlps.load_state_dict(checkpoint["state_dict"]["mlps"], strict=False)

    save_log_name = f"valid_log"
    valid_stats = test_network(valid_loader, model, args.n_last_blocks, output_dir=args.output_dir, full_finetune=args.full_finetune, save_log_name=save_log_name)
    with (Path(args.output_dir) / f"{save_log_name}.txt").open("a") as f:
        f.write(f"Epoch {best_epoch}" + "\n")
        f.write(f"Load {len(validset.data_df)} samples for valid set" + "\n")
        f.write(f"Evaluation of the network on the {len(validset)} valid images" + "\n")
        f.write(
            f"sample ACC / micro AUC / macro AUC / micro F1 / macro F1: {valid_stats['micro_acc']*100:.3f}% / {valid_stats['micro_auc']:.3f}% / {valid_stats['macro_auc_objattr']:.3f}%/ {valid_stats['micro_f1']:.3f}% / {valid_stats['macro_f1_objattr']:.3f}"
            + "\n"
        )

    save_log_name = f"test_log" 
    test_stats = test_network(test_loader, model, args.n_last_blocks, output_dir=args.output_dir, full_finetune=args.full_finetune, save_log_name=save_log_name)
    with (Path(args.output_dir) / f"{save_log_name}.txt").open("a") as f:
        f.write(f"Load {len(testset.data_df)} samples for test set" + "\n")
        f.write(f"Evaluation of the network on the {len(testset)} test images" + "\n")
        f.write(
            f"sample ACC / micro AUC / macro AUC / micro F1 / macro F1: {test_stats['micro_acc']*100:.3f}% / {test_stats['micro_auc']:.3f}% / {test_stats['macro_auc_objattr']:.3f}%/ {test_stats['micro_f1']:.3f}% / {test_stats['macro_f1_objattr']:.3f}"
            + "\n"
        )

    print(f"Accuracy of the network on the {len(testset)} test images: {test_stats['micro_acc']*100:.3f}%")


def finetune_linear_projection(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    # fix the seed for reproduciability
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    print("seed: ", args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    ### Dataset loading ###
    if args.data_aug:
        _data_aug_rot = 25
        _data_aug_trans = 0.15
        _data_aug_scale = 0.10
        data_aug = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomAffine(
                    _data_aug_rot,
                    translate=(_data_aug_trans, _data_aug_trans),
                    scale=(1.0 - _data_aug_scale, 1.0 + _data_aug_scale),
                ),
            ]
        )
    else:
        data_aug = None

    ### Set wandb logging 
    if args.wandb and utils.is_main_process():
        if args.wandb_id is None:
            args.wandb_id = wandb.util.generate_id()
            
        wandb.init(
            config=args,
            entity=args.wandb_entity_name,
            project=args.wandb_project_name,
            name=args.output_dir.split("/")[-2],
            id=args.wandb_id,
            resume=args.wandb_resume,
            reinit=not args.wandb_resume,
        )

    trainset = EHRMultiheadallClassification(args=args, phase="train", data_aug=data_aug, debug=args.debug)
    sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    train_loader = torch.utils.data.DataLoader(
        trainset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    validset = EHRMultiheadallClassification(args=args, phase="valid", data_aug=data_aug, debug=args.debug)
    val_loader = torch.utils.data.DataLoader(
        validset,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    with (Path(args.output_dir) / "log.txt").open("a") as f:
        print(f"Data loaded with {len(trainset)} train and {len(validset)} val imgs.")

    ### model setup ###
    model = DINOMIMICClassification(args, validset.attr_pool)
    model = model.cuda()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    if args.wandb and utils.is_main_process():
        wandb.watch(model)

    # set optimizer
    if args.full_finetune:
        params_groups = utils.get_params_groups(model)
    else:
        params_groups = utils.get_params_groups(model.module.mlps)

    optimizer = torch.optim.SGD(
        params_groups,
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.0,  # linear scaling rule
        momentum=0.9,
        weight_decay=0,  # we do not apply weight decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0.0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth.tar"),
        run_variables=to_restore,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    if os.path.isfile(os.path.join(args.output_dir, "checkpoint.pth.tar")):
        checkpoint = torch.load(os.path.join(args.output_dir, "checkpoint.pth.tar"), map_location="cpu")
        if args.full_finetune:
            model.module.model.load_state_dict(checkpoint["state_dict"]["model"])
        model.module.mlps.load_state_dict(checkpoint["state_dict"]["mlps"])

    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]

    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)

        train_stats = train(model, optimizer, train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens, args.full_finetune)
        scheduler.step()

        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch}
        if args.full_finetune:
            save_state_dict = {
                "model": model.module.model.state_dict(),
                "mlps": model.module.mlps.state_dict(),
            }
        else:
            save_state_dict = {
                "mlps": model.module.mlps.state_dict(),
            }

        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats = validate_network(val_loader, model, args.n_last_blocks, args.avgpool_patchtokens, args.full_finetune)
            print(f"Accuracy & f1 score at epoch {epoch} of the network on the {len(validset)} valid images: {test_stats['acc1']:.1f}% / {test_stats['micro_f1']:.2f}")
            best_acc = max(best_acc, test_stats["acc1"])
            print(f"Max accuracy so far: {best_acc:.2f}%")
            log_stats = {**{k: v for k, v in log_stats.items()}, **{f"valid_{k}": v for k, v in test_stats.items()}}
            
            if args.wandb and utils.is_main_process():
                wandb.log(log_stats)

            if best_acc == test_stats["acc1"]:
                save_dict = {
                    "epoch": epoch + 1,
                    "state_dict": save_state_dict,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_acc": best_acc,
                }
                torch.save(save_dict, os.path.join(args.output_dir, "best_val_checkpoint.pth.tar"))

        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": save_state_dict,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth.tar"))
    print("Training of the supervised linear classifier on frozen features completed.\n" "Top-1 valid accuracy: {acc:.1f}".format(acc=best_acc))
    return model


def train(model, optimizer, loader, epoch, n, avgpool, full_finetune=False):
    model.module.mlps.train()
    if full_finetune:
        model.module.model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    for data in metric_logger.log_every(loader, 20, header):
        # move to gpu
        data = {k:v.cuda(non_blocking=True) if k not in ["object", "attribute"] else v for k,v in data.items()}
        data["attribute"] = np.array(data["attribute"])
        data["object"] = np.array(data["object"])

        # forward
        output = model(data, n)
        loss = nn.CrossEntropyLoss()(output, data["label"])

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(val_loader, model, n, avgpool, full_finetune):
    model.module.mlps.eval()
    if full_finetune:
        model.module.model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    task = "binary"
    total_preds = torch.FloatTensor([]).cuda(non_blocking=True)
    total_labels = torch.LongTensor([]).cuda(non_blocking=True)
    header = "Test:"
    for data in metric_logger.log_every(val_loader, 20, header):
        # move to gpu)
        data = {k:v.cuda(non_blocking=True) if k not in ["object", "attribute"] else v for k,v in data.items()}
        data["attribute"] = np.array(data["attribute"])
        data["object"] = np.array(data["object"])

        # forward
        output = model(data, n)
        loss = nn.CrossEntropyLoss()(output, data["label"])

        (acc1,) = utils.accuracy(output, data["label"], topk=(1,))
        if task == "binary":
            output = torch.softmax(output, dim=1)[:, 1]
        else:
            output = F.softmax(output, dim=1)

        total_preds = torch.cat([total_preds, output], dim=0)
        total_labels = torch.cat([total_labels, data["label"]], dim=0)

        batch_size = data["img"].shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)

    total_preds = total_preds.cpu().numpy()
    total_labels = total_labels.cpu().numpy()
    f1 = f1_score(total_labels, total_preds >= 0.5, average='micro')
    metric_logger.meters["micro_f1"].update(f1, n=total_labels.shape[0])
    metric_logger.synchronize_between_processes()
    print("* Acc@1 {top1.global_avg:.3f} | f1 {f1:.3f} | loss {losses.global_avg:.3f}".format(top1=metric_logger.acc1, f1=f1, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test_network(val_loader, model, n, output_dir, full_finetune=False, split="test", save_log_name=None):
    # n = args.n_last_blocks
    model.module.mlps.eval()
    if full_finetune:
        model.module.model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    
    total_preds = torch.FloatTensor([]).cuda(non_blocking=True)
    total_labels = torch.LongTensor([]).cuda(non_blocking=True)
    total_objects = []
    total_attributes = []

    header = "Test:"
    for data in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        data = {k:v.cuda(non_blocking=True) if k not in ["object", "attribute"] else v for k,v in data.items()}
        data["attribute"] = np.array(data["attribute"])
        data["object"] = np.array(data["object"])

        # forward
        preds = model(data, n)
        loss = nn.CrossEntropyLoss()(preds, data["label"])

        total_preds = torch.cat([total_preds, preds], dim=0)
        total_labels = torch.cat([total_labels, data["label"]], dim=0)
        total_objects.extend(data["object"])
        total_attributes.extend(data["attribute"])

    total_preds = total_preds.cpu()
    total_labels = total_labels.cpu()
    total_objects = np.array(total_objects)
    total_attributes = np.array(total_attributes)
 
    output = {}
    task = "binary"
    total_probs = F.softmax(total_preds, dim=1)[:, 1] # probability of positive class (1)

    output = {
        **output,
        **{
            "micro_acc": accuracy_score(total_labels, total_probs >= 0.5),
            "micro_f1": f1_score(total_labels, total_probs >= 0.5, average='micro'),
            "micro_auc": roc_auc_score(total_labels, total_probs),
        }
    }
    
    # output - per attribute - micro
    from collections import Counter
    output_attr = {}
    for attr in val_loader.dataset.data_df["attribute"].unique():
        total_probs_attr = total_probs[total_attributes == attr]
        total_labels_attr = total_labels[total_attributes == attr]
        if len(total_labels_attr.unique()) == 2: # if there are both positive and negative samples
            output_attr[attr] = {
                "acc": accuracy_score(total_labels_attr, total_probs_attr >= 0.5),
                "f1": f1_score(total_labels_attr, total_probs_attr >= 0.5, average='micro'),
                "auc": roc_auc_score(total_labels_attr, total_probs_attr),
            }
            output_attr[attr]["support"] = [v for (k, v) in sorted(dict(Counter(total_labels_attr.numpy())).items(), key=lambda item: item[0])]
            
    # output - per attribute - macro
    output = {
        **output,
        **{
            "macro_acc_attr": np.mean([v["acc"] for v in output_attr.values()]),
            "macro_f1_attr": np.mean([v["f1"] for v in output_attr.values()]),
            "macro_auc_attr": np.mean([v["auc"] for v in output_attr.values()]),
        }
    }
            
    # output - per object_attribute - micro
    output_objattr = {}
    for (obj, attr) in val_loader.dataset.data_df[["object", "attribute"]].drop_duplicates().values:
        total_probs_objattr = total_probs[(total_objects == obj) & (total_attributes == attr)]
        total_labels_objattr = total_labels[(total_objects == obj) & (total_attributes == attr)]
        if len(total_labels_objattr.unique()) == 2: # if there are both positive and negative samples
            objattr = f"{obj}_{attr}"
            output_objattr[objattr] = {
                "acc": accuracy_score(total_labels_objattr, total_probs_objattr >= 0.5),
                "f1": f1_score(total_labels_objattr, total_probs_objattr >= 0.5, average='micro'),
                "auc": roc_auc_score(total_labels_objattr, total_probs_objattr),
            }
            output_objattr[objattr]["support"] = [v for (k, v) in sorted(dict(Counter(total_labels_objattr.numpy())).items(), key=lambda item: item[0])]
    
    # output - per object_attribute - macro        
    output = {
        **output,
        **{
            "macro_acc_objattr": np.mean([v["acc"] for v in output_objattr.values()]),
            "macro_f1_objattr": np.mean([v["f1"] for v in output_objattr.values()]),
            "macro_auc_objattr": np.mean([v["auc"] for v in output_objattr.values()]),
        },
    }
    
    # save output
    save_pt_name = f"{save_log_name}.pt"
    torch.save(
        {
            "labels": total_labels, 
            "logits": total_preds,
            "objects": total_objects,
            "attributes": total_attributes,
        },
        os.path.join(output_dir, save_pt_name)
    )
    
    # save logging
    save_log_name = f"{save_log_name}.txt" 
    with (Path(output_dir) / save_log_name).open("a") as f:
        f.write("-" * 100 + "\n")
        f.write(f"{len(output_attr)} attribute pairs" + "\n")
        for attr in val_loader.dataset.data_df["attribute"].sort_values().unique():
            auc = output_attr[attr]["auc"] if attr in output_attr else (-1 if task == "binary" else [-1])
            f1 = output_attr[attr]["f1"] if attr in output_attr else (-1 if task == "binary" else [-1])
            acc = output_attr[attr]["acc"] if attr in output_attr else (-1 if task == "binary" else [-1])
            support = output_attr[attr]["support"] if attr in output_attr else (-1 if task == "binary" else [-1])
            if task == "binary":
                f.write(f"{attr:80s} | auc & f1 & acc score: {auc:.3f} / {f1:.3f} / {acc:.3f} | supp: {support}" + "\n")
            else:
                f.write(f"{attr:80s} | AUC & f1 & acc score: " + " ".join(f"{x:.3f}" for x in auc) + " / " +  " ".join(f"{x:.3f}" for x in f1) + " / " +  " ".join(f"{x:.3f}" for x in acc) + f" | supp: {support}\n")

        f.write("-" * 100 + "\n")
        f.write(f"{len(output_objattr)} object-attribute pairs" + "\n")
        for (obj, attr) in val_loader.dataset.data_df[["object", "attribute"]].drop_duplicates().sort_values(by=["object", "attribute"]).values:
            objattr = f"{obj}_{attr}"
            auc = output_objattr[objattr]["auc"] if objattr in output_objattr else (-1 if task == "binary" else [-1])
            f1 = output_objattr[objattr]["f1"] if objattr in output_objattr else (-1 if task == "binary" else [-1])
            acc = output_objattr[objattr]["acc"] if objattr in output_objattr else (-1 if task == "binary" else [-1])
            support = output_objattr[objattr]["support"] if objattr in output_objattr else (-1 if task == "binary" else [-1])
            if task == "binary":
                f.write(f"{objattr:80s} | auc & f1 & acc score: {auc:.3f} / {f1:.3f} / {acc:.3f} | supp: {support}" + "\n")
            else:
                f.write(f"{objattr:80s} | AUC & f1 & acc score: " + " ".join(f"{x:.3f}" for x in auc) + " / " +  " ".join(f"{x:.3f}" for x in f1) + " / " +  " ".join(f"{x:.3f}" for x in acc) + f" | supp: {support}\n")

        f.write("-" * 100 + "\n")
        if task == "binary":
            f.write(f"""micro AUC & f1 & acc           : {output["micro_auc"]:.3f} / {output["micro_f1"]:.3f} / {output["micro_acc"] * 100:.3f}""" + "\n")
            f.write(f"""macro AUC & f1 & acc (attr)    : {output["macro_auc_attr"]:.3f} / {output["macro_f1_attr"]:.3f} / {output["macro_acc_attr"] * 100:.3f}""" + "\n")
            f.write(f"""macro AUC & f1 & acc (obj-attr): {output['macro_auc_objattr']:.3f} / {output["macro_f1_objattr"].item():.3f} / {output["macro_acc_objattr"] * 100:.3f}""" + "\n")
        else:
            f.write(f"micro AUC & f1 & acc           :" + " ".join(f"{x:.3f}" for x in output["micro_auc"]) + " / " +  " ".join(f"{x:.3f}" for x in output["micro_f1"]) + " / " +  " ".join(f"{x*100:.3f}" for x in output["macro_acc_attr"]) + "\n")
            f.write(f"micro AUC & f1 & acc (attr)    :" + " ".join(f"{x:.3f}" for x in output["macro_auc_attr"]) + " / " +  " ".join(f"{x:.3f}" for x in output["macro_f1_attr"]) + " / " +  " ".join(f"{x*100:.3f}" for x in output["micro_acc"]) + "\n")
            f.write(f"micro AUC & f1 & acc (obj-attr):" + " ".join(f"{x:.3f}" for x in output["macro_auc_objattr"]) + " / " +  " ".join(f"{x:.3f}" for x in output["macro_f1_objattr"]) + " / " +  " ".join(f"{x*100:.3f}" for x in output["macro_acc_objattr"])+ "\n")
        
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Finetune linear classification on MIMIC-CXR")
    parser.add_argument(
        "--n_last_blocks",
        default=4,
        type=int,
        help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""",
    )
    parser.add_argument(
        "--avgpool_patchtokens",
        default=False,
        type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""",
    )
    parser.add_argument("--arch", default="vit_small", type=str, help="Architecture")
    parser.add_argument("--patch_size", default=16, type=int, help="Patch resolution of the model.")
    parser.add_argument("--pretrained_weights", default="", type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs of training.")
    parser.add_argument(
        "--lr",
        default=0.001,
        type=float,
        help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""",
    )
    parser.add_argument("--batch_size_per_gpu", default=128, type=int, help="Per-GPU batch-size")
    parser.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""",
    )
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--data_path", default="/path/to/imagenet/", type=str)
    parser.add_argument("--num_workers", default=4, type=int, help="Number of data loading workers per GPU.")
    parser.add_argument("--val_freq", default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument("--output_dir", default=".", help="Path to save logs and checkpoints")
    parser.add_argument("--evaluate", dest="evaluate", action="store_true", help="evaluate model on validation set")

    #### For MIMIC ####
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument("--imgroot", default="../../../physionet.org/files/mimic-cxr-jpg/re512_3ch_contour_cropped", type=str)
    parser.add_argument("--dataroot", default="../../dataset", type=str)
    
    parser.add_argument("--cropping_type", dest="cropping_type", type=str, default=None, choices=["img_crop", "feat_crop", "full_img"])
    parser.add_argument("--feature_type", dest="feature_type", type=str, default="only_global", choices=["only_global", "only_local", "both_local_global"])
    parser.add_argument("--from_scratch", dest="from_scratch", action="store_true", help="from the scratch model")
    parser.add_argument("--full_finetune", dest="full_finetune", action="store_true", help="full model finetune or linear probe")
    parser.add_argument("--data_aug", dest="data_aug", action="store_true", help="use data augmentation or not")
    parser.add_argument("--debug", dest="debug", action="store_true", help="debugging mode or not")

    ## wandb logging
    parser.add_argument("--wandb", action="store_true", default=False, help="Whether to use wandb logging")
    parser.add_argument("--wandb_entity_name", type=str, default="ehr-vqg")
    parser.add_argument("--wandb_project_name", type=str, default="mimic-cxr-vqa-reference")
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--wandb_resume", action="store_true", default=False, help="Whether to allow wandb resume")

    args = parser.parse_args()
    if args.evaluate:
        evaluate(args)
    else:
        model = finetune_linear_projection(args)
        evaluate(args, model)
