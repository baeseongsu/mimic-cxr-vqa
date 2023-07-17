"""BERT for report generation finetuning."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import datetime

now = datetime.datetime.now()
now = now.strftime("%Y%m%d_%H%M%S")
print("START", now)
import os
import copy
import glob
import json
import wandb
import pickle
import logging
import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm, trange
from torch.utils.data.distributed import DistributedSampler

import torch
from torch.utils.data import DataLoader, RandomSampler
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score

# custom pkgs
from loader_utils import batch_list_to_batch_tensors
from pytorch_pretrained_bert.model import MedViLLForVQA
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from data_loader import PipelineForVQAMIMIC, VQAMIMICDataset

os.environ["NO_PROXY"] = "huggingface.co"


logger = logging.getLogger(__name__)


def _get_max_epoch_model(output_dir):
    fn_model_list = glob.glob(os.path.join(output_dir, "model.*.bin"))
    fn_optim_list = glob.glob(os.path.join(output_dir, "optim.*.bin"))
    if (not fn_model_list) or (not fn_optim_list):
        return None
    both_set = set([int(Path(fn).stem.split(".")[-1]) for fn in fn_model_list]) & set([int(Path(fn).stem.split(".")[-1]) for fn in fn_optim_list])
    if both_set:
        return max(both_set)
    else:
        return None


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def set_seed(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_rank():
    import torch.distributed as dist

    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def main(args):

    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.global_rank = int(os.environ["RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    print("global_rank: {}, local rank: {}".format(args.global_rank, args.local_rank))

    # Load pre-trained model (from origianl github)
    if args.model_recover_path == "bert-base-uncased":
        args.config_path = "bert-base-uncased"
    elif args.model_recover_path != None:
        args.config_path = os.path.join(os.path.dirname(args.model_recover_path), "config.json")
    else:
        args.config_path = None

    # set experiment name
    if args.model_recover_path == "bert-base-uncased":
        args.exp_name = "bert-base-uncased"
    elif args.model_recover_path != None:
        if args.exp_name == "":
            args.exp_name = args.model_recover_path.split("/")[-2]
    else:
        args.exp_name == "scratch"
        # raise ValueError()

    # set output directory
    if args.model_recover_path == "bert-base-uncased":
        args.output_dir = os.path.join(args.output_dir, f"""{args.model_recover_path}_{args.exp_name}""")
    elif args.model_recover_path != None:
        args.output_dir = os.path.join(args.output_dir, f"""{args.model_recover_path.split("/")[-3]}_{args.exp_name}""")
    else:
        args.output_dir = os.path.join(args.output_dir, f"""{args.model_recover_path}_{args.exp_name}""")
    print("args.output_dir", args.output_dir)

    # set max sequence length
    args.max_seq_length = args.max_len_b + args.len_vis_input + 3  # +3 for 2x[SEP] and [CLS]

    print(" # PID :", os.getpid())
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(args.output_dir, "opt.json"), "w"), sort_keys=True, indent=2)

    logging.basicConfig(
        filename=os.path.join(args.output_dir, args.log_file), filemode="w", format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )

    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    args.device = torch.device("cuda", args.local_rank)

    torch.distributed.init_process_group(backend="nccl", init_method=args.dist_url, world_size=args.world_size, rank=args.global_rank)

    logger.info("device: {} distributed training: {}, 16-bits training: {}".format(device, bool(args.local_rank != -1), args.fp16))
    torch.distributed.barrier()
    setup_for_distributed(args.local_rank == 0)
    torch.backends.cudnn.benchmark

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    # fix random seed
    set_seed(seed=args.seed)

    if args.wandb and is_main_process():
        wandb.init(
            config=args,
            entity=args.wandb_entity_name,
            project=args.wandb_project_name,
            name=args.exp_name,
            reinit=True,
        )

    # load tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)


    PIPELINE_CLASS_MAPPING = {
        "vqa-mimic": PipelineForVQAMIMIC,
    }

    DATASET_CLASS_MAPPING = {
        "vqa-mimic": VQAMIMICDataset,
    }

    preproc_pipeline = PIPELINE_CLASS_MAPPING[args.vqa_dataset](
        args=args,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_length,
        len_vis_input=args.len_vis_input,
    )

    if not args.vqa_eval:
        train_dataset = DATASET_CLASS_MAPPING[args.vqa_dataset](
            args=args,
            split="train",
            file_src=args.src_file,  
            img_root=args.img_path,
            batch_size=args.train_batch_size,
            tokenizer=tokenizer,
            preproc_pipeline=preproc_pipeline,
        )

        # get train sampler
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)

        # get train dataloader
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            sampler=train_sampler if not args.vqa_eval else None,
            collate_fn=batch_list_to_batch_tensors,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        t_total = int(len(train_dataloader) * args.num_train_epochs * 1.0 / args.gradient_accumulation_steps)

    """ eval dataset """
    if args.vqa_dataset == "vqa-mimic":
        _split_for_valid = "valid"

    eval_dataset = DATASET_CLASS_MAPPING[args.vqa_dataset](
        args=args,
        split=_split_for_valid,
        file_src=args.src_file, 
        img_root=args.img_path,
        batch_size=args.train_batch_size,
        tokenizer=tokenizer,
        preproc_pipeline=preproc_pipeline,
    )

    # define dataloader
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=batch_list_to_batch_tensors,
        drop_last=False,  # True
    )

    """ test dataset """
    # get test dataset
    test_dataset = DATASET_CLASS_MAPPING[args.vqa_dataset](
        args=args,
        split="test",
        file_src=args.src_file,  
        img_root=args.img_path,
        batch_size=args.train_batch_size,
        tokenizer=tokenizer,
        preproc_pipeline=preproc_pipeline,
    )

    # define dataloader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=batch_list_to_batch_tensors,
        drop_last=False,  # True
    )

    # prepare model
    recover_step = _get_max_epoch_model(args.output_dir)
    if args.vqa_dataset == "vqa-mimic":
        cls_num_labels = 110
    else:
        raise ValueError()

    type_vocab_size = 2
    relax_projection = 4 if args.relax_projection else 0
    task_idx_proj = 0

    # BERT model will be loaded! from scratch
    if args.model_recover_path is None:
        MODEL_CLASS = MedViLLForVQA
        from pytorch_pretrained_bert.model import BertConfig

        config = BertConfig(vocab_size_or_config_json_file=30522)
        model = MODEL_CLASS(config=config, args=args, num_labels=cls_num_labels)
        print("scratch training")
        torch.cuda.empty_cache()

    elif args.model_recover_path == "bert-base-uncased":
        MODEL_CLASS = MedViLLForVQA
        model = MODEL_CLASS.from_pretrained(
            pretrained_model_name="bert-base-uncased",
            args=args,
            num_labels=cls_num_labels,  # cls_num_labels
            type_vocab_size=type_vocab_size,
            relax_projection=relax_projection,
            task_idx=task_idx_proj,
            max_position_embeddings=args.max_position_embeddings,
            label_smoothing=args.label_smoothing,
            fp32_embedding=args.fp32_embedding,
            drop_prob=args.drop_prob
        )
        print("scratch training")
        torch.cuda.empty_cache()

    else:
        # print("Task :", args.tasks, args.s2s_prob)
        model_recover = None
        for model_recover_path in glob.glob(args.model_recover_path.strip()):
            print("Recover path: ", model_recover_path)
            logger.info("***** Recover model: %s *****", args.model_recover_path)
            model_recover = torch.load(model_recover_path)

            for key in list(model_recover.keys()):
                model_recover[key.replace("enc.", "").replace("mlm.", "cls.")] = model_recover.pop(key)

        MODEL_CLASS = MedViLLForVQA

        model = MODEL_CLASS.from_pretrained(
            args.bert_model,
            state_dict=model_recover,
            args=args,
            num_labels=cls_num_labels,  # cls_num_labels
            type_vocab_size=type_vocab_size,
            relax_projection=relax_projection,
            config_path=args.config_path,
            task_idx=task_idx_proj,
            max_position_embeddings=args.max_position_embeddings,
            label_smoothing=args.label_smoothing,
            fp32_embedding=args.fp32_embedding,
            # cache_dir=args.output_dir+'/.pretrained_model_{}'.format(args.global_rank),
            drop_prob=args.drop_prob,
            # len_vis_input=args.len_vis_input,
            # tasks=args.tasks,
        )

        model.load_state_dict(model_recover, strict=False)
        print("The pretrained model loaded and fine-tuning.")
        del model_recover
        torch.cuda.empty_cache()

    model.to(device)
    if args.wandb and is_main_process():
        wandb.watch(model)

    try:
        from torch.nn.parallel import DistributedDataParallel as DDP
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    if not args.vqa_eval:
        # get optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup_proportion, schedule=args.sche_mode, t_total=t_total)
        if recover_step:
            logger.info("***** Recover optimizer: %d *****", recover_step)
            optim_recover = torch.load(os.path.join(args.output_dir, "optim.{0}.bin".format(recover_step)))
            if hasattr(optim_recover, "state_dict"):
                optim_recover = optim_recover.state_dict()
            optimizer.load_state_dict(optim_recover)
            if args.loss_scale == 0:
                logger.info("***** Recover optimizer: dynamic_loss_scale *****")
                optimizer.dynamic_loss_scale = True

    logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()
    args.eval_output_dir = os.path.join(os.path.dirname(args.output_dir), "eval_outputs")
    os.makedirs(args.eval_output_dir, exist_ok=True)
    if args.vqa_eval:
        print("Start evaluation")
        args.eval_output_dir = os.path.join(os.path.dirname(args.output_dir), "eval_outputs")
        os.makedirs(args.eval_output_dir, exist_ok=True)
        print(f"save results on {args.eval_output_dir}") 

        eval_vqa_acc, eval_closed_acc, eval_open_acc, eval_vqa_labels, eval_vqa_logits = evaluate_vqa_model(
            args=args,
            model=model,
            eval_dataloader=eval_dataloader,
            return_preds=True,
        )
        eval_metrics = {
            "eval/acc (total)": eval_vqa_acc,
            "eval/acc (closed)": eval_closed_acc,
            "eval/acc (open)": eval_open_acc,
        }
        print(eval_metrics)

        test_vqa_acc, test_closed_acc, test_open_acc, test_vqa_labels, test_vqa_logits = evaluate_vqa_model(
            args=args,
            model=model,
            eval_dataloader=test_dataloader,
            return_preds=True,
        )
        test_metrics = {
            "test/acc (total)": test_vqa_acc,
            "test/acc (closed)": test_closed_acc,
            "test/acc (open)": test_open_acc,
        }
        print(test_metrics)

        num_epochs_of_recovered_model = os.path.basename(args.model_recover_path).split(".")[1]
        eval_output_fpath = os.path.join(args.eval_output_dir, f"{num_epochs_of_recovered_model}_result.bin")
        eval_output = {
            "eval_vqa_labels": eval_vqa_labels,
            "eval_vqa_logits": eval_vqa_logits,
            "test_vqa_labels": test_vqa_labels,
            "test_vqa_logits": test_vqa_logits,
            **eval_metrics,
            **test_metrics,
        }

        torch.save(eval_output, eval_output_fpath)

    else:
        if args.do_train:

            logger.info("***** Running training *****")
            model.train()
            print("Total Parameters:", sum([p.nelement() for p in model.parameters()]))

            global_step = 0

            if recover_step:
                start_epoch = recover_step + 1
                print("Recoverd epoch", start_epoch)
            else:
                start_epoch = 1

            # training epoch loop
            for i_epoch in trange(start_epoch, args.num_train_epochs + 1, desc="Epoch"):
                model.train()  # NOTE: temporary remedy to handle RNN error message

                if args.local_rank != -1:
                    train_sampler.set_epoch(i_epoch - 1)
                iter_bar = tqdm(train_dataloader, desc="Iter (loss=X.XXX)")

                # init metrics
                train_loss = []
                total_vqa_score, total_closed_score, total_open_score = [], [], []

                for step, batch in enumerate(iter_bar):

                    # prepare inputs
                    batch = [t.to(device) for t in batch]
                    input_ids, segment_ids, attention_mask, img, vis_pe, ans_labels, ans_type, _organ = batch

                    # half precision
                    if args.fp16:
                        img = img.half()
                        vis_pe = vis_pe.half()

                    # run model
                    loss_tuple = model(
                        img,
                        vis_pe,
                        input_ids,
                        segment_ids,
                        attention_mask,
                        ans_labels,
                        ans_type=ans_type,
                    )

                    # _masked_lm_loss, vqa_loss, vqa_acc, closed_acc, open_acc = loss_tuple
                    vqa_loss, vqa_score, vqa_logits, closed_score, open_score = loss_tuple

                    # loss
                    loss = vqa_loss.mean()
                    train_loss.append(loss.item())
                    iter_bar.set_description("Iter (loss=%5.3f)" % (loss.item()))
                    if args.wandb and is_main_process():
                        wandb.log({"train/loss": loss, "global_step": global_step})

                    # accuracy (step)
                    total_vqa_score += vqa_score
                    total_closed_score += closed_score
                    total_open_score += open_score

                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    loss.backward()

                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                        if args.fp16:
                            for param_group in optimizer.param_groups:
                                param_group["lr"] = lr_this_step
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1

                # accuracy (epoch)
                train_epoch_vqa_acc = sum(total_vqa_score) / len(total_vqa_score)
                if len(total_closed_score) > 0:
                    train_epoch_closed_acc = sum(total_closed_score) / len(total_closed_score)
                else:
                    train_epoch_closed_acc = 0
                if len(total_open_score) > 0:
                    train_epoch_open_acc = sum(total_open_score) / len(total_open_score)
                else:
                    train_epoch_open_acc = 0
                train_epoch_metrics = {
                    "train/epoch acc (total)": train_epoch_vqa_acc,
                    "train/epoch acc (closed)": train_epoch_closed_acc,
                    "train/epoch acc (open)": train_epoch_open_acc,
                }
                # print(train_epoch_metrics)
                if args.wandb and is_main_process():
                    wandb.log(train_epoch_metrics, step=global_step)
                    wandb.log({"learning rate": lr_this_step}, step=global_step)

                logger.info("** ** * Saving fine-tuned model and optimizer ** ** * ")
                model_to_save = model.module if hasattr(model, "module") else model  # Only save the model it-self
                output_config_file = os.path.join(args.output_dir, "config.json")

                with open(output_config_file, "w") as f:
                    f.write(model_to_save.config.to_json_string())

                output_model_file = os.path.join(args.output_dir, "model.{0}.bin".format(i_epoch))
                if args.global_rank in (-1, 0):  # save model if the first device or no dist
                    torch.save(copy.deepcopy(model_to_save).cpu().state_dict(), output_model_file)

                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()

                if args.world_size > 1:
                    torch.distributed.barrier()

                # evaulation (epoch)
                eval_vqa_acc, eval_closed_acc, eval_open_acc = evaluate_vqa_model(args=args, model=model, eval_dataloader=eval_dataloader)
                eval_metrics = {
                    "eval/acc (total)": eval_vqa_acc,
                    "eval/acc (closed)": eval_closed_acc,
                    "eval/acc (open)": eval_open_acc,
                }
                print(eval_metrics)

                if args.wandb and is_main_process():
                    wandb.log(eval_metrics, step=global_step)


def evaluate_vqa_model(args, model, eval_dataloader, return_preds=False):
    """
    Modified Version (2022.07.10, Seongsu Bae)
    """

    logger.info("***** Running Evaulation *****")
    model.eval()
    print("Total Parameters:", sum([p.nelement() for p in model.parameters()]))

    # init metrics
    test_loss = []
    total_vqa_score, total_closed_score, total_open_score = [], [], []
    total_vqa_logits, total_vqa_labels = [], []

    with torch.no_grad():
        for step, batch in enumerate(tqdm(eval_dataloader)):
            # prepare inputs
            batch = [t.to(args.device) for t in batch]
            input_ids, segment_ids, attention_mask, img, vis_pe, ans_labels, ans_type, _organ = batch

            # half precision
            if args.fp16:
                img = img.half()
                vis_pe = vis_pe.half()

            # run model
            loss_tuple = model(
                img,
                vis_pe,
                input_ids,
                segment_ids,
                attention_mask,
                ans_labels,
                ans_type=ans_type,
            )
            vqa_loss, vqa_score, vqa_logits, closed_score, open_score = loss_tuple

            loss = vqa_loss.mean()
            test_loss.append(loss.item())

            total_vqa_score += vqa_score
            total_closed_score += closed_score
            total_open_score += open_score

            total_vqa_logits += vqa_logits.cpu()
            total_vqa_labels += ans_labels.cpu()

    total_vqa_labels = torch.stack(total_vqa_labels, axis=0)
    total_vqa_labels = total_vqa_labels.numpy()

    total_vqa_logits = torch.stack(total_vqa_logits, axis=0)
    total_vqa_logits = total_vqa_logits.numpy()

    if args.vqa_dataset == "vqa-mimic":
        if args.exp_type == "all":
            reports = classification_report(total_vqa_labels, total_vqa_logits >= 0.5)
            print(reports)
            acc = accuracy_score(total_vqa_labels, total_vqa_logits >= 0.5)
            micro_f1 = f1_score(total_vqa_labels, total_vqa_logits >= 0.5, average='micro')
            macro_f1 = f1_score(total_vqa_labels, total_vqa_logits >= 0.5, average='weighted')
            
            file_path = os.path.join(eval_dataloader.dataset.file_src, f"{eval_dataloader.dataset.split}.json")
            eval_df = pd.DataFrame(json.load(open(file_path)), dtype=str)
            
            with open(os.path.join(args.eval_output_dir, f"{args.exp_type}_{eval_dataloader.dataset.split}_log.txt"), 'a') as f:
                import datetime
                datetime_string = datetime.datetime.now().strftime("%d-%m-%y %H:%M:%S")
                f.write("-" * 100 + "\n")
                f.write('%s : logging start' % (datetime_string) + "\n")
                f.write(f"classification report\n")
                f.write(f"{reports}\n")
                f.write(f"score_best_epoch (micro)   | " + "{:.4f}".format(acc * 100) + "\n")
                f.write(f"micro_f1_score_best_epoch  | " + "{:.4f}".format(micro_f1) + "\n")
                f.write(f"macro_f1_score_best_epoch  | " + "{:.4f}".format(macro_f1) + "\n\n")

                result_df = {_cat: {} for _cat in eval_df.content_type.unique()}
                for _cat in sorted(eval_df.content_type.unique()):
                    _y_true = total_vqa_labels[(eval_df.content_type == _cat)]
                    _y_pred = total_vqa_logits[(eval_df.content_type == _cat)]
                    try:
                        acc = accuracy_score(_y_true, _y_pred >= 0.5)
                        f1 = f1_score(np.array(_y_true), np.array(_y_pred) >= 0.5, average="micro")
                    except:
                        import pdb; pdb.set_trace()

                    result_df[_cat]["acc"] = acc * 100
                    result_df[_cat]["f1"] = f1
                result_df = pd.DataFrame(result_df).T
                f.write(f"category-wise acc" + "\n")
                f.write(str(result_df) + "\n")

        elif args.exp_type == "ref":
            result_df = {}
            eval_df = json.load(open(os.path.join(eval_dataloader.dataset.file_src, f"{eval_dataloader.dataset.split}_ref.json")))
            eval_df = pd.DataFrame(eval_df)

            obj_att_pairs = sorted(set(eval_df.groupby(["object", "attribute"]).image_id.count().index.unique()))
            closed_ans_idx = pickle.load(open(os.path.join(eval_dataloader.dataset.file_src, "ans2idx.pkl"), "rb"))['yes']
            total_vqa_labels = total_vqa_labels[:, closed_ans_idx]
            total_vqa_logits = total_vqa_logits[:, closed_ans_idx]

            for (obj, att) in tqdm(obj_att_pairs):
                _key = f"{obj}_{att}"
                result_df[_key] = {}
                _y_true = total_vqa_labels[(eval_df.object == obj) & (eval_df.attribute == att)]
                _y_pred = total_vqa_logits[(eval_df.object == obj) & (eval_df.attribute == att)]
                
                if len(np.unique(_y_true)) == 1:
                    result_df[_key]["acc"] = -1
                    result_df[_key]["f1"] = -1
                    result_df[_key]["auroc"] = -1
                else:
                    acc = accuracy_score(np.array(_y_true), np.array(_y_pred)>=0.5)
                    f1 = f1_score(np.array(_y_true), np.array(_y_pred)>=0.5)
                    auroc = roc_auc_score(_y_true, _y_pred)
                    result_df[_key]["acc"] = acc
                    result_df[_key]["f1"] = f1
                    result_df[_key]["auroc"] = auroc
                result_df[_key]["support"] = len(_y_true)
                
            result_df = pd.DataFrame(result_df).T
            pd.set_option('display.max_rows', None)
            with open(os.path.join(args.eval_output_dir, f"{args.exp_type}_{eval_dataloader.dataset.split}_ref_log.txt"), 'a') as f:
                import datetime
                datetime_string = datetime.datetime.now().strftime("%d-%m-%y %H:%M:%S")
                f.write("-" * 100 + "\n")
                f.write('%s : logging start' % (datetime_string) + "\n")
                f.write(f"{str(result_df)}" + "\n")
                f.write(f"ACC (pairwise acc)    : {result_df[result_df.acc != -1].acc.mean() * 100}")
                f.write(f"AUROC (pairwise auroc): {result_df[result_df.auroc != -1].auroc.mean()}")
                f.write(f"F1 (pairwise f1)      : {result_df[result_df.f1 != -1].f1.mean()}")

    eval_vqa_acc = sum(total_vqa_score) / len(total_vqa_score)
    if len(total_closed_score) != 0:
        eval_closed_acc = sum(total_closed_score) / len(total_closed_score)
    else:
        eval_closed_acc = 0
    if len(total_open_score) != 0:
        eval_open_acc = sum(total_open_score) / len(total_open_score)
    else:
        eval_open_acc = 0

    # empty cache
    logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()

    if return_preds:
        eval_output = eval_vqa_acc, eval_closed_acc, eval_open_acc, total_vqa_labels, total_vqa_logits
    else:
        eval_output = eval_vqa_acc, eval_closed_acc, eval_open_acc

    return eval_output


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased")

    # model
    parser.add_argument("--model_recover_path", default=None, type=str, help="The file of fine-tuned pretraining model. ex)'./pretrained_model/pytorch_model.bin'")
    parser.add_argument("--len_vis_input", type=int, default=256, help="The length of visual token input")
    parser.add_argument("--img_encoding", type=str, default="fully_use_cnn", choices=["random_sample", "fully_use_cnn"])
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--img_hidden_sz", type=int, default=2048, help="Whether to use amp for fp16")
    parser.add_argument("--img_postion", default=True, help="It will produce img_position.")
    parser.add_argument("--drop_prob", default=0.1, type=float)

    # truncate_config for input
    parser.add_argument("--max_len_b", type=int, default=253, help="Truncate_config: maximum length of segment B. (Language)")
    parser.add_argument("--trunc_seg", type=str, default="b", help="Truncate_config: first truncate segment A/B (option: a, b).")
    parser.add_argument("--always_truncate_tail", action="store_true", help="Truncate_config: Whether we should always truncate tail.")

    # dataset
    parser.add_argument("--vqa_dataset", default="vqa-mimic", type=str, choices=["vqa-mimic"])
    # dataset (vqa-mimic)
    parser.add_argument("--exp_type", default="all", type=str, choices=["all", "ref"])

    # training
    parser.add_argument("--do_train", action="store_true", default=True, help="Whether to run training. This should ALWAYS be set to True.")
    parser.add_argument("--num_train_epochs", default=50, type=int)
    parser.add_argument("--train_batch_size", default=16, type=int)

    parser.add_argument("--sche_mode", type=str, default="warmup_linear", help="warmup_linear | warmup_constant | warmup_cosine")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")  # 3e-5
    parser.add_argument("--label_smoothing", default=0, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="The weight decay rate for Adam.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Proportion of training to perform linear learning rate warmup for. " "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--fp16", action="store_true", default=False, help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument("--fp32_embedding", action="store_true", default=False, help="Whether to use 32-bit float precision instead of 32-bit for embeddings")
    parser.add_argument("--vqa_eval", action="store_true", help="vqa_eval | True | False")

    parser.add_argument(
        "--loss_scale",
        type=float,
        default=0,
        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n" "0 (default value): dynamic loss scaling.\n" "Positive power of 2: static loss scaling value.\n",
    )
    parser.add_argument("--amp", action="store_true", default=False, help="Whether to use amp for fp16")

    # logging / directory
    parser.add_argument("--wandb", action="store_true", default=False, help="Whether to use wandb logging")
    parser.add_argument("--wandb_entity_name", type=str, default="ehr-vqg")
    parser.add_argument("--wandb_project_name", type=str, default="MIMIV-VQA-MedViLL")
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--src_file", type=str, default="../../dataset")
    parser.add_argument("--img_path", type=str, default="../../../physionet.org/files/mimic-cxr-jpg/2.0.0/re512_3ch_contour_cropped")
    parser.add_argument(
        "--output_dir", default="./saved_results/vqa_finetune", type=str, help="The output directory where the model predictions and checkpoints will be written."
    )
    parser.add_argument("--log_file", default="training.log", type=str, help="The output directory where the log will be written.")

    parser.add_argument("--num_workers", default=8, type=int, help="Number of workers for the data loader.")
    parser.add_argument("--max_position_embeddings", type=int, default=None, help="max position embeddings")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--relax_projection", action="store_true", help="Use different projection layers for tasks.")

    args = parser.parse_args()

    main(args)
