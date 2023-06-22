from collections import defaultdict

from .base_datamodule import BaseDataModule
from ..datasets import VQAMMEHRDataset


class VQAMMEHRataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return VQAMMEHRDataset

    @property
    def dataset_name(self):
        return "vqa_mmehr"
    
    def set_train_dataset(self):
        if not self.config["test_only"]:
            self.train_dataset = self.dataset_cls(
                self.data_dir,
                self.train_transform_keys,
                split="train",
                image_size=self.image_size,
                max_text_len=self.max_text_len,
                draw_false_image=self.draw_false_image,
                draw_false_text=self.draw_false_text,
                image_only=self.image_only,
                label_column_name=self.label_column_name,
                exp_name=self.config["exp_name"],
            )
        else:
            self.train_dataset = self.dataset_cls(
                self.data_dir,
                self.val_transform_keys,
                split="test",
                image_size=self.image_size,
                max_text_len=self.max_text_len,
                draw_false_image=self.draw_false_image,
                draw_false_text=self.draw_false_text,
                image_only=self.image_only,
                label_column_name=self.label_column_name,
                exp_name=self.config["exp_name"],
            )

    def set_val_dataset(self):
        self.val_dataset = self.dataset_cls(
            self.data_dir,
            self.val_transform_keys,
            split="val",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            image_only=self.image_only,
            label_column_name=self.label_column_name,
            exp_name=self.config["exp_name"],
        )

        if hasattr(self, "dataset_cls_no_false"):
            self.val_dataset_no_false = self.dataset_cls_no_false(
                self.data_dir,
                self.val_transform_keys,
                split="val",
                image_size=self.image_size,
                max_text_len=self.max_text_len,
                draw_false_image=0,
                draw_false_text=0,
                image_only=self.image_only,
                label_column_name=self.label_column_name,
            )

    def set_test_dataset(self):
        self.test_dataset = self.dataset_cls(
            self.data_dir,
            self.val_transform_keys,
            split="test",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            image_only=self.image_only,
            label_column_name=self.label_column_name,
            exp_name=self.config["exp_name"],
        )

    def setup(self, stage):
        super().setup(stage)

        train_answers = self.train_dataset.table["answers"].to_pandas().tolist()
        val_answers = self.val_dataset.table["answers"].to_pandas().tolist()
        train_labels = self.train_dataset.table["answer_labels"].to_pandas().tolist()
        val_labels = self.val_dataset.table["answer_labels"].to_pandas().tolist()
        
        all_answers = [c for c in train_answers + val_answers if c is not None]
        all_labels = [c for c in train_labels + val_labels if c is not None]

        # NOTE: Sicne len(all_answers) != len(all_labels), we do not use below codes.

        # all_answers = [l for lll in all_answers for ll in lll for l in ll]
        # all_labels = [l for lll in all_labels for ll in lll for l in ll]
        
        # self.answer2id = {k: v for k, v in zip(all_answers, all_labels)}
        # self.num_class = len(all_labels[0])

        # self.id2answer = defaultdict(lambda: "unknown")
        # for k, v in self.answer2id.items():
        #     self.id2answer[v] = k
