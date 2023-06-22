from .base_dataset import BaseDataset


class MIMICCXRDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["mimic_cxr_train"]
        elif split == "val":
            names = ["mimic_cxr_val"]
        elif split == "test":
            names = ["mimic_cxr_test"]
        else:
            raise ValueError

        super().__init__(*args, **kwargs, names=names, text_column_name="caption")

    def __getitem__(self, index):
        return self.get_suite(index)
