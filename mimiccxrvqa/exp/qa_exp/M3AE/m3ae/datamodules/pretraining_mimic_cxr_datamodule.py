from .base_datamodule import BaseDataModule
from ..datasets import MIMICCXRDataset


class MIMICCXRDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return MIMICCXRDataset

    @property
    def dataset_cls_no_false(self):
        return MIMICCXRDataset

    @property
    def dataset_name(self):
        return "mimix_cxr"
