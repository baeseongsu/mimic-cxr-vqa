from .pretraining_medicat_datamodule import MedicatDataModule
from .pretraining_roco_datamodule import ROCODataModule
from .vqa_mmehr_datamodule import VQAMMEHRataModule
from .pretraining_mimic_cxr_datamodule import MIMICCXRDataModule

_datamodules = {
    "medicat": MedicatDataModule,
    "roco": ROCODataModule,
    'vqa_mmehr':VQAMMEHRataModule,
    'mimic_cxr':MIMICCXRDataModule,
}
