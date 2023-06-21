import os
import json
import torch
import random
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
from torch.utils.data import Dataset
from torchvision.transforms.functional import resized_crop


class EHRMultiheadallClassification(Dataset):
    def __init__(self, args, phase, data_aug=None, debug=False):
        super(EHRMultiheadallClassification, self).__init__()
        assert phase in ["train", "valid", "test"]
        assert args.img_size == 224
        self.phase = phase
        self.data_aug = data_aug

        self.seed = args.seed
        self.img_size = args.img_size
        self.patch_size = args.patch_size
        self.cropping_type = args.cropping_type

        self.imgroot = args.imgroot
        self.dataroot = args.dataroot

        # if True, only use cropped image. else, use full image & cropped feature map
        self.use_only_bbox = True if self.cropping_type == 'img_crop' else False 

        #### Need to change ####
        self.transform = torch.nn.Sequential(
            transforms.Resize([self.img_size, self.img_size]),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        )

        data_path = os.path.join(self.dataroot, f"{phase}_ref.json")
        print(f"Load dataset {data_path}")
        data_df = pd.DataFrame(json.load(open(os.path.join(self.dataroot, f"{phase}_ref.json"))))         
        print(f"Build upperbound dataset {len(data_df)} samples ({data_df.image_id.nunique()} images)")
        print(f"# of pairs : {data_df.comb.nunique()} / # of objects : {data_df.object.nunique()} /  # of attributes: {data_df.attribute.nunique()}")

        self.data_df = data_df
        self.attr_pool = data_df.attribute.unique()

        print(f"Load {len(data_df)} samples for {phase} set")
        print(data_df.relation.value_counts())

    def __getitem__(self, index):
        sample = {}
        sample["index"] = index
        sample["label"] = self.data_df["relation"].iloc[index]
        sample["object"] = self.data_df["object"].iloc[index]
        sample["attribute"] = self.data_df["attribute"].iloc[index]

        img_path = self.data_df["image_path"].iloc[index]
        img_path = os.path.join(self.imgroot, img_path)
        img = read_image(img_path, mode=ImageReadMode.RGB)
        sample["img"] = self.transform(img)

        sample["coord224"] = torch.tensor(self.data_df["coord224"].iloc[index], dtype=torch.int64)

        _H = self.data_df["height"].iloc[index]
        _W = self.data_df["width"].iloc[index]
        if self.use_only_bbox:
            try:
                # img: Tensor, top: int, left: int, height: int, width: int, size: List[int]
                sample["img"] = resized_crop(sample["img"], min(sample["coord224"][1], sample["coord224"][3]), min(sample["coord224"][0], sample["coord224"][2]), _H, _W, sample["img"].shape[1:])
            except:
                raise NotImplementedError

        if self.data_aug is not None:
            assert self.cropping_type != 'feat_crop'
            transform_seed = np.random.randint(2147483647)
            random.seed(transform_seed)
            sample['img'] = self.data_aug(sample["img"])
        
        if not isinstance(sample, dict):
            raise NotImplementedError
        return sample

    def __len__(self):
        return len(self.data_df)

