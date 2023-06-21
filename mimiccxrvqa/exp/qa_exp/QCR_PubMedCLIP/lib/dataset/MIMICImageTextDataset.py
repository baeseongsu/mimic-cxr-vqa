import json
import unicodedata

from typing import Callable, Optional
from torchvision.io import read_image
from torchvision.transforms import transforms
from torchvision.datasets import VisionDataset
from lib.utils.transform_wrapper import TRANSFORMS


class ImageTextDataset(VisionDataset):
    """
    Dtaset for loading image-text data for tasks like CLIP training, Image Captioning.

    Args:
        root: (string): The root path where the dataset is stored
        file_path: (string): Path to the file containing the image_paths and associated captions.
            The expected format is jsonlines where each line is a json object containing to keys.
            `image_path`: The path to the image.
            `captions`: An `array` of captions.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        name,
        cfg,
        root="",
        max_seq_length=70,
        transform: Optional[Callable] = None
    ):
        super().__init__(root, transform)
        self.input_size = cfg.INPUT_SIZE
        self.cfg = cfg
        self.mode = name
        self.transform = transform
        self.update_transform()


        if self.mode == "train":
            file_path = cfg.DATASET.TRAIN_JSON
        elif self.mode == "eval" or name == "val":
            file_path = cfg.DATASET.VALID_JSON
        elif self.mode == "test":
            file_path = cfg.DATASET.TEST_JSON
        else:
            raise ValueError(f"{name} dataset is not supported!")

        with open(file_path, "r") as f:
            examples = [json.loads(line) for line in f.readlines()]

        examples = [{'image_path': example['img'], 'caption': example['text']} for example in examples]
        
        self.captions = []
        self.image_paths = []
        self.max_seq_length = cfg.TRAIN.MAX_SEQ_LENGTH

        for example in examples:
            self.captions.append(example["caption"])
            self.image_paths.append(example["image_path"])

        self.captions = [unicodedata.normalize("NFKD", c) for c in self.captions]

    def _load_image(self, idx: int):
        path = self.image_paths[idx]
        image = read_image(path)
        return image

    def _load_target(self, idx):
        return self.captions[idx]

    def __getitem__(self, index: int):
        image = self._load_image(index)
        target = self._load_target(index)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        image = self.transform(image)

        return image, target

    def __len__(self) -> int:
        return len(self.captions)

    def update_transform(self, input_size=None):
        normalize = TRANSFORMS["normalize"](cfg=self.cfg, input_size=input_size)
        transform_list = [transforms.ToPILImage()]
        transform_ops = (
                self.cfg.TRANSFORMS.TRAIN_TRANSFORMS
                if self.mode == "train"
                else self.cfg.TRANSFORMS.TEST_TRANSFORMS
                )
        for tran in transform_ops:
            transform_list.append(TRANSFORMS[tran](cfg=self.cfg, input_size=input_size))
        transform_list.extend([transforms.ToTensor(), normalize])
        self.transform = transforms.Compose(transform_list)
