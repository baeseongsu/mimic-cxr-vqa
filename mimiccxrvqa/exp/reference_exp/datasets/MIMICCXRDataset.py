import os
import jsonlines
from PIL import Image
from torch.utils.data import Dataset


class MIMICCXRData(Dataset):
    def __init__(self, dataroot, corpus='train', transform=None):
        super(MIMICCXRData, self).__init__()
        ids = []
        img_paths = []
        labels = []
        if corpus == 'train':
            file_name = os.path.join(dataroot, 'pretrain_train.jsonl')
        elif corpus == 'valid':
            file_name = os.path.join(dataroot, 'pretrain_valid.jsonl')
            
        with jsonlines.open(file_name) as f:
            for line in f:
                if os.path.isfile(line['img']):
                    ids.append(line['id'])
                    img_paths.append(line['img'])

        self.imgs = img_paths
        self.ids = ids
        self.transform = transform

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        img_path = self.imgs[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.imgs)
