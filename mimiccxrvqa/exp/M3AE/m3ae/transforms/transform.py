from PIL import Image
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop, Grayscale

from .randaug import RandAugment
from .utils import (
    inception_normalize,
    imagenet_normalize,
)


def imagenet_transform(size=800):
    return transforms.Compose(
        [
            Resize(size, interpolation=Image.BICUBIC),
            CenterCrop(size),
            transforms.ToTensor(),
            imagenet_normalize,
        ]
    )


def imagenet_transform_randaug(size=800):
    trs = transforms.Compose(
        [
            Resize(size, interpolation=Image.BICUBIC),
            CenterCrop(size),
            transforms.ToTensor(),
            imagenet_normalize,
        ]
    )
    trs.transforms.insert(0, RandAugment(2, 9))
    return trs


def vit_transform(size=800):
    return transforms.Compose(
        [
            Resize(size, interpolation=Image.BICUBIC),
            CenterCrop(size),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )


def vit_transform_randaug(size=800):
    trs = transforms.Compose(
        [
            Resize(size, interpolation=Image.BICUBIC),
            CenterCrop(size),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )
    trs.transforms.insert(0, RandAugment(2, 9))
    return trs


def mmehr_transform(size):
    return transforms.Compose([
        # Grayscale(num_output_channels=3),
        Resize([size, size]),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def clip_transform(size):
    return Compose([
        Resize(size, interpolation=Image.BICUBIC),
        CenterCrop(size),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def clip_transform_resizedcrop(size):
    return Compose([
        RandomResizedCrop(size, scale=(0.9, 1.0), interpolation=Image.BICUBIC),
        CenterCrop(size),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def clip_transform_randaug(size):
    trs = Compose([
        Resize(size, interpolation=Image.BICUBIC),
        CenterCrop(size),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    trs.transforms.insert(0, lambda image: image.convert('RGBA'))
    trs.transforms.insert(0, RandAugment(2, 9))
    trs.transforms.insert(0, lambda image: image.convert('RGB'))
    return trs
