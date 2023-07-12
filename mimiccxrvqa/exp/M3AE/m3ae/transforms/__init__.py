from .transform import (
    clip_transform,
    clip_transform_randaug,
    clip_transform_resizedcrop,
    mmehr_transform,
    mmehr_transform_aug
)

_transforms = {
    "clip": clip_transform,
    "clip_randaug": clip_transform_randaug,
    "clip_resizedcrop": clip_transform_resizedcrop,
    "mmehr": mmehr_transform,
    "mmehr_aug": mmehr_transform_aug,
}


def keys_to_transforms(keys: list, size=224):
    return [_transforms[key](size=size) for key in keys]
