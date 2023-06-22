from .transform import (
    clip_transform,
    clip_transform_randaug,
    clip_transform_resizedcrop,
    mmehr_transform
)

_transforms = {
    "clip": clip_transform,
    "clip_randaug": clip_transform_randaug,
    "clip_resizedcrop": clip_transform_resizedcrop,
    "mmehr": mmehr_transform
}


def keys_to_transforms(keys: list, size=224):
    return [_transforms[key](size=size) for key in keys]