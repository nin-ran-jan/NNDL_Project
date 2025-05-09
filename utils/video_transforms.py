from torchvision.transforms import Compose, Lambda
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    ShortSideScale,
    UniformTemporalSubsample,
    Permute, # To ensure C, T, H, W
)
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    RandomHorizontalFlipVideo # If applicable to your data
)

# Standard ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_timesformer_video_transform(
    is_train: bool,
    num_frames_to_sample: int = 8, # Number of frames TimeSformer expects
    target_spatial_size: tuple = (224, 224), # e.g., (224,224) for timesformer_base_patch16_224
    min_short_side_scale: int = 256, # For RandomShortSideScale or ShortSideScale
    max_short_side_scale: int = 320, # For RandomShortSideScale
    horizontal_flip_prob: float = 0.5,
):
    """
    Creates a video transformation pipeline for TimeSformer.
    Input to this transform is expected to be a dictionary {'video': video_tensor_TCHW_uint8}.
    Output will be {'video': video_tensor_CTHW_float_normalized}.
    """
    transform_list = [
        UniformTemporalSubsample(num_frames_to_sample),
        Lambda(lambda x: x / 255.0),  # To float and scale to [0, 1]
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), # Normalization
        Lambda(lambda x: x.permute(1, 0, 2, 3)), # TCHW -> CTHW (PyTorchVideo expects CTHW by default for many models)
    ]

    if is_train:
        transform_list.extend([
            RandomShortSideScale(min_size=min_short_side_scale, max_size=max_short_side_scale),
            CenterCropVideo(crop_size=target_spatial_size), # Or RandomCropVideo
            # RandomHorizontalFlipVideo(p=horizontal_flip_prob), # Uncomment if applicable
        ])
    else: # Validation or Test
        transform_list.extend([
            ShortSideScale(size=min_short_side_scale), # Use min_short_side_scale for consistent val/test sizing
            CenterCropVideo(crop_size=target_spatial_size),
        ])

    return ApplyTransformToKey(key="video", transform=Compose(transform_list))