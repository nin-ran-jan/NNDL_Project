from torchvision.transforms import Compose, Lambda
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
)
import torch
from torchvision import transforms
import random

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

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class ApplyToFrames:
    """
    Apply a torchvision image transform to each frame in a video tensor.
    Assumes video tensor is TCHW or T,H,W,C. Outputs TCHW.
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, video_tensor_tchw):
        # video_tensor_tchw expected to be (T, C, H, W)
        # Permute to (T, H, W, C) for some transforms if needed, or apply directly if transform handles TCHW slices
        # Most torchvision transforms expect (C, H, W)
        
        transformed_frames = []
        for i in range(video_tensor_tchw.size(0)):
            frame = video_tensor_tchw[i] # (C, H, W)
            transformed_frame = self.transform(frame)
            transformed_frames.append(transformed_frame)
        return torch.stack(transformed_frames)

class RandomTemporalChunkCrop:
    """
    Randomly select a contiguous chunk of `num_frames_to_sample` from a longer sequence of frames.
    If the input sequence is shorter, it pads or repeats frames.
    """
    def __init__(self, num_frames_to_sample: int, pad_mode: str = 'edge'):
        self.num_frames_to_sample = num_frames_to_sample
        self.pad_mode = pad_mode

    def __call__(self, video_frames_tchw: torch.Tensor):
        num_input_frames = video_frames_tchw.shape[0]

        if num_input_frames == self.num_frames_to_sample:
            return video_frames_tchw
        elif num_input_frames < self.num_frames_to_sample:
            # Pad
            padding_needed = self.num_frames_to_sample - num_input_frames
            if self.pad_mode == 'edge':
                # Repeat last frame
                last_frame = video_frames_tchw[-1:, ...] # Keep dimension
                padding = last_frame.repeat(padding_needed, 1, 1, 1)
            else: # 'zeros'
                padding_shape = (padding_needed,) + video_frames_tchw.shape[1:]
                padding = torch.zeros(padding_shape, dtype=video_frames_tchw.dtype, device=video_frames_tchw.device)
            return torch.cat((video_frames_tchw, padding), dim=0)
        else: # num_input_frames > self.num_frames_to_sample
            start_index = random.randint(0, num_input_frames - self.num_frames_to_sample)
            return video_frames_tchw[start_index : start_index + self.num_frames_to_sample]


def get_video_augmentation_transforms(
    is_train: bool,
    target_spatial_size: tuple = (224, 224), # For RandomResizedCrop or final size
    # num_frames_to_sample: int = 8 # This will be handled by the dataset before processor
):
    """
    Returns a transform pipeline for video data augmentation.
    These transforms are applied *before* the Hugging Face processor if they modify
    the frame content in ways the processor doesn't (e.g., advanced color jitter, frame-level crops).
    The HF processor will handle final resizing and normalization.
    Input: TCHW uint8 tensor. Output: TCHW float tensor (typically).
    """
    if is_train:
        # These transforms are applied to each frame individually
        frame_transforms = [
            transforms.ToPILImage(), # For torchvision transforms that expect PIL
            transforms.RandomResizedCrop(
                size=target_spatial_size, # Crop to this size directly
                scale=(0.6, 1.0), # Crop 60% to 100% of the image
                ratio=(0.75, 1.33)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.ToTensor(), # Converts PIL to FloatTensor in [0,1] and C,H,W
            # Normalization will be handled by the Hugging Face processor later
        ]
        # Video-level transforms (applied to TCHW tensor)
        video_pipeline = transforms.Compose([
            # Example: RandomTemporalChunkCrop(num_frames_to_sample), # If you want to sample chunks before HF processor
            ApplyToFrames(transforms.Compose(frame_transforms)),
            # Note: HF processor will do final normalization and potentially resizing.
            # These augmentations focus on geometric and color variations.
        ])
    else: # Validation / Test
        # Minimal transforms, usually just resize/crop to what model expects.
        # The HF processor handles this for eval. So, minimal pre-processor transforms.
        frame_transforms_eval = [
            transforms.ToPILImage(),
            transforms.Resize(target_spatial_size[0]), # Resize shorter edge
            transforms.CenterCrop(target_spatial_size),
            transforms.ToTensor(),
        ]
        video_pipeline = transforms.Compose([
            ApplyToFrames(transforms.Compose(frame_transforms_eval)),
        ])
    return video_pipeline