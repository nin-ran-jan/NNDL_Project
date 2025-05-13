from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
)
import torch
from torchvision import transforms
import random

# AI Prompts used
# improve formating of the code
# improve variable names
# improve docstrings

# Standard ImageNet normalization
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
    target_spatial_size: tuple = (224, 224),
):
    """
    Returns a transform pipeline for video data augmentation.
    These transforms are applied *before* the Hugging Face processor if they modify
    the frame content in ways the processor doesn't (e.g., advanced color jitter, frame-level crops).
    The HF processor will handle final resizing and normalization.
    Input: TCHW uint8 tensor. Output: TCHW float tensor (typically).
    """
    if is_train:
        frame_transforms = [
            transforms.ToPILImage(), # The processor expects PIL images
            transforms.RandomResizedCrop(
                size=target_spatial_size, 
                scale=(0.6, 1.0), 
                ratio=(0.75, 1.33)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.ToTensor(), 
        ]
        video_pipeline = transforms.Compose([
            ApplyToFrames(transforms.Compose(frame_transforms)),
        ])
    else: # val/test path
        # processor for hugging face handles this for eval. hence minimal
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