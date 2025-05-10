# datasets/videomae_dataset.py
import torch
from torch.utils.data import Dataset
from transformers import VideoMAEImageProcessor
import numpy as np

class VideoMAEDataset(Dataset):
    def __init__(self, frames_per_video, labels_per_video, processor: VideoMAEImageProcessor):
        self.frames_per_video = frames_per_video
        self.labels_per_video = labels_per_video
        self.processor = processor

    def __len__(self):
        return len(self.frames_per_video)

    def __getitem__(self, idx):
        frames = self.frames_per_video[idx]
        labels = self.labels_per_video[idx]
        # Convert list of frames to numpy array with shape (num_frames, height, width, channels)
        video = np.stack(frames)
        # Preprocess the video frames
        processed = self.processor(list(video), return_tensors="pt")
        # Determine binary label: 1 if any frame label > 0.5, else 0
        binary_label = 1.0 if max(labels) > 0.5 else 0.0
        return processed["pixel_values"].squeeze(0), torch.tensor(binary_label, dtype=torch.float32)
