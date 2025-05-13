import torch
import torchvision.io as tv_io
import os
import pandas as pd
import numpy as np
import cv2
from transformers.models.x_clip import XCLIPProcessor

# GPU-based augmentations
import kornia.augmentation as K


def compute_frame_label(t, alert_time, sigma_before=2.0, sigma_after=0.5, atol=0.18):
    if pd.isna(alert_time): return 0.0
    if np.isclose(t, alert_time, atol=atol): return 1.0
    if t < alert_time: return np.exp(-((alert_time - t)**2) / (2 * sigma_before**2))
    else: return np.exp(-((t - alert_time)**2) / (2 * sigma_after**2))


class XCLIPVideoDataset(torch.utils.data.Dataset):
    def __init__(self, df, video_dir,
                 processor_name: str,
                 num_frames: int = 8,
                 target_fps: int = 3,
                 sequence_window_seconds: float = 10.0,
                 is_train: bool = False,
                 augmentation_params: dict = None
                ):
        self.df = df.reset_index(drop=True)
        self.video_dir = video_dir
        self.processor = XCLIPProcessor.from_pretrained(processor_name)
        self.num_frames = num_frames
        self.target_fps = target_fps
        self.sequence_window_seconds = sequence_window_seconds
        self.atol_val = 1.0 / self.target_fps if self.target_fps > 0 else 0.18
        self.is_train = is_train
        
        # Default augmentation parameters
        self.augmentation_params = {
            'strength': 0.5,
            'color_jitter_prob': 0.8,
            'gray_scale_prob': 0.2,
            'random_crop_scale': (0.6, 1.0),
            'random_crop_ratio': (0.75, 1.33)
        }
        
        # Update with user provided params
        if augmentation_params:
            self.augmentation_params.update(augmentation_params)
            
        # Setup GPU augmentations if training
        if self.is_train:
            self.setup_gpu_augmentations()
        else:
            self.eval_transforms = K.Resize(size=(224, 224))
            
    def setup_gpu_augmentations(self):
        # Define augmentations for training (on GPU)
        strength = self.augmentation_params['strength']
        self.train_transforms = K.AugmentationSequential(
            K.RandomResizedCrop(
                size=(224, 224), 
                scale=self.augmentation_params['random_crop_scale'],
                ratio=self.augmentation_params['random_crop_ratio']
            ),
            K.RandomHorizontalFlip(p=0.5),
            K.ColorJitter(
                brightness=0.4 * strength,
                contrast=0.4 * strength,
                saturation=0.2 * strength,
                hue=0.1 * strength,
                p=self.augmentation_params['color_jitter_prob']
            ),
            K.RandomGrayscale(p=self.augmentation_params['gray_scale_prob']),
            # Additional augmentations for diversity
            K.RandomMotionBlur(kernel_size=3, angle=45, direction=0.5, p=0.2),
            K.RandomSharpness(sharpness=0.3, p=0.3),
            K.Normalize(
                mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]),
                std=torch.tensor([0.26862954, 0.26130258, 0.27577711])
            ),
            data_keys=["input"],
        )

    def __len__(self):
        return len(self.df)
    
    def get_placeholder_data(self):
        """Return placeholder data in case of errors"""
        pixel_values = torch.zeros(self.num_frames, 3, 224, 224, dtype=torch.float32)
        frame_labels = torch.zeros(self.num_frames, dtype=torch.float32)
        binary_label = torch.tensor(0.0, dtype=torch.float32)
        return {
            "pixel_values": pixel_values, 
            "frame_labels": frame_labels,
            "binary_label": binary_label, 
            "video_id": "placeholder", 
            "is_valid": torch.tensor(False)
        }

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_id = str(row["id"]).zfill(5)
        video_path = os.path.join(self.video_dir, f"{video_id}.mp4")
        
        if not os.path.exists(video_path):
            return self.get_placeholder_data()
        
        try:
            # Get video metadata
            cap = cv2.VideoCapture(video_path)
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            total_original_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            if original_fps <= 0 or total_original_frames <= 0:
                return self.get_placeholder_data()
                
            original_duration_sec = total_original_frames / original_fps

            # Determine alert time from the data
            alert_time_sec = row.get("time_of_alert", None)
            is_positive_event = not pd.isna(alert_time_sec)
            
            # Define window timing
            tta_for_window_end = np.random.uniform(0.5, 1.5)
            
            if not is_positive_event:
                window_start_time_sec = 0.0
                window_end_time_sec = min(self.sequence_window_seconds, original_duration_sec)
            elif alert_time_sec < self.sequence_window_seconds:
                window_start_time_sec = 0.0
                window_end_time_sec = min(self.sequence_window_seconds, original_duration_sec)
            else:
                window_end_time_sec = min(alert_time_sec + tta_for_window_end, original_duration_sec)
                window_start_time_sec = max(0.0, window_end_time_sec - self.sequence_window_seconds)
            
            if window_start_time_sec >= window_end_time_sec:
                window_start_time_sec = 0.0
                window_end_time_sec = min(self.sequence_window_seconds, original_duration_sec)
                
            if window_start_time_sec >= window_end_time_sec:
                return self.get_placeholder_data()
            
            # Read video segment
            vframes_segment, _, info = tv_io.read_video(
                video_path, 
                start_pts=window_start_time_sec, 
                end_pts=window_end_time_sec,
                pts_unit='sec'
            )
            
            num_read_frames_in_segment = vframes_segment.shape[0]
            if num_read_frames_in_segment == 0:
                return self.get_placeholder_data()
            
            # Sample frames from the segment
            if num_read_frames_in_segment < self.num_frames:
                indices_to_sample = np.pad(
                    np.arange(num_read_frames_in_segment),
                    (0, self.num_frames - num_read_frames_in_segment), 
                    'edge'
                )
            else:
                indices_to_sample = np.linspace(
                    0, num_read_frames_in_segment - 1, 
                    self.num_frames, dtype=int, 
                    endpoint=True
                )
            
            sampled_frames = vframes_segment[indices_to_sample]  # Shape: [num_frames, H, W, C]
            
            # Prepare for GPU augmentations - move to BCHW format
            frames_bchw = sampled_frames.permute(0, 3, 1, 2).float() / 255.0
            
            # Apply augmentations
            # Keep transformations on CPU to prevent unnecessary device transfers
            if self.is_train:
                # Apply strong augmentations for training
                frames_bchw = self.train_transforms(frames_bchw)
            else:
                # Simple resize for evaluation
                frames_bchw = self.eval_transforms(frames_bchw)
                
            # Make sure we return float tensors in the expected range
            if frames_bchw.dtype != torch.float32:
                frames_bchw = frames_bchw.float()
                
            # If we have normalized with Kornia, we don't need another normalization
            
            # Calculate frame timestamps and labels
            fps_of_read_segment = info.get("video_fps", original_fps)
            if fps_of_read_segment <= 0:
                fps_of_read_segment = self.target_fps
            
            timestamps = window_start_time_sec + (indices_to_sample / fps_of_read_segment)
            frame_labels = torch.tensor(
                [compute_frame_label(ts, alert_time_sec, atol=self.atol_val) for ts in timestamps],
                dtype=torch.float32
            )
            
            binary_label = torch.tensor(1.0 if is_positive_event else 0.0, dtype=torch.float32)
            
            return {
                "pixel_values": frames_bchw,  # [num_frames, C, H, W]
                "frame_labels": frame_labels,  # [num_frames]
                "binary_label": binary_label,  # scalar
                "video_id": video_id,
                "is_valid": torch.tensor(True)
            }
            
        except Exception as e:
            # print(f"Error processing video {video_id}: {e}")
            return self.get_placeholder_data()


# Collate function for the dataloader
def collate_fn_xclip(batch):
    valid_batch = [item for item in batch if item["is_valid"]]
    if not valid_batch:
        return None
    
    pixel_values_list = [item["pixel_values"] for item in valid_batch]
    
    # X-CLIP processor expects [batch, num_frames, channels, height, width]
    # Our pixel_values are already in shape [num_frames, channels, height, width]
    # So we stack them into a batch
    pixel_values_batch = torch.stack(pixel_values_list)
    
    frame_labels_batch = torch.stack([item["frame_labels"] for item in valid_batch])
    binary_labels_batch = torch.stack([item["binary_label"] for item in valid_batch])
    video_ids = [item["video_id"] for item in valid_batch]
    
    return {
        "pixel_values": pixel_values_batch,  # [batch, num_frames, C, H, W]
        "frame_labels": frame_labels_batch,  # [batch, num_frames]
        "binary_label": binary_labels_batch,  # [batch]
        "video_id": video_ids,
    }