# datasets/gpu_video_dataset.py
import torch
import torchvision.io as tv_io
import torchvision.transforms.functional as TF
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize # For CLIP
import os
import pandas as pd
import numpy as np
import cv2 # Temporary: For robust FPS and total_frame count, can be replaced if a pure torchvision way is found

def compute_frame_label(t, alert_time, sigma_before=2.0, sigma_after=0.5, atol=0.18):
    if pd.isna(alert_time):
        return 0.0
    if np.isclose(t, alert_time, atol=atol):
        return 1.0
    if t < alert_time:
        return np.exp(-((alert_time - t)**2) / (2 * sigma_before**2))
    else:
        return np.exp(-((t - alert_time)**2) / (2 * sigma_after**2))

class VideoFrameDataset(torch.utils.data.Dataset):
    def __init__(self, df, video_dir, fps_target, sequence_length, 
                 clip_processor=None, target_device='cpu'):
        self.df = df.reset_index(drop=True)
        self.video_dir = video_dir
        self.fps_target = fps_target
        self.sequence_length = sequence_length
        self.atol_val = 1.0 / self.fps_target if self.fps_target > 0 else 0.18
        self.target_device = target_device # Device to move tensors to ultimately
        
        # If a CLIP processor is provided, we can use its image preprocessing
        # Otherwise, a generic one might be needed if frames are directly fed to model
        self.clip_processor = clip_processor
        if self.clip_processor is None:
            # Fallback generic transforms if no processor (e.g. resize, to_tensor, normalize)
            # CLIP default input size is often 224x224
            image_size = 224
            self.transform = Compose([
                Resize(image_size, interpolation=TF.InterpolationMode.BICUBIC),
                CenterCrop(image_size),
                lambda image: image.float() / 255.0, # Convert to float [0,1]
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        else:
            # The processor itself will handle transformations when called with PIL images or uint8 tensors
            self.transform = None # Processor handles it

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_id = row["id"]
        video_path = os.path.join(self.video_dir, f"{video_id}.mp4")

        # Placeholder for default return values on error
        # CLIP typically expects 3 channels.
        # Assuming processor handles resize, so original H,W might not matter if frames can be read
        # For placeholder, use a common small size if processor is not available.
        placeholder_size = 224 if self.clip_processor else 224 # Default size for placeholder
        default_frames_tensor = torch.zeros((self.sequence_length, 3, placeholder_size, placeholder_size), dtype=torch.float32)
        default_labels_tensor = torch.zeros(self.sequence_length, dtype=torch.float32)

        if not os.path.exists(video_path):
            print(f"Warning: Video file not found {video_path}. Returning placeholders for {video_id}.")
            return video_id, default_frames_tensor, default_labels_tensor

        try:
            # Using OpenCV temporarily for robust metadata, as torchvision's metadata can be tricky/slow
            cap_meta = cv2.VideoCapture(video_path)
            fps_cv = cap_meta.get(cv2.CAP_PROP_FPS)
            total_frames_cv = int(cap_meta.get(cv2.CAP_PROP_FRAME_COUNT))
            cap_meta.release()

            fps = fps_cv if fps_cv > 0 and not np.isnan(fps_cv) else 30.0
            duration = total_frames_cv / fps if fps > 0 and total_frames_cv > 0 else 0.0
            
            if total_frames_cv == 0 : # Video is empty or unreadable
                 raise ValueError("Video has 0 frames or is unreadable.")


            alert_time = row["time_of_alert"]
            is_positive = not pd.isna(alert_time)

            tta = np.random.uniform(0.5, 1.5)
            window_start_time_sec, window_end_time_sec = 0.0, 0.0

            if not is_positive:
                window_start_time_sec = 0.0
                window_end_time_sec = min(10.0, duration)
            elif alert_time < 10.0:
                window_start_time_sec = 0.0
                window_end_time_sec = min(10.0, duration)
            else:
                window_end_time_sec = min(alert_time + tta, duration)
                window_start_time_sec = max(0.0, window_end_time_sec - 10.0)
            
            # Ensure window is valid
            if window_start_time_sec >= window_end_time_sec :
                if duration > window_start_time_sec and duration > 0 : # if end is too small, read a tiny bit
                    window_end_time_sec = window_start_time_sec + (1.0 / fps) # min 1 frame duration
                else: # Cannot define a valid window
                    raise ValueError(f"Cannot define a valid read window for {video_id}. Start: {window_start_time_sec}, End: {window_end_time_sec}, Duration: {duration}")
            window_end_time_sec = min(window_end_time_sec, duration)


            # Read video segment using torchvision.io
            # This attempts to use hardware acceleration if PyTorch is built with it.
            # `read_video` loads frames into CPU memory as uint8 tensors (T, H, W, C) by default.
            # We specify output_format="TCHW" for (T, C, H, W)
            vframes_tchw_uint8, _, info = tv_io.read_video(
                video_path,
                start_pts=window_start_time_sec,
                end_pts=window_end_time_sec,
                pts_unit='sec',
                output_format="TCHW" # T, C, H, W and uint8
            )

            num_read_frames = vframes_tchw_uint8.shape[0]

            final_frames_tensor = default_frames_tensor
            labels_list = [0.0] * self.sequence_length

            if num_read_frames > 0:
                # Subsample to self.sequence_length frames
                selected_indices_in_vframes = np.linspace(0, num_read_frames - 1, self.sequence_length, dtype=int, endpoint=True)
                subsampled_frames_tchw_uint8 = vframes_tchw_uint8[selected_indices_in_vframes] # (seq_len, C, H, W)

                # Calculate labels
                # For label calculation, use original video fps and intended sampling pattern for timing
                orig_start_frame_idx = int(window_start_time_sec * fps)
                orig_end_frame_idx = min(int(window_end_time_sec * fps), total_frames_cv -1)
                if orig_start_frame_idx > orig_end_frame_idx: orig_start_frame_idx = orig_end_frame_idx
                
                # These are the frame indices in the *original* video we aimed for
                original_sampled_indices_for_label = np.linspace(orig_start_frame_idx, orig_end_frame_idx, self.sequence_length, dtype=int, endpoint=True)

                for i in range(self.sequence_length):
                    # Timestamp for label calculation based on original video's FPS and intended sample
                    t_for_label = original_sampled_indices_for_label[i] / fps
                    labels_list[i] = compute_frame_label(t_for_label, alert_time, atol=self.atol_val)

                # The CLIPProcessor expects a list of PIL Images or a batch of pixel_values.
                # If we pass uint8 tensors, it should handle it.
                # If self.transform is defined (no clip_processor), apply it
                if self.transform: # Not using CLIP processor directly here
                    # Convert TCHW (uint8) to list of CHW (float) for transform
                    processed_frames_list = []
                    for i in range(subsampled_frames_tchw_uint8.shape[0]):
                        frame_chw_uint8 = subsampled_frames_tchw_uint8[i] # CHW
                        processed_frames_list.append(self.transform(frame_chw_uint8))
                    final_frames_tensor = torch.stack(processed_frames_list) # (seq_len, C, H, W) float32 normalized
                else:
                    # If CLIP processor will be used later, keep as uint8 TCHW or convert to list of PIL
                    # For simplicity, Vit_feature_extract will take this uint8 tensor.
                    final_frames_tensor = subsampled_frames_tchw_uint8 # (seq_len, C, H, W) uint8
            
            labels_tensor = torch.tensor(labels_list, dtype=torch.float32)
            
            # Move to target_device if specified (usually done in DataLoader's collate_fn or main loop)
            # For now, let __getitem__ return CPU tensors; DataLoader can move batches to GPU.
            # This is safer with num_workers > 0.
            return video_id, final_frames_tensor.cpu(), labels_tensor.cpu()

        except Exception as e:
            print(f"Error processing video {video_id} ({video_path}): {e}. Returning placeholders.")
            return video_id, default_frames_tensor.cpu(), default_labels_tensor.cpu()

def collate_fn_videos(batch):
    video_ids = [item[0] for item in batch]
    # item[1] is (T, C, H, W)
    # item[2] is (T,)
    
    # Pad frame tensors if they have different T (sequence_length) - should not happen with this dataset design
    # Pad label tensors if they have different T - should not happen
    frames_list = [item[1] for item in batch]
    labels_list = [item[2] for item in batch]

    # If all items have the same shape, stack them.
    # This creates (B, T, C, H, W) for frames and (B, T) for labels.
    try:
        frames_batch = torch.stack(frames_list, dim=0)
        labels_batch = torch.stack(labels_list, dim=0)
    except RuntimeError as e:
        # Handle cases where stacking might fail due to inconsistent shapes (e.g., from errors)
        print(f"Error during collate_fn stacking: {e}. Check placeholder shapes or errors in __getitem__.")
        # Fallback: return lists, or handle more gracefully
        # This indicates an issue in __getitem__ if shapes are not consistent for sequence_length
        # For now, let it raise or return uncollated if needed.
        # A robust collate_fn would pad, but our dataset should give fixed seq_len.
        return video_ids, frames_list, labels_list # Fallback

    return video_ids, frames_batch, labels_batch