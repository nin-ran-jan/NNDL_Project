import torch
import torchvision.io as tv_io
import os
import pandas as pd
import numpy as np
import cv2 # For metadata
from transformers.models.auto.image_processing_auto import AutoImageProcessor


from utils.video_transforms import get_video_augmentation_transforms # Import new transforms

def compute_frame_label(t, alert_time, sigma_before=2.0, sigma_after=0.5, atol=0.18):
    if pd.isna(alert_time): return 0.0
    if np.isclose(t, alert_time, atol=atol): return 1.0
    if t < alert_time: return np.exp(-((alert_time - t)**2) / (2 * sigma_before**2))
    else: return np.exp(-((t - alert_time)**2) / (2 * sigma_after**2))

class HFVideoDataset(torch.utils.data.Dataset):
    def __init__(self, df, video_dir,
                 hf_processor_name: str,
                 num_clip_frames: int,
                 target_processing_fps: int,
                 sequence_window_seconds: float = 10.0,
                 is_train: bool = False, # To apply different transforms
                 augmentation_spatial_size: tuple = (224, 224) # For aug transforms
                ):
        self.df = df.reset_index(drop=True)
        self.video_dir = video_dir
        try:
            self.processor = AutoImageProcessor.from_pretrained(hf_processor_name)
        except Exception as e:
            print(f"Error loading HuggingFace processor {hf_processor_name}: {e}")
            raise
        self.num_clip_frames = num_clip_frames
        self.target_processing_fps = target_processing_fps
        self.sequence_window_seconds = sequence_window_seconds
        self.atol_val = 1.0 / self.target_processing_fps if self.target_processing_fps > 0 else 0.18
        self.is_train = is_train
        self.augmentation_spatial_size = augmentation_spatial_size

        # Initialize augmentation pipeline
        self.augmentation_pipeline = get_video_augmentation_transforms(
            is_train=self.is_train,
            target_spatial_size=self.augmentation_spatial_size
        )

    def __len__(self):
        return len(self.df)

    def _get_placeholder_pixel_values(self):
        # ... (previous _get_placeholder_pixel_values logic, ensuring it produces a 4D tensor
        #      of shape (self.num_clip_frames, 3, H, W) consistent with processor output)
        proc_size_config = getattr(self.processor, 'size', {}) # ViTImageProcessor has "size": {"shortest_edge": 224}
        if isinstance(proc_size_config, dict):
             h = w = proc_size_config.get('shortest_edge', proc_size_config.get('height',224))
        elif isinstance(proc_size_config, (int, float)): h = w = int(proc_size_config)
        else: h = w = 224 # Default

        # Create frames that will go through augmentation and then processor
        dummy_frames_tchw_uint8 = torch.randint(0, 256, (self.num_clip_frames, 3, h, h), dtype=torch.uint8)
        
        # Apply augmentation pipeline (optional for dummy, but good for consistency check)
        # aug_pipeline expects TCHW uint8, outputs TCHW float [0,1]
        # For simplicity in placeholder, we might skip aug if it's complex or error-prone on zeros
        # dummy_frames_tchw_float = self.augmentation_pipeline(dummy_frames_tchw_uint8)
        
        # Processor expects list of HWC numpy/PIL or TCHW/THWC tensors
        # Let's simulate the format processor expects after augmentations (list of HWC numpy)
        # If aug pipeline output TCHW float, convert
        # dummy_frames_for_processor = [f.permute(1, 2, 0).cpu().numpy() * 255 for f in dummy_frames_tchw_float.to(torch.uint8)]
        # Or simpler for placeholder:
        dummy_frames_for_processor = [np.zeros((h,h,3), dtype=np.uint8) for _ in range(self.num_clip_frames)]
        try:
            rescale_arg = {} # Default for ViTImageProcessor is do_rescale=True, normalizes image to [0,1]
            processed_output = self.processor(images=dummy_frames_for_processor, return_tensors="pt", **rescale_arg)
            pixel_values = processed_output["pixel_values"] # (T_clip, C, H_proc, W_proc)
        except Exception:
            pixel_values = torch.zeros(self.num_clip_frames, 3, h, w, dtype=torch.float32) # Fallback

        if pixel_values.ndim == 5 and pixel_values.shape[0] == 1: pixel_values = pixel_values.squeeze(0)
        if not (pixel_values.ndim == 4 and pixel_values.shape[0] == self.num_clip_frames and pixel_values.shape[1] == 3):
            pixel_values = torch.zeros(self.num_clip_frames, 3, h, w, dtype=torch.float32)
        return pixel_values


    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_id = str(row["id"]).zfill(5)
        video_path = os.path.join(self.video_dir, f"{video_id}.mp4")

        default_pixel_values = self._get_placeholder_pixel_values()
        default_frame_labels = torch.zeros(self.num_clip_frames, dtype=torch.float32)
        default_binary_label = torch.tensor(0.0, dtype=torch.float32)
        return_dict_on_error = {
            "pixel_values": default_pixel_values, "frame_labels": default_frame_labels,
            "binary_label": default_binary_label, "video_id": video_id, "is_valid": torch.tensor(False)
        }

        if not os.path.exists(video_path): return return_dict_on_error

        try:
            # --- Video loading and windowing (same as before) ---
            cap = cv2.VideoCapture(video_path)
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            total_original_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if original_fps <= 0 or total_original_frames <= 0: raise ValueError("Invalid video metadata")
            original_duration_sec = total_original_frames / original_fps

            alert_time_sec = row["time_of_alert"]
            is_positive_event = not pd.isna(alert_time_sec)

            tta_for_window_end = np.random.uniform(0.5, 1.5)
            window_start_time_sec, window_end_time_sec = 0.0, 0.0
            # ... (your windowing logic from previous version) ...
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
                if window_start_time_sec >= window_end_time_sec: raise ValueError(f"Cannot define valid read window for {video_id}")
            window_end_time_sec = min(window_end_time_sec, original_duration_sec)


            vframes_segment_tchw_uint8, _, info = tv_io.read_video(
                video_path, start_pts=window_start_time_sec, end_pts=window_end_time_sec,
                pts_unit='sec', output_format="TCHW"
            )
            num_read_frames_in_segment = vframes_segment_tchw_uint8.shape[0]
            if num_read_frames_in_segment == 0: raise ValueError(f"Read 0 frames from segment for {video_id}")

            # --- Uniformly sample `self.num_clip_frames` from the read segment ---
            # This sampling provides the temporal consistency for label generation.
            if num_read_frames_in_segment < self.num_clip_frames:
                indices_to_sample = np.pad(np.arange(num_read_frames_in_segment),
                                           (0, self.num_clip_frames - num_read_frames_in_segment), 'edge')
            else:
                indices_to_sample = np.linspace(0, num_read_frames_in_segment - 1, self.num_clip_frames, dtype=int, endpoint=True)
            
            sampled_frames_tchw_uint8 = vframes_segment_tchw_uint8[indices_to_sample] # (T_clip, C, H, W) uint8

            # --- Apply Augmentations (if training) ---
            # augmentation_pipeline expects TCHW, often uint8, and might output float TCHW [0,1]
            augmented_frames_tchw = self.augmentation_pipeline(sampled_frames_tchw_uint8)
            # Ensure it's float [0,1] if ToTensor was last, or convert if needed by processor
            # The example aug_pipeline ends with ToTensor(), so it should be float [0,1]

            # --- Convert to list of HWC NumPy arrays for Hugging Face processor ---
            # Processor typically handles normalization and final resizing.
            # Input to processor: list of PIL Images or HWC NumPy arrays.
            # If augmented_frames_tchw is float [0,1], convert back to uint8 [0,255] for some processors,
            # or ensure processor is configured for float [0,1] input.
            # Most ViTImageProcessors expect uint8 [0,255] HWC or float [0,1] HWC and handle rescaling internally.
            # Let's provide HWC uint8
            
            frames_for_processor = []
            for i in range(augmented_frames_tchw.shape[0]):
                frame_chw_float = augmented_frames_tchw[i] # C, H, W float [0,1]
                frame_hwc_uint8 = (frame_chw_float.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
                frames_for_processor.append(frame_hwc_uint8)
            
            # --- Process with Hugging Face processor ---
            # `do_rescale=True` is often default for ViTImageProcessor, which scales 0-255 to 0-1.
            # Since our aug output is already float [0,1] with ToTensor(), if we pass uint8,
            # the processor's default `do_rescale=True` and `do_normalize=True` are what we want.
            rescale_arg = {} # ViTImageProcessor usually rescales uint8 [0,255] to float [0,1] by default
            processed_output = self.processor(images=frames_for_processor, return_tensors="pt", **rescale_arg)
            pixel_values = processed_output["pixel_values"] # Expected: (T_clip, C, H_proc, W_proc)

            # Shape correction (ensure 4D for collate_fn)
            if pixel_values.ndim == 5 and pixel_values.shape[0] == 1: pixel_values = pixel_values.squeeze(0)
            if not (pixel_values.ndim == 4 and pixel_values.shape[0] == self.num_clip_frames and pixel_values.shape[1] == 3):
                print(f"Warning: ID {video_id}, final pixel_values shape {pixel_values.shape} is unexpected. Using placeholder.")
                pixel_values = default_pixel_values

            # --- Generate frame labels (based on original timing of sampled frames) ---
            fps_of_read_segment = info.get("video_fps", original_fps)
            if fps_of_read_segment <= 0: fps_of_read_segment = self.target_processing_fps

            timestamps_for_read_segment_frames_sec = window_start_time_sec + (np.arange(num_read_frames_in_segment) / fps_of_read_segment)
            selected_original_timestamps_sec = timestamps_for_read_segment_frames_sec[indices_to_sample]

            frame_labels_list = [compute_frame_label(ts, alert_time_sec, atol=self.atol_val) for ts in selected_original_timestamps_sec]
            frame_labels = torch.tensor(frame_labels_list, dtype=torch.float32)

            binary_label_val = 1.0 if is_positive_event else 0.0
            binary_label = torch.tensor(binary_label_val, dtype=torch.float32)

            return {
                "pixel_values": pixel_values, "frame_labels": frame_labels,
                "binary_label": binary_label, "video_id": video_id, "is_valid": torch.tensor(True)
            }
        except Exception as e:
            # print(f"ERROR processing {video_id} ({video_path}): {type(e).__name__} - {e}")
            return return_dict_on_error

# collate_fn_hf_videos remains the same
def collate_fn_hf_videos(batch):
    # ... (same as your previous working version)
    valid_batch = [item for item in batch if item["is_valid"]]
    if not valid_batch: return None
    pixel_values_list = [item["pixel_values"] for item in valid_batch]
    pixel_values_batch = torch.stack(pixel_values_list)
    frame_labels_batch = torch.stack([item["frame_labels"] for item in valid_batch])
    binary_labels_batch = torch.stack([item["binary_label"] for item in valid_batch])
    video_ids = [item["video_id"] for item in valid_batch]
    return {
        "pixel_values": pixel_values_batch, "frame_labels": frame_labels_batch,
        "binary_label": binary_labels_batch, "video_id": video_ids,
    }