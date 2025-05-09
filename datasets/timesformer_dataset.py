import torch
import torchvision.io as tv_io
import os
import pandas as pd
import numpy as np
import cv2 # For metadata
from transformers.models.auto.image_processing_auto import AutoImageProcessor

# Keep your compute_frame_label function (can be in a utils file too)
def compute_frame_label(t, alert_time, sigma_before=2.0, sigma_after=0.5, atol=0.18):
    if pd.isna(alert_time): return 0.0
    if np.isclose(t, alert_time, atol=atol): return 1.0
    if t < alert_time: return np.exp(-((alert_time - t)**2) / (2 * sigma_before**2))
    else: return np.exp(-((t - alert_time)**2) / (2 * sigma_after**2))

class HFVideoDataset(torch.utils.data.Dataset):
    def __init__(self, df, video_dir,
                 hf_processor_name: str, # e.g., "facebook/timesformer-base-finetuned-k400"
                 num_clip_frames: int,   # Number of frames the HF TimeSformer expects
                 target_processing_fps: int, # For your windowing and label alignment
                 sequence_window_seconds: float = 10.0):
        self.df = df.reset_index(drop=True)
        self.video_dir = video_dir
        self.processor = AutoImageProcessor.from_pretrained(hf_processor_name)
        self.num_clip_frames = num_clip_frames
        self.target_processing_fps = target_processing_fps # Used for your internal frame timing
        self.sequence_window_seconds = sequence_window_seconds
        self.atol_val = 1.0 / self.target_processing_fps if self.target_processing_fps > 0 else 0.18

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_id = str(row["id"]).zfill(5)
        video_path = os.path.join(self.video_dir, f"{video_id}.mp4")

        # --- Create default/placeholder outputs ---
        # Determine expected image size from processor if possible, else default
        proc_size_config = getattr(self.processor, 'size', None)
        if isinstance(proc_size_config, dict): # Newer HF versions have size as dict
            # Look for common keys like 'shortest_edge', 'height', 'width'
            h = proc_size_config.get('height', proc_size_config.get('shortest_edge', 224))
            w = proc_size_config.get('width', proc_size_config.get('shortest_edge', 224))
        elif isinstance(proc_size_config, (int, float)): # Older might just have an int
            h = w = int(proc_size_config)
        else: # Fallback
            h = w = 224

        dummy_frames_for_processor = [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(self.num_clip_frames)]
        try:
            default_processed_output = self.processor(images=dummy_frames_for_processor, return_tensors="pt", do_rescale=False if hasattr(self.processor, 'do_rescale') else True) # Some processors rescale by default
            default_pixel_values = default_processed_output["pixel_values"]
        except Exception: # Fallback if processor errors on zero frames
            default_pixel_values = torch.zeros(self.num_clip_frames, 3, h, w)


        default_frame_labels = torch.zeros(self.num_clip_frames, dtype=torch.float32)
        default_binary_label = torch.tensor(0.0, dtype=torch.float32)
        return_dict_on_error = {
            "pixel_values": default_pixel_values, "frame_labels": default_frame_labels,
            "binary_label": default_binary_label, "video_id": video_id, "is_valid": torch.tensor(False)
        }

        if not os.path.exists(video_path): return return_dict_on_error

        try:
            cap = cv2.VideoCapture(video_path)
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            total_original_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if original_fps <= 0 or total_original_frames <= 0: raise ValueError("Invalid video metadata")
            original_duration_sec = total_original_frames / original_fps

            alert_time_sec = row["time_of_alert"]
            is_positive_event = not pd.isna(alert_time_sec)

            # --- Windowing logic (reuse from your previous dataset) ---
            tta_for_window_end = np.random.uniform(0.5, 1.5)
            window_start_time_sec, window_end_time_sec = 0.0, 0.0
            if not is_positive_event: # Negative case
                window_start_time_sec = 0.0
                window_end_time_sec = min(self.sequence_window_seconds, original_duration_sec)
            elif alert_time_sec < self.sequence_window_seconds: # Positive, alert early
                window_start_time_sec = 0.0
                window_end_time_sec = min(self.sequence_window_seconds, original_duration_sec)
            else: # Positive, alert later
                window_end_time_sec = min(alert_time_sec + tta_for_window_end, original_duration_sec)
                window_start_time_sec = max(0.0, window_end_time_sec - self.sequence_window_seconds)

            if window_start_time_sec >= window_end_time_sec:
                window_start_time_sec = 0.0
                window_end_time_sec = min(self.sequence_window_seconds, original_duration_sec)
                if window_start_time_sec >= window_end_time_sec: raise ValueError(f"Cannot define valid read window for {video_id}")
            window_end_time_sec = min(window_end_time_sec, original_duration_sec)

            # --- Read video segment (TCHW uint8) ---
            vframes_segment_tchw_uint8, _, info = tv_io.read_video(
                video_path, start_pts=window_start_time_sec, end_pts=window_end_time_sec,
                pts_unit='sec', output_format="TCHW"
            )
            num_read_frames_in_segment = vframes_segment_tchw_uint8.shape[0]
            if num_read_frames_in_segment == 0: raise ValueError(f"Read 0 frames from segment for {video_id}")

            # --- Uniformly sample `self.num_clip_frames` from the read segment ---
            if num_read_frames_in_segment < self.num_clip_frames:
                indices_to_sample = np.pad(np.arange(num_read_frames_in_segment),
                                           (0, self.num_clip_frames - num_read_frames_in_segment), 'edge')
            else:
                indices_to_sample = np.linspace(0, num_read_frames_in_segment - 1, self.num_clip_frames, dtype=int, endpoint=True)
            sampled_frames_tchw_uint8 = vframes_segment_tchw_uint8[indices_to_sample]

            # --- Convert to list of HWC NumPy arrays for Hugging Face processor ---
            frames_for_processor = [frame.permute(1, 2, 0).numpy() for frame in sampled_frames_tchw_uint8]

            # --- Process with Hugging Face processor ---
            processed_output = self.processor(images=frames_for_processor, return_tensors="pt", do_rescale=False if hasattr(self.processor, 'do_rescale') else True)
            pixel_values = processed_output["pixel_values"] # Expected: (T_clip, C, H, W)

            # --- Generate frame labels for the `self.num_clip_frames` ---
            fps_of_read_segment = info.get("video_fps", original_fps)
            if fps_of_read_segment <= 0: fps_of_read_segment = self.target_processing_fps # Fallback

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
            # print(f"ERROR processing {video_id}: {e}") # Reduce verbosity for production
            return return_dict_on_error

# Your collate_fn_hf_videos (from previous thought process) can be reused here.
# Ensure it handles the output of this dataset correctly.
# Key part: `pixel_values_batch = torch.stack(pixel_values_list)` will create (B, T_clip, C, H, W)
def collate_fn_hf_videos(batch):
    valid_batch = [item for item in batch if item["is_valid"]]
    if not valid_batch: return None

    pixel_values_list = [item["pixel_values"] for item in valid_batch] # List of (T_clip, C, H, W)
    pixel_values_batch = torch.stack(pixel_values_list) # -> (B, T_clip, C, H, W)

    frame_labels_batch = torch.stack([item["frame_labels"] for item in valid_batch])
    binary_labels_batch = torch.stack([item["binary_label"] for item in valid_batch])
    video_ids = [item["video_id"] for item in valid_batch]

    return {
        "pixel_values": pixel_values_batch, "frame_labels": frame_labels_batch,
        "binary_label": binary_labels_batch, "video_id": video_ids,
    }