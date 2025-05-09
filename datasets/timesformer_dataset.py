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
                 hf_processor_name: str,
                 num_clip_frames: int,
                 target_processing_fps: int,
                 sequence_window_seconds: float = 10.0):
        # ... (init attributes as before) ...
        self.df = df.reset_index(drop=True)
        self.video_dir = video_dir
        try:
            self.processor = AutoImageProcessor.from_pretrained(hf_processor_name)
        except Exception as e:
            print(f"Error loading HuggingFace processor {hf_processor_name}: {e}")
            print("Please ensure the processor name is correct and you have an internet connection.")
            raise
        self.num_clip_frames = num_clip_frames
        self.target_processing_fps = target_processing_fps
        self.sequence_window_seconds = sequence_window_seconds
        self.atol_val = 1.0 / self.target_processing_fps if self.target_processing_fps > 0 else 0.18


    def __len__(self):
        return len(self.df)

    def _get_placeholder_pixel_values(self):
        # Helper to create consistent placeholder
        proc_size_config = getattr(self.processor, 'size', None)
        if isinstance(proc_size_config, dict):
            h = proc_size_config.get('height', proc_size_config.get('shortest_edge', 224))
            w = proc_size_config.get('width', proc_size_config.get('shortest_edge', 224))
        elif isinstance(proc_size_config, (int, float)):
            h = w = int(proc_size_config)
        else:
            h = w = 224
        
        dummy_frames_for_processor = [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(self.num_clip_frames)]
        try:
            # Use do_rescale=False if processor has it, some newer ones handle it via other flags or by default.
            # This ensures we don't double-scale if we already scaled to [0,1]
            rescale_arg = {}
            if 'do_rescale' in self.processor.__init__.__code__.co_varnames: # Check if processor supports do_rescale
                 rescale_arg['do_rescale'] = False

            processed_output = self.processor(images=dummy_frames_for_processor, return_tensors="pt", **rescale_arg)
            pixel_values = processed_output["pixel_values"]
        except Exception as e:
            print(f"Warning: Processor errored on dummy frames: {e}. Using zero tensor.")
            pixel_values = torch.zeros(self.num_clip_frames, 3, h, w, dtype=torch.float32)

        # Ensure 4D: (num_clip_frames, C, H, W)
        if pixel_values.ndim == 5 and pixel_values.shape[0] == 1:
            pixel_values = pixel_values.squeeze(0)
        elif pixel_values.ndim == 3 and pixel_values.shape[0] == 3 : # Potentially (C, H, W) for each, stacked makes (T,C,H,W) - this check might be too specific
            # This case is usually fine if processor output for list is (T,C,H,W)
            pass
        
        # Final check for the expected 4D shape
        if not (pixel_values.ndim == 4 and pixel_values.shape[0] == self.num_clip_frames and pixel_values.shape[1] == 3):
            print(f"Warning: Default pixel_values has unexpected shape {pixel_values.shape}. Adjusting to zeros.")
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
            # ... (video metadata loading, windowing logic - remains the same) ...
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

            if num_read_frames_in_segment < self.num_clip_frames:
                indices_to_sample = np.pad(np.arange(num_read_frames_in_segment),
                                           (0, self.num_clip_frames - num_read_frames_in_segment), 'edge')
            else:
                indices_to_sample = np.linspace(0, num_read_frames_in_segment - 1, self.num_clip_frames, dtype=int, endpoint=True)
            sampled_frames_tchw_uint8 = vframes_segment_tchw_uint8[indices_to_sample]
            frames_for_processor = [frame.permute(1, 2, 0).numpy() for frame in sampled_frames_tchw_uint8]

            # --- Process with Hugging Face processor ---
            # Check if processor expects rescaling or if frames are already [0,1]
            # ViTImageProcessor by default rescales unless do_rescale=False
            # Your original code had `do_rescale=False if hasattr(self.processor, 'do_rescale') else True`
            # If your frames_for_processor are uint8 (0-255), then do_rescale=True (default) is fine.
            # If they were already float [0,1], then do_rescale=False.
            # Since frames_for_processor are from uint8, default rescaling is usually okay.
            processed_output = self.processor(images=frames_for_processor, return_tensors="pt")
            pixel_values = processed_output["pixel_values"]

            # **Crucial Shape Correction:**
            # Ensure pixel_values is 4D: (num_clip_frames, C, H, W)
            if pixel_values.ndim == 5 and pixel_values.shape[0] == 1:
                # This happens if the processor treats the list of frames as a batch of one video
                pixel_values = pixel_values.squeeze(0)
            elif pixel_values.ndim == 3 and pixel_values.shape[0] == 3: # If it returned (C,H,W) per frame and they got stacked to (T,C,H,W)
                 # This condition might be too specific; standard HF processors for lists of images usually return (N, C, H, W)
                 pass # This means pixel_values is already (T, C, H, W)
            
            # After potential squeeze, verify the shape is as expected for a single item (4D)
            if not (pixel_values.ndim == 4 and \
                    pixel_values.shape[0] == self.num_clip_frames and \
                    pixel_values.shape[1] == 3): # Assuming 3 channels
                error_shape = pixel_values.shape
                # Fallback or raise error if shape is still not right
                print(f"Warning: Video ID {video_id}, corrected pixel_values shape {error_shape} is still unexpected. Using placeholder.")
                pixel_values = default_pixel_values # Fallback to a consistently shaped placeholder

            # ... (frame label generation - remains the same) ...
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
            # print(f"ERROR processing {video_id}: {type(e).__name__} - {e}") # Be careful with too much printing in __getitem__
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