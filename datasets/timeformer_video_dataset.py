# timesformer_pipeline/datasets/timesformer_video_dataset.py
import torch
import torchvision.io as tv_io
import os
import pandas as pd
import numpy as np
import cv2 # For metadata if tv_io info is not sufficient

# You can keep your compute_frame_label function here or move it to a common utils file
def compute_frame_label(t, alert_time, sigma_before=2.0, sigma_after=0.5, atol=0.18):
    # (Your existing compute_frame_label logic)
    if pd.isna(alert_time):
        return 0.0
    if np.isclose(t, alert_time, atol=atol):
        return 1.0
    if t < alert_time:
        return np.exp(-((alert_time - t)**2) / (2 * sigma_before**2))
    else: # t > alert_time
        return np.exp(-((t - alert_time)**2) / (2 * sigma_after**2))

class TimesformerVideoDataset(torch.utils.data.Dataset):
    def __init__(self, df, video_dir, video_transforms,
                 num_clip_frames: int, # Number of frames per clip for TimeSformer (e.g., 8 or 16)
                 target_processing_fps: int, # FPS to aim for when selecting the 10s window
                 sequence_window_seconds: float = 10.0):

        self.df = df.reset_index(drop=True)
        self.video_dir = video_dir
        self.video_transforms = video_transforms # The transform pipeline from video_utils.py
        self.num_clip_frames = num_clip_frames
        self.target_processing_fps = target_processing_fps
        self.sequence_window_seconds = sequence_window_seconds
        self.atol_val = 1.0 / self.target_processing_fps if self.target_processing_fps > 0 else 0.18

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_id = str(row["id"]).zfill(5)
        video_path = os.path.join(self.video_dir, f"{video_id}.mp4")

        # Create default/placeholder tensors (CTHW format after transforms)
        # Transform usually takes TCHW and outputs CTHW for TimeSformer.
        # So, placeholder before transform should be TCHW. Let's assume 224x224 target.
        default_frames_tensor_tchw_uint8 = torch.zeros(
            (self.num_clip_frames, 3, 224, 224), dtype=torch.uint8
        )
        # Apply transforms to placeholder to get correct output shape and type for default
        default_transformed_clip = self.video_transforms({"video": default_frames_tensor_tchw_uint8})["video"]
        default_frame_labels = torch.zeros(self.num_clip_frames, dtype=torch.float32)
        default_binary_label = torch.tensor(0.0, dtype=torch.float32)

        if not os.path.exists(video_path):
            # print(f"Warning: Video file not found {video_path}. Returning placeholders.")
            return {
                "video_clip": default_transformed_clip,
                "frame_labels": default_frame_labels,
                "binary_label": default_binary_label,
                "video_id": video_id,
                "is_valid": torch.tensor(False) # Flag for invalid sample
            }

        try:
            # Get video metadata (duration, original FPS)
            # Using OpenCV for robust metadata, as tv_io.read_video_timestamps can be slow/problematic
            cap = cv2.VideoCapture(video_path)
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            total_original_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if original_fps <= 0 or total_original_frames <= 0:
                raise ValueError(f"Invalid metadata for {video_id}: FPS {original_fps}, Frames {total_original_frames}")
            original_duration_sec = total_original_frames / original_fps

            # --- Windowing logic (adapted from your VideoFrameDataset) ---
            alert_time_sec = row["time_of_alert"] # This is in seconds
            is_positive_event = not pd.isna(alert_time_sec)

            tta_for_window_end = np.random.uniform(0.5, 1.5) # As in your original code for positive cases
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

            # Ensure window is valid
            if window_start_time_sec >= window_end_time_sec:
                if original_duration_sec > window_start_time_sec and original_duration_sec > 0:
                    window_end_time_sec = window_start_time_sec + (1.0 / original_fps) # Min 1 frame duration
                else: # Cannot define a valid window, try taking first 10s or full video if shorter
                    window_start_time_sec = 0.0
                    window_end_time_sec = min(self.sequence_window_seconds, original_duration_sec)
                    if window_start_time_sec >= window_end_time_sec: # Still invalid
                         raise ValueError(f"Invalid window for {video_id}")
            window_end_time_sec = min(window_end_time_sec, original_duration_sec) # Ensure not exceeding duration

            # --- Read video segment ---
            # tv_io.read_video expects TCHW output format by default for frames
            vframes_segment_tchw_uint8, _, info = tv_io.read_video(
                video_path, start_pts=window_start_time_sec, end_pts=window_end_time_sec,
                pts_unit='sec', output_format="TCHW"
            )
            num_read_frames_in_segment = vframes_segment_tchw_uint8.shape[0]

            if num_read_frames_in_segment == 0:
                 raise ValueError(f"Read 0 frames from segment for {video_id}")

            # --- Apply video transforms (includes UniformTemporalSubsample to get self.num_clip_frames) ---
            # The transform expects a dictionary with a "video" key
            transformed_data = self.video_transforms({"video": vframes_segment_tchw_uint8})
            final_clip_tensor = transformed_data["video"] # Shape: C, T_clip, H, W

            # --- Generate frame labels for the `self.num_clip_frames` ---
            # We need the original timestamps of the frames *selected by UniformTemporalSubsample*
            # This is tricky as UniformTemporalSubsample doesn't directly return indices.
            # Approximation: assume UniformTemporalSubsample picks evenly spaced frames from the *read segment*.
            # The frames in the segment correspond to an effective processing FPS.
            
            # Timestamps for frames within the *read segment* (num_read_frames_in_segment)
            segment_effective_fps = info.get('video_fps', self.target_processing_fps) # Use info if available
            # More robust way:
            # segment_duration_read = vframes_segment_tchw_uint8.shape[0] / segment_effective_fps
            # actual_window_duration_sec = window_end_time_sec - window_start_time_sec
            # If segment_effective_fps is from `info`, it's about the source.
            # We are selecting frames from the *loaded* segment.

            # Create timestamps for the `num_read_frames_in_segment` we loaded:
            original_indices_in_segment = np.linspace(0, num_read_frames_in_segment - 1, num_read_frames_in_segment)
            # Map these segment indices back to original video time (seconds)
            # Start time of the segment + offset within the segment
            # FPS of the read segment could be `info['video_fps']` or calculated if not available
            fps_of_read_segment = info.get("video_fps", original_fps) # Use original_fps as a fallback
            if fps_of_read_segment <= 0: fps_of_read_segment = 30.0 # Default if bad metadata

            timestamps_for_read_segment_frames_sec = window_start_time_sec + (original_indices_in_segment / fps_of_read_segment)
            
            # Now, which of these did UniformTemporalSubsample pick for the final `self.num_clip_frames`?
            # Assume it picks evenly from the `timestamps_for_read_segment_frames_sec`
            if num_read_frames_in_segment < self.num_clip_frames:
                # If fewer frames read than needed, UniformTemporalSubsample might duplicate.
                # Or, it might error. PytorchVideo's UTS handles this by taking all available and padding/repeating.
                # For simplicity, let's assume if fewer frames, we use all of them and pad labels later or ensure enough frames.
                # The transforms will handle the frame count to `self.num_clip_frames`.
                # So, we generate labels for the effective frames sampled.
                 indices_picked_by_uts = np.linspace(0, num_read_frames_in_segment - 1, self.num_clip_frames, dtype=int, endpoint=True)
            else:
                 indices_picked_by_uts = np.linspace(0, num_read_frames_in_segment - 1, self.num_clip_frames, dtype=int, endpoint=True)

            selected_original_timestamps_sec = timestamps_for_read_segment_frames_sec[indices_picked_by_uts]

            frame_labels_list = [
                compute_frame_label(ts, alert_time_sec, atol=self.atol_val)
                for ts in selected_original_timestamps_sec
            ]
            frame_labels = torch.tensor(frame_labels_list, dtype=torch.float32)
            
            # Ensure frame_labels has length `self.num_clip_frames` (UTS should ensure output clip has this many frames)
            if len(frame_labels) != self.num_clip_frames:
                # This might happen if UTS logic is complex with very short inputs.
                # Pad or truncate if necessary, though ideally UTS output and labels match.
                # print(f"Warning: Label length mismatch for {video_id}. Got {len(frame_labels)}, expected {self.num_clip_frames}")
                padded_labels = torch.zeros(self.num_clip_frames, dtype=torch.float32)
                actual_len = min(len(frame_labels), self.num_clip_frames)
                padded_labels[:actual_len] = frame_labels[:actual_len]
                frame_labels = padded_labels


            # Determine sequence-level (binary) label for the clip
            binary_label_val = 0.0
            if is_positive_event:
                # A simple way: if any frame in the clip has a high label, or if alert is relevant to the window
                if torch.max(frame_labels) > 0.5: # Example threshold
                    binary_label_val = 1.0
                # More robust: check if alert_time_sec is within/near the *overall window* this clip was sampled from.
                # The problem defines positive examples as collision/near-miss.
                # If row['time_of_alert'] is not NaN, it's a positive sequence.
                # The critical part is if the *sampled clip* reflects this positivity.
                # For simplicity, if the original video is positive, let's initially mark the clip as positive.
                # The frame_labels will then guide the learning of *when* it becomes positive.
                binary_label_val = 1.0


            binary_label = torch.tensor(binary_label_val, dtype=torch.float32)

            return {
                "video_clip": final_clip_tensor,
                "frame_labels": frame_labels,
                "binary_label": binary_label,
                "video_id": video_id,
                "is_valid": torch.tensor(True)
            }

        except Exception as e:
            print(f"ERROR processing video {video_id} ({video_path}): {type(e).__name__} - {e}. Returning placeholders.")
            return {
                "video_clip": default_transformed_clip,
                "frame_labels": default_frame_labels,
                "binary_label": default_binary_label,
                "video_id": video_id,
                "is_valid": torch.tensor(False)
            }

def collate_fn_timesformer_videos(batch):
    # Filter out invalid samples (where "is_valid" is False)
    valid_batch = [item for item in batch if item["is_valid"]]
    if not valid_batch: # All samples in batch were invalid
        # Return a dummy batch structure or raise an error
        # This requires careful handling in the training loop
        print("Warning: All samples in the current batch are invalid.")
        # You might need to return something that the training loop can gracefully skip
        # For now, let's assume the caller handles an empty list or similar
        return None # Or a structure with empty tensors

    video_ids = [item["video_id"] for item in valid_batch]
    video_clips = torch.stack([item["video_clip"] for item in valid_batch])
    frame_labels_batch = torch.stack([item["frame_labels"] for item in valid_batch])
    binary_labels_batch = torch.stack([item["binary_label"] for item in valid_batch])

    return {
        "video_clip": video_clips,
        "frame_labels": frame_labels_batch,
        "binary_label": binary_labels_batch,
        "video_id": video_ids,
    }