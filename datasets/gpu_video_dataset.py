# datasets/gpu_video_dataset.py
import torch
import torchvision.io as tv_io
import torchvision.transforms.functional as TF
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize # Removed ToTensor, will handle manually
import os
import pandas as pd
import numpy as np
import cv2 # Temporary: For robust FPS and total_frame count

# Define a picklable function or class for the transformation
class ToFloatAndDivideBy255:
    def __call__(self, image_tensor):
        return image_tensor.float() / 255.0

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
                 clip_processor=None, target_device='cpu'): # target_device param added but not used heavily in __init__
        self.df = df.reset_index(drop=True)
        self.video_dir = video_dir
        self.fps_target = fps_target
        self.sequence_length = sequence_length
        self.atol_val = 1.0 / self.fps_target if self.fps_target > 0 else 0.18
        # self.target_device = target_device # Not directly used for tensor creation in __init__


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_id = row["id"]
        video_path = os.path.join(self.video_dir, f"{video_id}.mp4")
        
        # for debugging
        # print(f"\n--- Debugging Video ID: {video_id} ---")
        # print(f"Raw row data: time_of_event={row.get('time_of_event', 'N/A')}, time_of_alert={row.get('time_of_alert', 'N/A')}")

        # Placeholder as uint8, as this is what we'll return for actual frames
        # Processor will resize, so exact placeholder H,W less critical, but use something:
        placeholder_h, placeholder_w = 224, 224
        default_frames_tensor = torch.zeros((self.sequence_length, 3, placeholder_h, placeholder_w), dtype=torch.uint8)
        default_labels_tensor = torch.zeros(self.sequence_length, dtype=torch.float32)

        if not os.path.exists(video_path):
            # print(f"Warning: Video file not found {video_path}. Returning placeholders for {video_id}.") # Can be too verbose
            return video_id, default_frames_tensor, default_labels_tensor

        try:
            cap_meta = cv2.VideoCapture(video_path)
            fps_cv = cap_meta.get(cv2.CAP_PROP_FPS)
            total_frames_cv = int(cap_meta.get(cv2.CAP_PROP_FRAME_COUNT))
            cap_meta.release()


            fps = fps_cv if fps_cv > 0 and not np.isnan(fps_cv) else 30.0
            duration = total_frames_cv / fps if fps > 0 and total_frames_cv > 0 else 0.0

            # debug statements
            # print(f"  Metadata: fps={fps:.2f}, duration={duration:.2f}s, total_frames_cv={total_frames_cv}")
            if total_frames_cv == 0:
                print(f"  ERROR: total_frames_cv is 0 for {video_id}. Raising ValueError.")
                raise ValueError("Video has 0 frames or is unreadable.")
            
            if total_frames_cv == 0:
                 raise ValueError("Video has 0 frames or is unreadable.")

            alert_time = row["time_of_alert"]
            is_positive = not pd.isna(alert_time)
            tta = np.random.uniform(0.5, 1.5)
            window_start_time_sec, window_end_time_sec = 0.0, 0.0


            # debug statements
            # print(f"  Alert Info: alert_time={alert_time}, is_positive={is_positive}")

            if not is_positive:
                window_start_time_sec = 0.0
                window_end_time_sec = min(10.0, duration)
            elif alert_time < 10.0:
                window_start_time_sec = 0.0
                window_end_time_sec = min(10.0, duration)
            else:
                window_end_time_sec = min(alert_time + tta, duration)
                window_start_time_sec = max(0.0, window_end_time_sec - 10.0)
            
            if window_start_time_sec >= window_end_time_sec :
                if duration > window_start_time_sec and duration > 0 :
                    window_end_time_sec = window_start_time_sec + (1.0 / fps)
                else:
                    raise ValueError(f"Cannot define valid read window for {video_id}. Start: {window_start_time_sec}, End: {window_end_time_sec}, Duration: {duration}")
            window_end_time_sec = min(window_end_time_sec, duration)

            # debug statements
            # print(f"  Window Calc: tta={tta:.2f}, calculated_start_sec={window_start_time_sec:.2f}, calculated_end_sec={window_end_time_sec:.2f}")
            # if window_start_time_sec >= window_end_time_sec:
            #     print(f"  ERROR: Invalid window! start_sec >= end_sec for {video_id}. Raising ValueError.")
            #     raise ValueError("Cannot define a valid read window.")
            # print(f"  Final Window: start_pts={window_start_time_sec:.2f}s, end_pts={window_end_time_sec:.2f}s")

            vframes_tchw_uint8, _, info = tv_io.read_video(
                video_path, start_pts=window_start_time_sec, end_pts=window_end_time_sec,
                pts_unit='sec', output_format="TCHW"
            )

            num_read_frames = vframes_tchw_uint8.shape[0]
            final_frames_to_return = default_frames_tensor

            # debug statements
            # print(f"  tv_io.read_video returned {num_read_frames} frames. Shape: {vframes_tchw_uint8.shape}. Info: {info}")

            final_frames_processed_tensor = default_frames_tensor # Initialize with placeholder
            labels_list = [0.0] * self.sequence_length

            if num_read_frames > 0:
                selected_indices_in_vframes = np.linspace(0, num_read_frames - 1, self.sequence_length, dtype=int, endpoint=True)
                subsampled_frames_tchw_uint8 = vframes_tchw_uint8[selected_indices_in_vframes]

                orig_start_frame_idx = int(window_start_time_sec * fps)
                orig_end_frame_idx = min(int(window_end_time_sec * fps), total_frames_cv -1)
                if orig_start_frame_idx > orig_end_frame_idx: orig_start_frame_idx = orig_end_frame_idx
                original_sampled_indices_for_label = np.linspace(orig_start_frame_idx, orig_end_frame_idx, self.sequence_length, dtype=int, endpoint=True)

                # debug statements
                # print(f"  Original Sampled Indices (for label calc): {original_sampled_indices_for_label}")
                # print(f"  Calculated atol_val for isclose: {self.atol_val}")

                for i in range(self.sequence_length):
                    t_for_label = original_sampled_indices_for_label[i] / fps
                    labels_list[i] = compute_frame_label(t_for_label, alert_time, atol=self.atol_val)
                    #for debugging
                    # print(f"  Frame {i}: t_for_label={t_for_label:.3f}s, alert_time={alert_time}, label_val={labels_list[i]:.3f}, isclose={np.isclose(t_for_label, alert_time, self.atol_val)}")

                final_frames_to_return = subsampled_frames_tchw_uint8 # Return uint8 TCHW tensors
                # print(f"  Final labels_list for {video_id} (inside num_read_frames > 0): {[f'{l:.3f}' for l in labels_list]}")
            else:
                print(f"  num_read_frames was NOT > 0 for {video_id}. Labels will be default zeros.")
            


            labels_tensor = torch.tensor(labels_list, dtype=torch.float32)
            # Return CPU tensors. DataLoader will handle moving to GPU if pin_memory=True.
            return video_id, final_frames_to_return.cpu(), labels_tensor.cpu()

        except Exception as e:
            print(f"Error processing video {video_id} ({video_path}): {e}. Returning placeholders.") # Can be verbose
            return video_id, default_frames_tensor.cpu(), default_labels_tensor.cpu()

def collate_fn_videos(batch): # Kept for B > 1, but current main script uses B=1
    video_ids = [item[0] for item in batch]
    frames_list = [item[1] for item in batch]
    labels_list = [item[2] for item in batch]
    try:
        frames_batch = torch.stack(frames_list, dim=0)
        labels_batch = torch.stack(labels_list, dim=0)
    except RuntimeError as e:
        # This might happen if placeholder shapes don't match actual data shapes
        # or if an error in __getitem__ returns tensors of unexpected shapes.
        print(f"Error during collate_fn stacking: {e}. Returning uncollated lists.")
        return video_ids, frames_list, labels_list
    return video_ids, frames_batch, labels_batch