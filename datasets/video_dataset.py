from multiprocessing import Pool
import torch
import cv2
import os
import pandas as pd
import numpy as np
from tqdm import tqdm


def compute_frame_label(t, alert_time, sigma_before=2.0, sigma_after=0.5, atol=0.18):
    """Soft Gaussian label centered at alert_time."""
    if np.isclose(t, alert_time, atol=atol):
        return 1.0
    if t < alert_time:
        return np.exp(-((alert_time - t)**2) / (2 * sigma_before**2))
    else:
        return np.exp(-((t - alert_time)**2) / (2 * sigma_after**2))

def process_single_video(args):
    row, video_dir_path, target_fps, sequence_len, atol_val, default_frame_h, default_frame_w = args

    video_id = row["id"]
    video_path = os.path.join(video_dir_path, f"{video_id}.mp4")

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 30.0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0

    event_time = row["time_of_event"]
    alert_time = row["time_of_alert"]
    is_positive = not pd.isna(alert_time)

    tta = np.random.uniform(0.5, 1.5)
    window_start_time = 0.0 
    window_end_time = 0.0 

    if not is_positive:
        window_start_time = 0.0
        window_end_time = min(10.0, duration)
    elif alert_time < 10.0: 
        window_start_time = 0.0
        window_end_time = min(10.0, duration)
    else:
        window_end_time = min(alert_time + tta, duration)
        window_start_time = max(0.0, window_end_time - 10.0)

    start_frame_idx = int(window_start_time * fps)
    end_frame_idx = int(window_end_time * fps)
    
    end_frame_idx = min(end_frame_idx, total_frames - 1)
    
    current_video_frames = []
    current_video_labels = []
    current_video_metadata_items = []

    sampled_indices = np.linspace(start_frame_idx, end_frame_idx, sequence_len, dtype=int, endpoint=True)

    for frame_idx in sampled_indices:
        if frame_idx < 0 or frame_idx >= total_frames:
            frame = np.zeros((default_frame_h, default_frame_w, 3), dtype=np.uint8)
            label = 0.0
            t = frame_idx / fps if fps > 0 else 0.0
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            t = frame_idx / fps if fps > 0 else 0.0
            label = compute_frame_label(t, alert_time, atol=atol_val) if is_positive else 0.0
        
        current_video_frames.append(frame)
        current_video_labels.append(label)
        current_video_metadata_items.append((video_id, t, event_time, alert_time))

    cap.release()
    return video_id, current_video_frames, current_video_labels, current_video_metadata_items


class FrameCollector:
    def __init__(self, df, video_dir, fps_target=5, sequence_length=20):
        self.df = df
        self.video_dir = video_dir
        self.fps_target = fps_target
        self.sequence_length = sequence_length
        self.frames_per_video = []
        self.labels_per_video = []
        self.metadata = []

    def collect(self):
        atol = 1.0 / self.fps_target

        for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
            video_id = row["id"]
            video_path = os.path.join(self.video_dir, f"{video_id}.mp4")
            if not os.path.exists(video_path):
                continue

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps

            event_time = row["time_of_event"]
            alert_time = row["time_of_alert"]
            is_positive = not pd.isna(alert_time)

            tta = np.random.uniform(0.5, 1.5)

            if not is_positive:
                window_start = 0.0
                window_end = min(10.0, duration)
            elif alert_time < 10.0:
                window_start = 0.0
                window_end = min(10.0, duration)
            else:
                window_end = min(alert_time + tta, duration)
                window_start = max(0.0, window_end - 10.0)

            start_frame = int(window_start * fps)
            end_frame = int(window_end * fps)

            sampled_indices = np.linspace(
                start_frame, end_frame - 1, self.sequence_length, dtype=int
            )

            video_frames = []
            video_labels = []

            for idx in sampled_indices:
                if idx < 0 or idx >= total_frames:
                    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                    label = 0.0
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if not ret:
                        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                    t = idx / fps
                    label = compute_frame_label(t, alert_time, atol=atol) if is_positive else 0.0

                video_frames.append(frame)
                video_labels.append(label)
                self.metadata.append((video_id, idx / fps, event_time, alert_time))

            self.frames_per_video.append(video_frames)
            self.labels_per_video.append(video_labels)

            cap.release()

        return self.frames_per_video, self.metadata, self.labels_per_video

    def collect_parallel(self, num_workers=None):
        """
        Collects frames, labels, and metadata from videos, optionally using multiprocessing.
        Args:
            num_workers (int, optional): Number of worker processes. Defaults to cpu_count().
                                         Set to 0 or 1 for sequential processing (debugging).
        """
        if num_workers is None:
            num_workers = 3
        # Ensure num_workers is at least 1 for the Pool, or handle sequential case
        if num_workers <=0 : num_workers = 1


        atol = 1.0 / self.fps_target
        default_frame_height = 720
        default_frame_width = 1280

        tasks_args = [
            (row_dict, self.video_dir, self.fps_target, self.sequence_length, atol, default_frame_height, default_frame_width)
            for row_dict in self.df.to_dict('records')
        ]
        
        self.video_order = [row_dict[0]['id'] for row_dict in tasks_args] # Store original video order

        self.frames_per_video_map = {}
        self.labels_per_video_map = {}
        self.metadata_flat_list = []


        if num_workers > 1 and len(tasks_args) > 1 : # Use multiprocessing
            print(f"Starting frame collection with {num_workers} workers...")
            with Pool(processes=num_workers) as pool:
                results_iterator = pool.imap_unordered(process_single_video, tasks_args)
                
                for result in tqdm(results_iterator, total=len(tasks_args), desc="Collecting Video Data"):
                    if result:
                        video_id_res, frames_res, labels_res, metadata_items_res = result
                        self.frames_per_video_map[video_id_res] = frames_res
                        self.labels_per_video_map[video_id_res] = labels_res
                        self.metadata_flat_list.extend(metadata_items_res)
        else: 
            print("Starting frame collection sequentially...")
            for task_arg_set in tqdm(tasks_args, desc="Collecting Video Data (Sequential)"):
                result = process_single_video(task_arg_set)
                if result:
                    video_id_res, frames_res, labels_res, metadata_items_res = result
                    self.frames_per_video_map[video_id_res] = frames_res
                    self.labels_per_video_map[video_id_res] = labels_res
                    self.metadata_flat_list.extend(metadata_items_res)

        final_frames_per_video = [self.frames_per_video_map.get(vid_id, []) for vid_id in self.video_order]
        final_labels_per_video = [self.labels_per_video_map.get(vid_id, []) for vid_id in self.video_order]
        
        return final_frames_per_video, self.metadata_flat_list, final_labels_per_video

class FrameBatchDataset(torch.utils.data.Dataset):
    def __init__(self, frames, transform):
        self.frames = frames
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        return self.transform(frame)
