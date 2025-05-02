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


class FrameCollector:
    def __init__(self, df, video_dir, fps_target=5):
        self.df = df
        self.video_dir = video_dir
        self.fps_target = fps_target
        self.frames_per_video = []
        self.labels_per_video = []
        self.metadata = []  # (video_id, timestamp, event_time, alert_time)

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
            frame_interval = int(fps / self.fps_target)

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
            sampled_indices = list(range(start_frame, end_frame, frame_interval))

            if len(sampled_indices) == 0:
                cap.release()
                continue

            video_frames = []
            video_labels = []

            for idx in sampled_indices:
                if idx < 0 or idx >= total_frames:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                t = idx / fps
                label = compute_frame_label(t, alert_time, atol=atol) if is_positive else 0.0
                video_frames.append(frame)
                video_labels.append(label)
                self.metadata.append((video_id, t, event_time, alert_time))

            if video_frames:
                self.frames_per_video.append(video_frames)
                self.labels_per_video.append(video_labels)

            cap.release()

        return self.frames_per_video, self.metadata, self.labels_per_video


class FrameBatchDataset(torch.utils.data.Dataset):
    def __init__(self, frames, transform):
        self.frames = frames
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        return self.transform(frame)
