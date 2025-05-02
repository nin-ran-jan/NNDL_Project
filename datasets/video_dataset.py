import torch
import cv2
import os
import pandas as pd
from tqdm import tqdm

class FrameCollector:
    def __init__(self, df, video_dir, fps_target=5):
        self.df = df
        self.video_dir = video_dir
        self.fps_target = fps_target
        self.frames = []
        self.metadata = []  # (video_id, timestamp, frame_idx)

    def collect(self):
        for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
            video_id = row["id"]
            video_path = os.path.join(self.video_dir, f"{video_id}.mp4")
            if not os.path.exists(video_path):
                continue

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_step = max(1, int(fps / self.fps_target))
            frame_idx = 0
            while True:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                self.frames.append(frame)
                self.metadata.append((video_id, frame_idx / fps, row["time_of_event"], row["time_of_alert"]))
                frame_idx += frame_step
            cap.release()

        return self.frames, self.metadata

class FrameBatchDataset(torch.utils.data.Dataset):
    def __init__(self, frames, transform):
        self.frames = frames
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        return self.transform(frame)
