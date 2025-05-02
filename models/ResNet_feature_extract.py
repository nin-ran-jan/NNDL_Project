import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

FPS_TARGET = 5             
SEQUENCE_LENGTH = 50        
SIGMA_BEFORE = 2.0         
SIGMA_AFTER = 0.5          
USE_ALERT_TIME = True      

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

resnet18 = models.resnet18(pretrained=True)
resnet18 = nn.Sequential(*list(resnet18.children())[:-1])
resnet18 = resnet18.to(device).eval()

resnet_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def compute_frame_label(t, alert_time, sigma_before=2.0, sigma_after=0.5):
    """Soft Gaussian label centered at alert_time."""
    if np.isclose(t, alert_time, atol=0.18):
        return 1.0
    if t < alert_time:
        return np.exp(-((alert_time - t)**2) / (2 * sigma_before**2))
    else:
        return np.exp(-((t - alert_time)**2) / (2 * sigma_after**2))

def extract_features_and_labels(video_dir, df):
    video_sequences = []
    video_labels = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        video_id = row["id"]
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            continue

        event_time = row["time_of_event"]
        alert_time = row["time_of_alert"]
        use_time = alert_time if USE_ALERT_TIME and not pd.isna(alert_time) else event_time
        if pd.isna(use_time):
            use_time = -1

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = max(1, int(fps / FPS_TARGET))

        features = []
        timestamps = []
        frame_idx = 0

        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            try:
                tensor = resnet_transform(frame).unsqueeze(0).to(device)
            except:
                break

            with torch.no_grad():
                feature = resnet18(tensor).squeeze().cpu().numpy() 
            features.append(feature)
            timestamps.append(frame_idx / fps)
            frame_idx += frame_step

        cap.release()

        if len(features) < SEQUENCE_LENGTH:
            continue

        features = np.array(features)
        timestamps = np.array(timestamps)

        if use_time != -1:
            event_idx = np.argmin(np.abs(timestamps - use_time))
            start_idx = max(0, event_idx - SEQUENCE_LENGTH // 2)
            end_idx = start_idx + SEQUENCE_LENGTH
            if end_idx > len(features):
                end_idx = len(features)
                start_idx = end_idx - SEQUENCE_LENGTH
            sequence = features[start_idx:end_idx]
            times = timestamps[start_idx:end_idx]
            labels = [compute_frame_label(t, use_time, SIGMA_BEFORE, SIGMA_AFTER) for t in times]
        else:
            mid_idx = len(features) // 2
            start_idx = max(0, mid_idx - SEQUENCE_LENGTH // 2)
            end_idx = start_idx + SEQUENCE_LENGTH
            if end_idx > len(features):
                end_idx = len(features)
                start_idx = end_idx - SEQUENCE_LENGTH
            sequence = features[start_idx:end_idx]
            labels = [0.0] * SEQUENCE_LENGTH

        video_sequences.append(sequence)
        video_labels.append(labels)

    return np.array(video_sequences), np.array(video_labels)

