import pandas as pd
import numpy as np
import os
from torchvision import transforms
from models.ResNet_feature_extract import extract_features_batched
from datetime import datetime
import cv2
from tqdm import tqdm

def get_frames_from_test_videos(df, video_dir, sequence_length=20, resolution=(720, 1280)):
    all_frames = []
    video_indices = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        video_id = row["id"]
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, sequence_length, dtype=int)

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((*resolution, 3), dtype=np.uint8)
            frames.append(frame)
        cap.release()

        all_frames.extend(frames)
        video_indices.append(len(frames))

    return all_frames, video_indices

if __name__ == "__main__":
    TEST_VIDEO_DIR = "nexar-collision-prediction/test"
    CSV_PATH = "nexar-collision-prediction/test.csv"
    OUTPUT_DIR = "ResNet_Features"
    FPS_TARGET = 2
    TIME_WINDOW = 10.0
    SEQUENCE_LENGTH = int(FPS_TARGET * TIME_WINDOW)

    df = pd.read_csv(CSV_PATH)
    df["id"] = df["id"].apply(lambda x: str(x).zfill(5))
    df = df.head(100)

    resnet_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    all_frames, video_indices = get_frames_from_test_videos(df, TEST_VIDEO_DIR, SEQUENCE_LENGTH)
    all_features = extract_features_batched(all_frames, transform=resnet_transform, batch_size=32)

    features_per_video = []
    start = 0
    for n in video_indices:
        features_per_video.append(all_features[start:start + n])
        start += n

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp_str = datetime.now().strftime("%d%m%H%M%S")

    np.save(f"{OUTPUT_DIR}/test_features_{timestamp_str}.npy", np.array(features_per_video, dtype=object))
    np.save(f"{OUTPUT_DIR}/test_ids_{timestamp_str}.npy", df["id"].values)

    print(f"Saved test features to {OUTPUT_DIR}/test_features_{timestamp_str}.npy")
    print(f"Saved test IDs to {OUTPUT_DIR}/test_ids_{timestamp_str}.npy")
