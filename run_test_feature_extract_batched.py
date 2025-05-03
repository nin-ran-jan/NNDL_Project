import pandas as pd
import numpy as np
import os
from torchvision import transforms
from models.ResNet_feature_extract import extract_features_batched
from datetime import datetime
import cv2
from tqdm import tqdm

def get_frames_for_batch(df_batch, video_dir, sequence_length=20, resolution=(720, 1280)):
    all_frames = []
    video_indices = []

    for _, row in tqdm(df_batch.iterrows(), total=len(df_batch)):
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
    FPS_TARGET = 3
    TIME_WINDOW = 10.0
    SEQUENCE_LENGTH = int(FPS_TARGET * TIME_WINDOW)
    SAVE_INTERVAL = 128
    BATCH_SIZE = 32

    df = pd.read_csv(CSV_PATH)
    df["id"] = df["id"].apply(lambda x: str(x).zfill(5))

    timestamp_str = datetime.now().strftime("%d%m%H%M%S")
    feature_save_dir = os.path.join(OUTPUT_DIR, f"test_batch_{timestamp_str}")
    os.makedirs(feature_save_dir, exist_ok=True)

    resnet_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    all_video_ids = df["id"].tolist()
    num_videos = len(all_video_ids)
    num_batches = (num_videos + SAVE_INTERVAL - 1) // SAVE_INTERVAL

    for batch_idx in range(num_batches):
        start = batch_idx * SAVE_INTERVAL
        end = min((batch_idx + 1) * SAVE_INTERVAL, num_videos)
        df_batch = df.iloc[start:end]
        print(f"\n--- Processing batch {batch_idx + 1}/{num_batches}: videos {start} to {end - 1} ---")

        all_frames, video_indices = get_frames_for_batch(df_batch, TEST_VIDEO_DIR, SEQUENCE_LENGTH)

        all_features = extract_features_batched(
            all_frames, transform=resnet_transform, batch_size=BATCH_SIZE
        )

        features_per_video = []
        idx = 0
        for count in video_indices:
            features_per_video.append(all_features[idx:idx + count])
            idx += count

        np.save(f"{feature_save_dir}/test_features_batch{batch_idx + 1}_{timestamp_str}.npy", np.array(features_per_video, dtype=object))
        np.save(f"{feature_save_dir}/test_ids_batch{batch_idx + 1}_{timestamp_str}.npy", df_batch["id"].values)
        print(f"Saved batch {batch_idx + 1} features and IDs.")

    print(f"\nAll test batches processed and saved in: {feature_save_dir}")
