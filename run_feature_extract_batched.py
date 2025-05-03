import pandas as pd
import numpy as np
import os
from torchvision import transforms
from datasets.video_dataset import FrameCollector
from models.ResNet_feature_extract import extract_features_batched
from datetime import datetime
from tqdm import tqdm

if __name__ == "__main__":
    df = pd.read_csv("nexar-collision-prediction/train.csv")
    df["id"] = df["id"].apply(lambda x: str(x).zfill(5))

    TRAIN_VIDEO_DIR = "nexar-collision-prediction/train"
    FPS_TARGET = 3
    TIME_WINDOW = 10.0
    SEQUENCE_LENGTH = int(FPS_TARGET * TIME_WINDOW)
    BATCH_SIZE = 32
    SAVE_INTERVAL = 128

    timestamp_str = datetime.now().strftime("%d%m%H%M%S")
    feature_save_dir = f"ResNet_Features/batch_{timestamp_str}"
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

    for batch_idx in tqdm(range(num_batches)):
        start = batch_idx * SAVE_INTERVAL
        end = min((batch_idx + 1) * SAVE_INTERVAL, num_videos)
        df_batch = df.iloc[start:end]

        print(f"\n--- Processing batch {batch_idx + 1}/{num_batches}: videos {start} to {end - 1} ---")

        collector = FrameCollector(df_batch, TRAIN_VIDEO_DIR, fps_target=FPS_TARGET)
        frames_per_video, metadata, labels_per_video = collector.collect()
        print(f"Collected frames from {len(frames_per_video)} videos")

        all_frames = [frame for video_frames in frames_per_video for frame in video_frames]

        all_features = extract_features_batched(
            all_frames, transform=resnet_transform, batch_size=BATCH_SIZE
        )

        features_per_video = []
        i = 0
        for video_frames in frames_per_video:
            n = len(video_frames)
            features_per_video.append(all_features[i:i + n])
            i += n

        np.save(f"{feature_save_dir}/train_features_batch{batch_idx+1}_{timestamp_str}.npy", np.array(features_per_video, dtype=object))
        np.save(f"{feature_save_dir}/train_labels_batch{batch_idx+1}_{timestamp_str}.npy", np.array(labels_per_video, dtype=object))

        print(f"Saved batch {batch_idx+1} features and labels to {feature_save_dir}/")

    print("\nAll batches processed and saved.")
