import pandas as pd
import numpy as np
from torchvision import transforms
from datasets.video_dataset import FrameCollector
from models.ResNet_feature_extract import extract_features_batched

if __name__ == "__main__":
    df = pd.read_csv("nexar-collision-prediction/train.csv")
    df["id"] = df["id"].apply(lambda x: str(x).zfill(5))
    df_subset = df.head(8)

    TRAIN_VIDEO_DIR = "nexar-collision-prediction/train"
    FPS_TARGET = 2
    TIME_WINDOW = 10.0
    SEQUENCE_LENGTH = int(FPS_TARGET * TIME_WINDOW)

    collector = FrameCollector(df_subset, TRAIN_VIDEO_DIR, fps_target=FPS_TARGET)
    frames_per_video, metadata, labels_per_video = collector.collect()
    print(f"Collected frames from {len(frames_per_video)} videos")

    all_frames = [frame for video_frames in frames_per_video for frame in video_frames]

    resnet_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    all_features = extract_features_batched(
        all_frames, transform=resnet_transform, batch_size=32
    )

    features_per_video = []
    i = 0
    for video_frames in frames_per_video:
        n = len(video_frames)
        features_per_video.append(all_features[i:i + n])
        i += n

    np.save("ResNet_Features/train_features.npy", np.array(features_per_video, dtype=object))
    np.save("ResNet_Features/train_labels.npy", np.array(labels_per_video, dtype=object))
    print("Saved per-video features and labels to ResNet_Features/")
