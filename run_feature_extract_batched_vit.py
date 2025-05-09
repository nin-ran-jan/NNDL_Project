from datetime import datetime
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

from datasets.video_dataset import FrameCollector
from models.Vit_feature_extract import extract_features_batched_hf

if __name__ == "__main__":
    df = pd.read_csv("nexar-collision-prediction/train.csv")
    df["id"] = df["id"].apply(lambda x: str(x).zfill(5))

    TRAIN_VIDEO_DIR = "nexar-collision-prediction/train"
    FPS_TARGET = 3 
    TIME_WINDOW = 10.0 
    SEQUENCE_LENGTH = int(FPS_TARGET * TIME_WINDOW)
    
    VIT_BATCH_SIZE = 32
    SAVE_INTERVAL_VIDEOS = 128

    VIT_MODEL_NAME = "openai/clip-vit-large-patch14"

    timestamp_str = datetime.now().strftime("%y%m%d_%H%M%S")
    feature_save_dir = f"CLIP_ViT_Features_{VIT_MODEL_NAME.split('/')[-1]}/run_{timestamp_str}"
    os.makedirs(feature_save_dir, exist_ok=True)
    print(f"Saving features to: {feature_save_dir}")

    all_video_ids = df["id"].tolist()
    num_videos = len(all_video_ids)
    num_saving_batches = (num_videos + SAVE_INTERVAL_VIDEOS - 1) // SAVE_INTERVAL_VIDEOS

    # AI prompt: generate helpful print statements
    print(f"Total videos to process: {num_videos}")
    print(f"Saving features in {num_saving_batches} batches of up to {SAVE_INTERVAL_VIDEOS} videos each.")
    print(f"ViT model for feature extraction: {VIT_MODEL_NAME}")
    print(f"ViT processing batch size (frames): {VIT_BATCH_SIZE}")


    for batch_idx in tqdm(range(num_saving_batches), desc="Video Batches for Saving"):
        start_video_index = batch_idx * SAVE_INTERVAL_VIDEOS
        end_video_index = min((batch_idx + 1) * SAVE_INTERVAL_VIDEOS, num_videos)
        df_video_batch = df.iloc[start_video_index:end_video_index]

        print(f"\n--- Processing video saving_batch {batch_idx + 1}/{num_saving_batches}: videos {start_video_index} to {end_video_index - 1} ---")

        collector = FrameCollector(df_video_batch, TRAIN_VIDEO_DIR, 
                                   fps_target=FPS_TARGET, 
                                   sequence_length=SEQUENCE_LENGTH)
        
        frames_per_video, metadata, labels_per_video = collector.collect()
        
        # AI prompt: generate helful corrective print statements
        if not frames_per_video:
            print(f"No frames collected for video batch {batch_idx + 1}. Skipping.")
            continue
        print(f"Collected frames from {len(frames_per_video)} videos in this saving batch.")

        # flatten for ViT processing
        all_frames_np_bgr = [frame for video_frames in frames_per_video if video_frames for frame in video_frames]

        # AI prompt: generate helful corrective print statements
        if not all_frames_np_bgr:
            print(f"No frames to process after flattening for video batch {batch_idx + 1}. Skipping.")
            continue
        
        print(f"Total frames to extract features from in this saving_batch: {len(all_frames_np_bgr)}")

        # (BGR NumPy -> RGB PIL) included in extract_features_batched_hf
        all_features_flat_np = extract_features_batched_hf(
            all_numpy_frames=all_frames_np_bgr,
            model_name=VIT_MODEL_NAME,
            batch_size=VIT_BATCH_SIZE
        )

        if all_features_flat_np.size == 0:
            print(f"No features extracted for video batch {batch_idx + 1}. Skipping saving.")
            continue

        # same as resnet code with better variable names
        reconstructed_features_per_video = []
        current_feature_idx = 0
        for video_frames_list in frames_per_video:
            num_frames_in_video = len(video_frames_list)
            video_features = all_features_flat_np[current_feature_idx : current_feature_idx + num_frames_in_video]
            reconstructed_features_per_video.append(video_features)
            current_feature_idx += num_frames_in_video


        save_path_features = f"{feature_save_dir}/train_features_video_batch{batch_idx+1}.npy"
        save_path_labels = f"{feature_save_dir}/train_labels_video_batch{batch_idx+1}.npy"


        np.save(save_path_features, np.array(reconstructed_features_per_video, dtype=object))
        np.save(save_path_labels, np.array(labels_per_video, dtype=object))

        print(f"Saved video batch {batch_idx+1} features to {save_path_features}")
        print(f"Saved video batch {batch_idx+1} labels to {save_path_labels}")

    print("\nAll video batches processed and features saved.")