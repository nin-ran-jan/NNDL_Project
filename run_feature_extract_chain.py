from datetime import datetime
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import json

from datasets.video_dataset import FrameCollector
from models.Vit_feature_extract import extract_features_batched_hf

if __name__ == "__main__":
    # --- Configuration ---
    BASE_DATA_DIR = "nexar-collision-prediction"
    TRAIN_CSV_PATH = os.path.join(BASE_DATA_DIR, "train.csv")
    TRAIN_VIDEO_DIR = os.path.join(BASE_DATA_DIR, "train")

    FPS_TARGET = 3
    TIME_WINDOW = 10.0  # seconds
    SEQUENCE_LENGTH = int(FPS_TARGET * TIME_WINDOW)

    VIT_MODEL_NAME = "openai/clip-vit-large-patch14"
    VIT_PROCESSING_BATCH_SIZE = 32 # ViT model's internal batch size for frames

    # How many videos' features/labels to group for saving to one .npy file
    OUTPUT_SAVE_INTERVAL_VIDEOS = 64 # e.g., 64 or 128

    # How many videos to process in a "mini-batch" for frame collection
    # These videos' frames will be in RAM after collection.
    # Adjust this based on RAM (e.g., 16, 32, 64).
    # For 30 frames/video @ ~2.64MB/frame = ~79.2MB/video
    # 32 videos * 79.2MB/video = ~2.5 GB for frames. This should be fine on 32GB RAM.
    VIDEOS_IN_PROCESSING_MINIBATCH = 32

    # Number of worker processes for FrameCollector's internal pool
    # when collecting frames for the VIDEOS_IN_PROCESSING_MINIBATCH
    FRAME_COLLECTOR_POOL_WORKERS = max(1, os.cpu_count() // 2 if os.cpu_count() else 4) # Or a fixed number like 4, 8


    # --- Output Directory Setup ---
    timestamp_str = datetime.now().strftime("%y%m%d_%H%M%S")
    base_output_folder = "processed_data"
    model_name_slug = VIT_MODEL_NAME.split('/')[-1]
    feature_save_dir = os.path.join(base_output_folder, f"CLIP_ViT_Features_{model_name_slug}", f"run_{timestamp_str}")
    os.makedirs(feature_save_dir, exist_ok=True)
    print(f"Output features and labels will be saved to: {feature_save_dir}")

    # --- Load Video Metadata ---
    df = pd.read_csv(TRAIN_CSV_PATH)
    df["id"] = df["id"].apply(lambda x: str(x).zfill(5))
    # df = df.head(VIDEOS_IN_PROCESSING_MINIBATCH * 2 + 5) # For testing

    num_total_videos = len(df)
    num_output_saving_batches = (num_total_videos + OUTPUT_SAVE_INTERVAL_VIDEOS - 1) // OUTPUT_SAVE_INTERVAL_VIDEOS

    # --- Save Run Metadata ---
    run_metadata = {
        "run_timestamp": timestamp_str,
        "vit_model_name": VIT_MODEL_NAME,
        "fps_target": FPS_TARGET,
        "time_window_seconds": TIME_WINDOW,
        "sequence_length_frames": SEQUENCE_LENGTH,
        "vit_processing_batch_size": VIT_PROCESSING_BATCH_SIZE,
        "output_save_interval_videos": OUTPUT_SAVE_INTERVAL_VIDEOS,
        "videos_in_processing_minibatch": VIDEOS_IN_PROCESSING_MINIBATCH,
        "frame_collector_pool_workers": FRAME_COLLECTOR_POOL_WORKERS,
        "source_csv_file": TRAIN_CSV_PATH,
        "video_directory": TRAIN_VIDEO_DIR,
        "num_total_videos_in_run": num_total_videos,
        "num_output_saving_batches_planned": num_output_saving_batches
    }
    with open(os.path.join(feature_save_dir, "run_metadata.json"), "w") as f:
        json.dump(run_metadata, f, indent=4)
    print(f"Saved run metadata. Processing {num_total_videos} videos in {num_output_saving_batches} saving batches.")

    # --- Main Processing Loop ---
    # Outer loop: Iterates for saving accumulated features
    for output_batch_idx in tqdm(range(num_output_saving_batches), desc="Output Saving Batches"):
        start_video_df_idx_for_saving = output_batch_idx * OUTPUT_SAVE_INTERVAL_VIDEOS
        end_video_df_idx_for_saving = min((output_batch_idx + 1) * OUTPUT_SAVE_INTERVAL_VIDEOS, num_total_videos)
        
        df_for_current_output_batch = df.iloc[start_video_df_idx_for_saving:end_video_df_idx_for_saving]

        print(f"\n--- Preparing Output Saving Batch {output_batch_idx + 1}/{num_output_saving_batches} (Videos DF Index: {start_video_df_idx_for_saving} to {end_video_df_idx_for_saving - 1}) ---")

        accumulated_features_in_output_batch = []
        accumulated_labels_in_output_batch = []
        accumulated_video_ids_in_output_batch = []

        # Middle loop: Iterates through df_for_current_output_batch in chunks of VIDEOS_IN_PROCESSING_MINIBATCH
        num_processing_minibatches = (len(df_for_current_output_batch) + VIDEOS_IN_PROCESSING_MINIBATCH - 1) // VIDEOS_IN_PROCESSING_MINIBATCH

        for proc_minibatch_idx in tqdm(range(num_processing_minibatches), desc="Processing Mini-Batches", leave=False):
            start_idx_proc_minibatch = proc_minibatch_idx * VIDEOS_IN_PROCESSING_MINIBATCH
            end_idx_proc_minibatch = min((proc_minibatch_idx + 1) * VIDEOS_IN_PROCESSING_MINIBATCH, len(df_for_current_output_batch))
            
            df_processing_minibatch = df_for_current_output_batch.iloc[start_idx_proc_minibatch:end_idx_proc_minibatch]

            if df_processing_minibatch.empty:
                continue

            print(f"  Processing Mini-Batch {proc_minibatch_idx + 1}/{num_processing_minibatches} (Size: {len(df_processing_minibatch)} videos)")

            # 1. Collect Frames and Labels for the current MINI-BATCH of videos.
            #    FrameCollector will use its internal pool with FRAME_COLLECTOR_POOL_WORKERS.
            collector_minibatch = FrameCollector(df_processing_minibatch, TRAIN_VIDEO_DIR,
                                                 fps_target=FPS_TARGET,
                                                 sequence_length=SEQUENCE_LENGTH)
            
            # frames_per_video_in_minibatch: list where each element is a list of frames for one video from the minibatch.
            # labels_per_video_in_minibatch: list where each element is a list of labels for one video.
            # These are ordered according to collector_minibatch.video_order.
            frames_per_video_in_minibatch, _, labels_per_video_in_minibatch = collector_minibatch.collect_parallel(
                num_workers=FRAME_COLLECTOR_POOL_WORKERS
            )
            
            # Get the actual order of video IDs processed by the collector
            processed_video_ids_in_minibatch = collector_minibatch.video_order

            # 2. Iterate through the results of this mini-batch (frames/labels now in main process memory)
            #    and extract features for each video's frames sequentially.
            for i, video_id in enumerate(processed_video_ids_in_minibatch):
                current_video_frames = frames_per_video_in_minibatch[i] if i < len(frames_per_video_in_minibatch) else None
                current_video_labels = labels_per_video_in_minibatch[i] if i < len(labels_per_video_in_minibatch) else None

                if not current_video_frames:
                    print(f"    Warning: No frames collected for video {video_id} in mini-batch. Appending empty.")
                    accumulated_features_in_output_batch.append(np.array([]))
                    accumulated_labels_in_output_batch.append([])
                    accumulated_video_ids_in_output_batch.append(video_id)
                    continue
                
                # print(f"    Extracting features for video {video_id} ({len(current_video_frames)} frames)")

                video_features_np = extract_features_batched_hf(
                    all_numpy_frames=current_video_frames, # Pass frames of ONE video
                    model_name=VIT_MODEL_NAME,
                    batch_size=VIT_PROCESSING_BATCH_SIZE
                )

                accumulated_features_in_output_batch.append(video_features_np)
                accumulated_labels_in_output_batch.append(np.array(current_video_labels if current_video_labels is not None else []))
                accumulated_video_ids_in_output_batch.append(video_id)

            # Clean up data for the processed mini-batch to free RAM
            del collector_minibatch, frames_per_video_in_minibatch, labels_per_video_in_minibatch, processed_video_ids_in_minibatch
            # Consider gc.collect() here if memory is extremely tight and not releasing fast enough, but usually not needed.

        # After processing all mini-batches for the current output_batch_idx, save the accumulated data
        if not accumulated_video_ids_in_output_batch:
            print(f"No videos processed for output saving batch {output_batch_idx + 1}. Skipping save.")
            continue

        batch_file_suffix = f"saving_batch_{output_batch_idx+1}"
        save_path_features = os.path.join(feature_save_dir, f"train_features_{batch_file_suffix}.npy")
        save_path_labels = os.path.join(feature_save_dir, f"train_labels_{batch_file_suffix}.npy")
        save_path_video_ids = os.path.join(feature_save_dir, f"train_video_ids_{batch_file_suffix}.json")

        try:
            np.save(save_path_features, np.array(accumulated_features_in_output_batch, dtype=object))
            np.save(save_path_labels, np.array(accumulated_labels_in_output_batch, dtype=object))
            with open(save_path_video_ids, "w") as f:
                json.dump(accumulated_video_ids_in_output_batch, f, indent=4)
            
            print(f"Saved Output Saving Batch {output_batch_idx+1}:")
            print(f"  Features: {save_path_features} ({len(accumulated_features_in_output_batch)} videos)")
            # ... (similar print for labels and video IDs)
        except Exception as e:
            print(f"Error saving data for output saving batch {output_batch_idx + 1}: {e}")

        del accumulated_features_in_output_batch, accumulated_labels_in_output_batch, accumulated_video_ids_in_output_batch

    print("\nAll video processing and feature saving completed.")