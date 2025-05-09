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
    TIME_WINDOW = 10.0
    SEQUENCE_LENGTH = int(FPS_TARGET * TIME_WINDOW)

    VIT_MODEL_NAME = "openai/clip-vit-large-patch14"
    # ViT model's internal batch size for processing frames passed to it
    VIT_PROCESSING_BATCH_SIZE = 32 # Keep this, it's for the DataLoader in extract_features_batched_hf

    # How many videos' features/labels to group for saving to one .npy file
    OUTPUT_SAVE_INTERVAL_VIDEOS = 64

    # How many videos to process in a "mini-batch". Frames for these videos
    # will be collected, then ALL their frames will be sent to ViT together.
    # Memory: 32 videos * ~79.2MB/video = ~2.5 GB for frames.
    VIDEOS_IN_PROCESSING_MINIBATCH = 32 # Adjust based on RAM (16, 32, 64 are good starts)

    # Number of worker processes for FrameCollector's internal pool
    FRAME_COLLECTOR_POOL_WORKERS = max(1, os.cpu_count() // 2 if os.cpu_count() else 4)

    # --- Output Directory Setup (Same as before) ---
    timestamp_str = datetime.now().strftime("%y%m%d_%H%M%S")
    base_output_folder = "processed_data"
    model_name_slug = VIT_MODEL_NAME.split('/')[-1]
    feature_save_dir = os.path.join(base_output_folder, f"CLIP_ViT_Features_{model_name_slug}", f"run_{timestamp_str}")
    os.makedirs(feature_save_dir, exist_ok=True)
    print(f"Output features and labels will be saved to: {feature_save_dir}")

    # --- Load Video Metadata (Same as before) ---
    df = pd.read_csv(TRAIN_CSV_PATH)
    df["id"] = df["id"].apply(lambda x: str(x).zfill(5))
    # df = df.head(VIDEOS_IN_PROCESSING_MINIBATCH + 5) # For quick testing

    num_total_videos = len(df)
    num_output_saving_batches = (num_total_videos + OUTPUT_SAVE_INTERVAL_VIDEOS - 1) // OUTPUT_SAVE_INTERVAL_VIDEOS

    # --- Save Run Metadata (Same as before, include new/changed params) ---
    run_metadata = {
        "run_timestamp": timestamp_str,
        "vit_model_name": VIT_MODEL_NAME,
        "fps_target": FPS_TARGET,
        "time_window_seconds": TIME_WINDOW,
        "sequence_length_frames": SEQUENCE_LENGTH,
        "vit_internal_processing_batch_size": VIT_PROCESSING_BATCH_SIZE, # Clarified name
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
    print(f"Saved run metadata. Processing {num_total_videos} videos.")

    # --- Main Processing Loop ---
    for output_batch_idx in tqdm(range(num_output_saving_batches), desc="Output Saving Batches"):
        start_video_df_idx_for_saving = output_batch_idx * OUTPUT_SAVE_INTERVAL_VIDEOS
        end_video_df_idx_for_saving = min((output_batch_idx + 1) * OUTPUT_SAVE_INTERVAL_VIDEOS, num_total_videos)
        df_for_current_output_batch = df.iloc[start_video_df_idx_for_saving:end_video_df_idx_for_saving]

        print(f"\n--- Preparing Output Saving Batch {output_batch_idx + 1}/{num_output_saving_batches} ---")

        accumulated_features_in_output_batch = []
        accumulated_labels_in_output_batch = []
        accumulated_video_ids_in_output_batch = [] # Will store IDs in the order of processed features/labels

        num_processing_minibatches = (len(df_for_current_output_batch) + VIDEOS_IN_PROCESSING_MINIBATCH - 1) // VIDEOS_IN_PROCESSING_MINIBATCH

        for proc_minibatch_idx in tqdm(range(num_processing_minibatches), desc="Processing Mini-Batches", leave=False):
            start_idx_proc_minibatch = proc_minibatch_idx * VIDEOS_IN_PROCESSING_MINIBATCH
            end_idx_proc_minibatch = min((proc_minibatch_idx + 1) * VIDEOS_IN_PROCESSING_MINIBATCH, len(df_for_current_output_batch))
            df_processing_minibatch = df_for_current_output_batch.iloc[start_idx_proc_minibatch:end_idx_proc_minibatch]

            if df_processing_minibatch.empty:
                continue

            print(f"  Processing Mini-Batch {proc_minibatch_idx + 1}/{num_processing_minibatches} (Size: {len(df_processing_minibatch)} videos)")

            collector_minibatch = FrameCollector(df_processing_minibatch, TRAIN_VIDEO_DIR,
                                                 fps_target=FPS_TARGET, sequence_length=SEQUENCE_LENGTH)
            frames_per_video_in_minibatch, _, labels_per_video_in_minibatch = collector_minibatch.collect_parallel(
                num_workers=FRAME_COLLECTOR_POOL_WORKERS
            )
            processed_video_ids_in_order = collector_minibatch.video_order # Critical for matching

            # --- Prepare data for BATCHED feature extraction ---
            all_frames_for_gpu_batch = []
            num_frames_per_video_for_reconstruction = []
            # These lists will store data for videos that actually yield frames
            ordered_ids_for_this_gpu_batch = []
            ordered_labels_for_this_gpu_batch = []

            for i, video_id in enumerate(processed_video_ids_in_order):
                current_video_frames = frames_per_video_in_minibatch[i] if i < len(frames_per_video_in_minibatch) else None
                current_video_labels = labels_per_video_in_minibatch[i] if i < len(labels_per_video_in_minibatch) else None

                if current_video_frames and len(current_video_frames) > 0:
                    all_frames_for_gpu_batch.extend(current_video_frames)
                    num_frames_per_video_for_reconstruction.append(len(current_video_frames))
                    ordered_ids_for_this_gpu_batch.append(video_id)
                    ordered_labels_for_this_gpu_batch.append(np.array(current_video_labels if current_video_labels is not None else []))
                else:
                    # Video had no frames, still need to account for it in the final output batch lists
                    print(f"    Warning: No frames for video {video_id}. It will have empty features/labels.")
                    # We will add placeholders for this video ID *after* processing the valid ones
                    # Or, ensure it's added to the main accumulated lists with empty data later.
                    # For simplicity in reconstruction, we can note it now.
                    num_frames_per_video_for_reconstruction.append(0) # Mark as 0 frames for reconstruction
                    ordered_ids_for_this_gpu_batch.append(video_id) # Keep ID in order
                    ordered_labels_for_this_gpu_batch.append(np.array([])) # Empty labels

            if not all_frames_for_gpu_batch:
                print(f"    No frames to extract features from in this mini-batch. Adding placeholders for all videos.")
                # Add placeholders for all videos in this processing mini-batch to the main accumulation lists
                for video_id in processed_video_ids_in_order: # Use the original list of IDs for this minibatch
                    accumulated_features_in_output_batch.append(np.array([]))
                    # Find its original labels (if any) or use empty. This needs careful alignment.
                    # For simplicity, if no frames, assume empty labels for this context too.
                    original_idx_in_minibatch = processed_video_ids_in_order.index(video_id)
                    label_to_add = labels_per_video_in_minibatch[original_idx_in_minibatch] if original_idx_in_minibatch < len(labels_per_video_in_minibatch) and labels_per_video_in_minibatch[original_idx_in_minibatch] is not None else []
                    accumulated_labels_in_output_batch.append(np.array(label_to_add))
                    accumulated_video_ids_in_output_batch.append(video_id)
                del collector_minibatch, frames_per_video_in_minibatch, labels_per_video_in_minibatch
                continue # Move to the next processing mini-batch

            print(f"    Extracting features for {len(all_frames_for_gpu_batch)} total frames from {len(ordered_ids_for_this_gpu_batch)} videos in mini-batch.")
            all_features_flat_np = extract_features_batched_hf(
                all_numpy_frames=all_frames_for_gpu_batch,
                model_name=VIT_MODEL_NAME,
                batch_size=VIT_PROCESSING_BATCH_SIZE
            )

            # --- Reconstruct features per video and Accumulate ---
            current_feature_idx = 0
            for i, num_frames in enumerate(num_frames_per_video_for_reconstruction):
                video_id_for_accumulation = ordered_ids_for_this_gpu_batch[i]
                video_labels_for_accumulation = ordered_labels_for_this_gpu_batch[i]

                if num_frames > 0 and all_features_flat_np.size > 0 : # Ensure features were actually extracted
                    video_features = all_features_flat_np[current_feature_idx : current_feature_idx + num_frames]
                    current_feature_idx += num_frames
                else:
                    video_features = np.array([]) # Placeholder for videos with no frames

                accumulated_features_in_output_batch.append(video_features)
                accumulated_labels_in_output_batch.append(video_labels_for_accumulation)
                accumulated_video_ids_in_output_batch.append(video_id_for_accumulation)
            
            del collector_minibatch, frames_per_video_in_minibatch, labels_per_video_in_minibatch
            del all_frames_for_gpu_batch, all_features_flat_np, num_frames_per_video_for_reconstruction
            del ordered_ids_for_this_gpu_batch, ordered_labels_for_this_gpu_batch
            # gc.collect() # Optional: if memory isn't being freed quickly enough

        # --- After processing all mini-batches for the current output_batch_idx, save accumulated data ---
        # (Saving logic is the same as before, using accumulated_features_in_output_batch, etc.)
        if not accumulated_video_ids_in_output_batch:
            print(f"No videos processed for output saving batch {output_batch_idx + 1}. Skipping save.")
            continue

        batch_file_suffix = f"saving_batch_{output_batch_idx+1}"
        # ... (paths for features, labels, video_ids)
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
            print(f"  Labels:   {save_path_labels} ({len(accumulated_labels_in_output_batch)} videos)")
            print(f"  VideoIDs: {save_path_video_ids}")
        except Exception as e:
            print(f"Error saving data for output saving batch {output_batch_idx + 1}: {e}")
        
        del accumulated_features_in_output_batch, accumulated_labels_in_output_batch, accumulated_video_ids_in_output_batch

    print("\nAll video processing and feature saving completed.")