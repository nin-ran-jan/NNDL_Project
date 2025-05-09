import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image 
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from models.Vit_feature_extract import extract_features_batched_hf

def process_single_test_video_for_batch_worker(args):
    row_dict, video_dir_path, sequence_len, frame_resolution_tuple = args
    video_id_str = row_dict["id"]
    video_path = os.path.join(video_dir_path, f"{video_id_str}.mp4")
    
    frames_for_video = []
    default_frame = np.zeros((*frame_resolution_tuple, 3), dtype=np.uint8)

    if not os.path.exists(video_path):
        for _ in range(sequence_len): frames_for_video.append(default_frame.copy())
        return video_id_str, frames_for_video

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        for _ in range(sequence_len): frames_for_video.append(default_frame.copy())
        return video_id_str, frames_for_video

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_video_frames == 0:
        cap.release()
        for _ in range(sequence_len): frames_for_video.append(default_frame.copy())
        return video_id_str, frames_for_video

    frame_indices_to_sample = np.linspace(0, total_video_frames - 1, sequence_len, dtype=int, endpoint=True)

    for idx in frame_indices_to_sample: 
        current_frame_to_add = default_frame.copy()
        if 0 <= idx < total_video_frames :
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                current_frame_to_add = frame
        frames_for_video.append(current_frame_to_add)
    
    while len(frames_for_video) < sequence_len:
        frames_for_video.append(default_frame.copy())
    if len(frames_for_video) > sequence_len: # Should not happen with linspace
        frames_for_video = frames_for_video[:sequence_len]

    cap.release()
    return video_id_str, frames_for_video

def get_frames_for_batch_parallel(df_batch, video_dir, sequence_length=20, resolution=(720, 1280), num_workers=None):
    if num_workers is None:
        num_workers = cpu_count()
    if num_workers <=0 : num_workers = 1

    tasks_args = [
        (row_data, video_dir, sequence_length, resolution)
        for row_data in df_batch.to_dict('records')
    ]

    video_frames_map = {} 

    if num_workers > 1 and len(tasks_args) > 1:
        with Pool(processes=num_workers) as pool:
            results_iterator = pool.imap_unordered(process_single_test_video_for_batch_worker, tasks_args)
            for video_id_res, frames_res in tqdm(results_iterator, total=len(tasks_args), desc="  Fetching Frames", leave=False):
                video_frames_map[video_id_res] = frames_res
    else:
        for task_arg_set in tqdm(tasks_args, desc="  Fetching Frames (Sequential)", leave=False):
            video_id_res, frames_res = process_single_test_video_for_batch_worker(task_arg_set)
            video_frames_map[video_id_res] = frames_res
            
    all_frames_ordered = []
    video_indices_ordered = [] 
    
    for video_id_original_order in df_batch["id"].tolist(): 
        frames_for_current_video = video_frames_map.get(video_id_original_order)
        
        if frames_for_current_video is None or len(frames_for_current_video) != sequence_length:
            print(f"Warning: Video {video_id_original_order} had {len(frames_for_current_video) if frames_for_current_video else 0} frames, expected {sequence_length}. Padding.")
            frames_for_current_video = [np.zeros((*resolution, 3), dtype=np.uint8) for _ in range(sequence_length)]
            
        all_frames_ordered.extend(frames_for_current_video)
        video_indices_ordered.append(len(frames_for_current_video))

    return all_frames_ordered, video_indices_ordered


# --- Main Script ---
if __name__ == "__main__":
    TEST_VIDEO_DIR = "nexar-collision-prediction/test"
    CSV_PATH = "nexar-collision-prediction/test.csv"
    
    VIT_MODEL_NAME = "openai/clip-vit-large-patch14" 
    OUTPUT_DIR_ROOT = f"CLIP_ViT_Features_Test_{VIT_MODEL_NAME.split('/')[-1]}"
    VIT_FRAME_BATCH_SIZE = 32

    FPS_TARGET = 3 
    TIME_WINDOW = 10.0 
    SEQUENCE_LENGTH = int(FPS_TARGET * TIME_WINDOW)

    SAVE_INTERVAL_VIDEOS = 128

    df = pd.read_csv(CSV_PATH)
    df["id"] = df["id"].apply(lambda x: str(x).zfill(5))

    timestamp_str = datetime.now().strftime("%y%m%d_%H%M%S")
    feature_save_dir = os.path.join(OUTPUT_DIR_ROOT, f"run_{timestamp_str}")
    os.makedirs(feature_save_dir, exist_ok=True)
    print(f"Saving features to: {feature_save_dir}")
    print(f"ViT model for feature extraction: {VIT_MODEL_NAME}")
    print(f"ViT processing batch size (frames): {VIT_FRAME_BATCH_SIZE}")
    print(f"Frames sampled per video (SEQUENCE_LENGTH): {SEQUENCE_LENGTH}")

    all_video_ids_list = df["id"].tolist()
    num_total_videos = len(all_video_ids_list)
    num_saving_batches = (num_total_videos + SAVE_INTERVAL_VIDEOS - 1) // SAVE_INTERVAL_VIDEOS

    for batch_idx in tqdm(range(num_saving_batches), desc="Video Saving Batches"):
        start_video_idx_in_df = batch_idx * SAVE_INTERVAL_VIDEOS
        end_video_idx_in_df = min((batch_idx + 1) * SAVE_INTERVAL_VIDEOS, num_total_videos)
        
        df_current_saving_batch = df.iloc[start_video_idx_in_df:end_video_idx_in_df]
        
        if df_current_saving_batch.empty:
            continue

        print(f"\n--- Processing saving_batch {batch_idx + 1}/{num_saving_batches}: videos {start_video_idx_in_df} to {end_video_idx_in_df - 1} ---")

        flat_list_all_frames_for_saving_batch, video_indices_in_saving_batch = get_frames_for_batch_parallel(
            df_current_saving_batch, TEST_VIDEO_DIR, SEQUENCE_LENGTH, num_workers=cpu_count() # Or specify number
        )

        if not flat_list_all_frames_for_saving_batch:
            print(f"No frames collected for saving_batch {batch_idx + 1}. Skipping feature extraction.")
            continue
        
        print(f"  Collected {len(flat_list_all_frames_for_saving_batch)} total frames for this saving_batch.")

        all_features_flat_np = extract_features_batched_hf(
            flat_list_all_frames_for_saving_batch, 
            model_name=VIT_MODEL_NAME, 
            batch_size=VIT_FRAME_BATCH_SIZE
        )

        if all_features_flat_np.size == 0:
            print(f"  No features extracted for saving_batch {batch_idx + 1}. Skipping save.")
            continue

        features_per_video_in_saving_batch = []
        current_feature_array_idx = 0
        for frame_count_for_video in video_indices_in_saving_batch:
            video_features = all_features_flat_np[current_feature_array_idx : current_feature_array_idx + frame_count_for_video]
            features_per_video_in_saving_batch.append(video_features)
            current_feature_array_idx += frame_count_for_video

        np.save(f"{feature_save_dir}/test_features_saving_batch{batch_idx+1}.npy", np.array(features_per_video_in_saving_batch, dtype=object))
        np.save(f"{feature_save_dir}/test_ids_saving_batch{batch_idx+1}.npy", df_current_saving_batch["id"].values)
        print(f"  Saved saving_batch {batch_idx + 1} features and IDs.")

    print(f"\nAll test batches processed and saved in: {feature_save_dir}")
