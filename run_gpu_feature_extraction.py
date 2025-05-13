# run_gpu_feature_extraction.py
from datetime import datetime
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import json
import torch
from torch.utils.data import DataLoader
from transformers.models.clip import CLIPProcessor 

from datasets.gpu_video_dataset import VideoFrameDataset, collate_fn_videos
from models.ViT_model import get_clip_vision_model
from models.Vit_feature_extract import extract_features_single_video_optimized


# AI prompts
# Improve formatting

if __name__ == "__main__":
    BASE_DATA_DIR = "nexar-collision-prediction"
    TRAIN_CSV_PATH = os.path.join(BASE_DATA_DIR, "train.csv")
    TRAIN_VIDEO_DIR = os.path.join(BASE_DATA_DIR, "train")

    FPS_TARGET = 3
    TIME_WINDOW = 10.0
    SEQUENCE_LENGTH = int(FPS_TARGET * TIME_WINDOW)

    VIT_MODEL_NAME = "openai/clip-vit-large-patch14"
    VIDEO_LOADER_BATCH_SIZE = 1
    DATALOADER_NUM_WORKERS = max(0, os.cpu_count() // 2 if os.cpu_count() else 2)
    OUTPUT_SAVE_INTERVAL_VIDEOS = 64

    TARGET_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using target device: {TARGET_DEVICE}")

    if DATALOADER_NUM_WORKERS > 0 and TARGET_DEVICE == 'cuda':
        current_start_method = torch.multiprocessing.get_start_method(allow_none=True)
        if current_start_method != 'spawn':
            try:
                torch.multiprocessing.set_start_method('spawn', force=True)
                print("Set PyTorch multiprocessing start method to 'spawn'.")
            except RuntimeError as e:
                print(f"Warning: Could not set start method to 'spawn': {e}. Using default: {current_start_method}")


    # output directory setup
    timestamp_str = datetime.now().strftime("%y%m%d_%H%M%S")
    base_output_folder = "processed_data"
    os.makedirs(base_output_folder, exist_ok=True)
    model_name_slug = VIT_MODEL_NAME.split('/')[-1]
    feature_save_dir = os.path.join(base_output_folder, f"CLIP_ViT_Features_{model_name_slug}", f"run_{timestamp_str}")
    os.makedirs(feature_save_dir, exist_ok=True)
    print(f"Output features and labels will be saved to: {feature_save_dir}")

    # load video metadata
    df = pd.read_csv(TRAIN_CSV_PATH)
    df["id"] = df["id"].apply(lambda x: str(x).zfill(5))

    num_total_videos = len(df)
    num_output_saving_batches = (num_total_videos + OUTPUT_SAVE_INTERVAL_VIDEOS - 1) // OUTPUT_SAVE_INTERVAL_VIDEOS

    # save run metadata
    run_metadata = {
        "run_timestamp": timestamp_str, "vit_model_name": VIT_MODEL_NAME,
        "fps_target": FPS_TARGET, "time_window_seconds": TIME_WINDOW,
        "sequence_length_frames": SEQUENCE_LENGTH,
        "video_loader_batch_size": VIDEO_LOADER_BATCH_SIZE,
        "dataloader_num_workers": DATALOADER_NUM_WORKERS,
        "output_save_interval_videos": OUTPUT_SAVE_INTERVAL_VIDEOS,
        "target_device": TARGET_DEVICE,
    }
    with open(os.path.join(feature_save_dir, "run_metadata.json"), "w") as f:
        json.dump(run_metadata, f, indent=4)
    print(f"Saved run metadata. Processing {num_total_videos} videos.")

    # initialize model and processor once
    print(f"Initializing model {VIT_MODEL_NAME} and processor...")
    feature_extraction_model = get_clip_vision_model(model_name=VIT_MODEL_NAME).to(TARGET_DEVICE).eval()
    try:
        feature_extraction_model = torch.compile(feature_extraction_model)
        print("ViT model compiled.")
    except Exception as e:
        print(f"ViT model compilation failed: {e}. Using uncompiled model.")
    
    clip_processor_for_dataset = CLIPProcessor.from_pretrained(VIT_MODEL_NAME, use_fast=False)
    print("Model and processor initialized.")

    # initialize dataset and dataloader
    video_dataset = VideoFrameDataset(
        df=df,
        video_dir=TRAIN_VIDEO_DIR,
        fps_target=FPS_TARGET,
        sequence_length=SEQUENCE_LENGTH,
        target_device='cpu'     # Dataset __getitem__ returns CPU tensors. DataLoader moves to GPU.
    )
    
    video_dataloader = DataLoader(
        video_dataset,
        batch_size=VIDEO_LOADER_BATCH_SIZE, # If 1, each item from iterator is (vid_id, frames, labels)
        shuffle=False, # Keep order for now
        num_workers=DATALOADER_NUM_WORKERS,
        collate_fn=collate_fn_videos if VIDEO_LOADER_BATCH_SIZE > 1 else None,
        pin_memory=True if TARGET_DEVICE == 'cuda' and DATALOADER_NUM_WORKERS > 0 else False,
        persistent_workers=True if DATALOADER_NUM_WORKERS > 0 else False, # Can speed up worker reuse
    )

    processed_video_count = 0
    current_saving_batch_features = []
    current_saving_batch_labels = []
    current_saving_batch_video_ids = []

    for batch_data in tqdm(video_dataloader, desc="Processing Videos via DataLoader"):
        
        if VIDEO_LOADER_BATCH_SIZE == 1 and video_dataloader.collate_fn is None:
            video_ids_in_batch = [batch_data[0]]
            frames_in_batch_tensor = batch_data[1].unsqueeze(0)
            labels_in_batch_tensor = batch_data[2].unsqueeze(0)
        else:
            video_ids_in_batch, frames_in_batch_tensor, labels_in_batch_tensor = batch_data

        frames_in_batch_tensor = frames_in_batch_tensor.to(TARGET_DEVICE)
        labels_in_batch_tensor = labels_in_batch_tensor.to(TARGET_DEVICE)


        for i in range(len(video_ids_in_batch)):
            video_id = video_ids_in_batch[i]
            single_video_frames_tensor = frames_in_batch_tensor[i]
            single_video_labels_numpy = labels_in_batch_tensor[i].cpu().numpy()

            if single_video_frames_tensor.nelement() == 0 or single_video_frames_tensor.shape[0] != SEQUENCE_LENGTH:
                print(f"  Warning: Insufficient/empty frames for video {video_id} (shape: {single_video_frames_tensor.shape}). Appending empty.")
                video_features_np = np.array([])
            else:
                video_features_np = extract_features_single_video_optimized(
                    video_frames_tensor_tchw=single_video_frames_tensor,
                    model=feature_extraction_model,
                    processor=clip_processor_for_dataset,
                    target_device=TARGET_DEVICE,
                    internal_model_batch_size=32
                )

            current_saving_batch_features.append(video_features_np)
            current_saving_batch_labels.append(single_video_labels_numpy)
            current_saving_batch_video_ids.append(video_id)
            processed_video_count += 1

            if processed_video_count % OUTPUT_SAVE_INTERVAL_VIDEOS == 0 or processed_video_count == num_total_videos:
                if not current_saving_batch_video_ids: continue

                current_output_batch_num = (processed_video_count + OUTPUT_SAVE_INTERVAL_VIDEOS - 1) // OUTPUT_SAVE_INTERVAL_VIDEOS
                print(f"\nSaving Output Batch {current_output_batch_num} ({len(current_saving_batch_video_ids)} videos)...")
                batch_file_suffix = f"saving_batch_{current_output_batch_num}"
                save_path_features = os.path.join(feature_save_dir, f"train_features_{batch_file_suffix}.npy")
                save_path_labels = os.path.join(feature_save_dir, f"train_labels_{batch_file_suffix}.npy")
                save_path_video_ids = os.path.join(feature_save_dir, f"train_video_ids_{batch_file_suffix}.json")

                try:
                    np.save(save_path_features, np.array(current_saving_batch_features, dtype=object))
                    np.save(save_path_labels, np.array(current_saving_batch_labels, dtype=object))
                    with open(save_path_video_ids, "w") as f:
                        json.dump(current_saving_batch_video_ids, f, indent=4)
                    print(f"  Successfully saved batch {current_output_batch_num}.")
                except Exception as e:
                    print(f"  Error saving data for output batch {current_output_batch_num}: {e}")
                
                current_saving_batch_features, current_saving_batch_labels, current_saving_batch_video_ids = [], [], []
    
    if current_saving_batch_video_ids:
        current_output_batch_num = (processed_video_count + OUTPUT_SAVE_INTERVAL_VIDEOS - 1) // OUTPUT_SAVE_INTERVAL_VIDEOS
        if processed_video_count % OUTPUT_SAVE_INTERVAL_VIDEOS != 0:
            current_output_batch_num = num_output_saving_batches
        print(f"\nSaving Final Output Batch {current_output_batch_num} ({len(current_saving_batch_video_ids)} videos)...")
        batch_file_suffix = f"saving_batch_{current_output_batch_num}"
        save_path_features = os.path.join(feature_save_dir, f"train_features_{batch_file_suffix}.npy")
        save_path_labels = os.path.join(feature_save_dir, f"train_labels_{batch_file_suffix}.npy")
        save_path_video_ids = os.path.join(feature_save_dir, f"train_video_ids_{batch_file_suffix}.json")
        try:
            np.save(save_path_features, np.array(current_saving_batch_features, dtype=object))
            np.save(save_path_labels, np.array(current_saving_batch_labels, dtype=object))
            with open(save_path_video_ids, "w") as f:
                json.dump(current_saving_batch_video_ids, f, indent=4)
            print(f"  Successfully saved final batch {current_output_batch_num}.")
        except Exception as e:
            print(f"  Error saving data for final output batch: {e}")


    print("\nAll video processing and feature saving completed.")