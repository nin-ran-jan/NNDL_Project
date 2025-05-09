import torch
import numpy as np
import os
from tqdm import tqdm
import cv2
from PIL import Image
from transformers.models.clip import CLIPProcessor

from datasets.video_dataset import FrameBatchDataset
from models.ViT_model import get_clip_vision_model 

def pil_list_collate_fn(batch):
    """
    Collate function for DataLoader.
    Receives a list of PIL Images (a batch from the Dataset)
    and returns this list directly.
    """
    return batch

def extract_features_batched_hf(all_numpy_frames,
                                 model_name="openai/clip-vit-large-patch14",
                                 batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = torch.compile(get_clip_vision_model(model_name=model_name).to(device).eval())
    # processor form huggingface
    processor = CLIPProcessor.from_pretrained(model_name, use_fast=False)

    # Define the transform to convert BGR NumPy array to RGB PIL Image
    def bgr_numpy_to_rgb_pil(numpy_frame):
        return Image.fromarray(cv2.cvtColor(numpy_frame, cv2.COLOR_BGR2RGB))

    dataset = FrameBatchDataset(all_numpy_frames, transform=bgr_numpy_to_rgb_pil)

    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=os.cpu_count() or 4, # Can be tuned
                                         pin_memory=True if device.type == 'cuda' else False,
                                         persistent_workers=False,
                                         collate_fn=pil_list_collate_fn
                                         )


    all_features_list = []
    with torch.no_grad():
        for batch_pil_images in tqdm(loader, desc=f"Extracting {model_name} features"):
            # huggingface processor takes care of preprocessing
            inputs = processor(images=batch_pil_images, return_tensors="pt", padding=True).to(device)
            pixel_values = inputs['pixel_values']

            # L4 can work with bfloat16
            amp_dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
            with torch.autocast(device_type=device.type if device.type != 'mps' else 'cpu', dtype=amp_dtype):
                outputs = model(pixel_values=pixel_values)
                image_embeds = outputs.image_embeds

            processed_image_embeds = image_embeds.cpu().to(torch.float32).numpy()

            all_features_list.append(processed_image_embeds)

    if not all_features_list:
        return np.array([])
        
    all_features_np = np.vstack(all_features_list)
    return all_features_np

def extract_features_single_video_optimized(
    video_frames_tensor_tchw,
    model,
    processor,
    target_device,
    internal_model_batch_size=32
):
    # --- BEGIN EXTRACTOR DEBUG ---
    print(f"    Extractor DEBUG: Received video_frames_tensor_tchw.shape: {video_frames_tensor_tchw.shape}, dtype: {video_frames_tensor_tchw.dtype}")
    # --- END EXTRACTOR DEBUG ---

    if video_frames_tensor_tchw.device.type != target_device: # Should be on target_device already from main script
        video_frames_tensor_tchw = video_frames_tensor_tchw.to(target_device)

    num_frames = video_frames_tensor_tchw.shape[0]
    if num_frames == 0:
        print("    Extractor DEBUG: num_frames is 0. Returning empty np.array.")
        return np.array([])

    all_image_embeds_list = []
    
    with torch.no_grad():
        for i_loop in range(0, num_frames, internal_model_batch_size):
            batch_of_frames_for_processor = video_frames_tensor_tchw[i_loop : i_loop + internal_model_batch_size]
            # --- BEGIN EXTRACTOR LOOP DEBUG ---
            print(f"      Extractor Loop DEBUG: Processing frame batch {i_loop // internal_model_batch_size + 1}, shape: {batch_of_frames_for_processor.shape}")
            # --- END EXTRACTOR LOOP DEBUG ---
            
            try:
                inputs = processor(images=batch_of_frames_for_processor, return_tensors="pt", padding=True)
                pixel_values = inputs['pixel_values'].to(target_device)
                # --- MORE EXTRACTOR DEBUG ---
                print(f"        Extractor DEBUG: Processor output pixel_values.shape: {pixel_values.shape}, dtype: {pixel_values.dtype}")
                # --- END MORE EXTRACTOR DEBUG ---

                model_inputs = {'pixel_values': pixel_values}
                
                amp_dtype = torch.bfloat16 if target_device == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
                if target_device == 'cpu': amp_dtype = torch.float32

                with torch.autocast(device_type=target_device, dtype=amp_dtype, enabled=(target_device != 'cpu')):
                    outputs = model(**model_inputs)
                    image_embeds_batch = outputs.image_embeds if hasattr(outputs, 'image_embeds') else outputs.last_hidden_state[:, 0, :]
                
                # --- MORE EXTRACTOR DEBUG ---
                print(f"        Extractor DEBUG: Model output image_embeds_batch.shape: {image_embeds_batch.shape}, dtype: {image_embeds_batch.dtype}")
                # --- END MORE EXTRACTOR DEBUG ---
                all_image_embeds_list.append(image_embeds_batch.cpu().to(torch.float32))

            except Exception as e_inner:
                print(f"      ERROR in Extractor Loop for video (frames {i_loop}-{i_loop+internal_model_batch_size}): {type(e_inner).__name__} - {e_inner}")
                # Decide if you want to continue or return empty if any batch fails
                # For now, let it continue and potentially result in an empty all_image_embeds_list if all fail

    if not all_image_embeds_list:
        print("    Extractor DEBUG: all_image_embeds_list is empty after loop. Returning empty np.array.")
        return np.array([])
        
    try:
        all_features_tensor = torch.cat(all_image_embeds_list, dim=0)
        # --- MORE EXTRACTOR DEBUG ---
        print(f"    Extractor DEBUG: Concatenated all_features_tensor.shape: {all_features_tensor.shape}")
        # --- END MORE EXTRACTOR DEBUG ---
        return all_features_tensor.numpy()
    except Exception as e_cat:
        print(f"    Extractor DEBUG: ERROR during torch.cat: {e_cat}. all_image_embeds_list content: {[(t.shape, t.dtype) for t in all_image_embeds_list]}")
        print("    Extractor DEBUG: Returning empty np.array due to torch.cat error.")
        return np.array([])