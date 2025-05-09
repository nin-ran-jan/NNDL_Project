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