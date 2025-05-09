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
    video_frames_tensor_tchw, # Expects (T, C, H, W) uint8 or float32 tensor for ONE video
    model,                    # Pre-initialized PyTorch model, already on target_device
    processor,                # Pre-initialized CLIPProcessor
    target_device,            # The device the model is on ('cuda' or 'cpu')
    internal_model_batch_size=32 # Frames to feed to CLIP model at once
):
    # Ensure input tensor is on the same device as the model
    if video_frames_tensor_tchw.device.type != target_device:
        video_frames_tensor_tchw = video_frames_tensor_tchw.to(target_device)

    num_frames = video_frames_tensor_tchw.shape[0]
    if num_frames == 0:
        return np.array([])

    all_image_embeds_list = []
    
    # The CLIPProcessor expects images (PIL, numpy, or PyTorch tensors).
    # If input is uint8 TCHW tensor, processor handles conversion & normalization.
    # If input is float32 TCHW tensor, it should ideally be normalized already.
    # VideoFrameDataset passes uint8 TCHW if self.transform is None, which is good for processor.
    
    with torch.no_grad():
        for i in range(0, num_frames, internal_model_batch_size):
            batch_of_frames_for_processor = video_frames_tensor_tchw[i : i + internal_model_batch_size]
            
            # `processor` handles resizing, normalization, and converts to (B, C, H_proc, W_proc)
            inputs = processor(images=batch_of_frames_for_processor, return_tensors="pt", padding=True)
            
            # Move processed inputs to the target device
            pixel_values = inputs['pixel_values'].to(target_device)
            # attention_mask = inputs.get('attention_mask', None) # Optional
            # if attention_mask is not None: attention_mask = attention_mask.to(target_device)
            
            model_inputs = {'pixel_values': pixel_values}
            # if attention_mask is not None: model_inputs['attention_mask'] = attention_mask

            amp_dtype = torch.bfloat16 if target_device == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
            if target_device == 'cpu': amp_dtype = torch.float32

            with torch.autocast(device_type=target_device, dtype=amp_dtype, enabled=(target_device != 'cpu')):
                outputs = model(**model_inputs)
                # Assuming model is CLIPVisionModelWithProjection, outputs.image_embeds exists
                # Or use outputs.pooler_output or last_hidden_state[:,0] if it's a base vision model
                image_embeds_batch = outputs.image_embeds if hasattr(outputs, 'image_embeds') else outputs.last_hidden_state[:, 0, :]


            all_image_embeds_list.append(image_embeds_batch.cpu().to(torch.float32))

    if not all_image_embeds_list:
        return np.array([])
        
    all_features_tensor = torch.cat(all_image_embeds_list, dim=0)
    return all_features_tensor.numpy()