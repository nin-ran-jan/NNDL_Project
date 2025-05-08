import os
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from .ResNet_model import get_resnet_model
from datasets.video_dataset import FrameBatchDataset

def extract_features_batched(frames, transform, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Compile the model for faster processing
    model = torch.compile(get_resnet_model().to(device).eval())

    dataset = FrameBatchDataset(frames, transform)
    loader = torch.utils.data.DataLoader(dataset, 
                                         batch_size=batch_size, 
                                         shuffle=False,
                                         num_workers=os.cpu_count() or 8,
                                         pin_memory=True,
                                         persistent_workers=True)

    all_features = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting ResNet features"):
            batch = batch.to(device)
            # Mixed precision for faster processing
            # L4 gpu has good support for bfloat16
            with torch.autocast(device.type, dtype=torch.bfloat16):
                feats = model(batch).squeeze(-1).squeeze(-1)
            all_features.append(feats.cpu().numpy())
    all_features = np.vstack(all_features)

    return all_features