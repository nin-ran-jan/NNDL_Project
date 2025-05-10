# models/videomae_model.py
from transformers import VideoMAEForVideoClassification

def get_videomae_model(num_labels=2):
    model = VideoMAEForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-base",
        num_labels=num_labels
    )
    return model
