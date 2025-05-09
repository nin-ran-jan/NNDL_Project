from transformers.models.clip import CLIPVisionModelWithProjection

def get_clip_vision_model(model_name="openai/clip-vit-large-patch14"):
    model = CLIPVisionModelWithProjection.from_pretrained(model_name)
    return model