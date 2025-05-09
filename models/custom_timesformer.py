# hf_timesformer_pipeline/models/hf_custom_timesformer.py
import torch
import torch.nn as nn
from transformers.models.auto.modeling_auto import AutoModel # Use TimesformerModel for more control if needed
from transformers.models.timesformer.configuration_timesformer import TimesformerConfig # Use TimesformerModel for more control if needed

class HFCustomTimeSformer(nn.Module):
    def __init__(self,
                 hf_model_name: str,      # e.g., "facebook/timesformer-base-finetuned-k400"
                 num_frames_input_clip: int,
                 backbone_feature_dim: int = 768, # Typically 768 for "base" ViT/TimeSformer
                 pretrained: bool = True):
        super().__init__()
        self.num_frames_input_clip = num_frames_input_clip # Number of frames in each clip

        if pretrained:
            self.backbone = AutoModel.from_pretrained(hf_model_name)
        else:
            # For training from scratch or with a custom config
            config = TimesformerConfig.from_pretrained(hf_model_name) # Loads config
            self.backbone = AutoModel.from_config(config) # Initializes with config

        # The output `last_hidden_state` of Hugging Face's `TimesformerModel` (which AutoModel often loads)
        # is (batch_size, num_frames * num_patches_per_frame, hidden_size).
        # It does NOT include a CLS token in this specific output tensor.
        # num_patches_per_frame = (config.image_size // config.patch_size) ** 2
        
        # We will average spatial patches for each frame to get per-frame features.
        self.frame_fc = nn.Linear(backbone_feature_dim, 1)
        self.seq_fc = nn.Linear(backbone_feature_dim, 1) # For sequence-level prediction

    def forward(self, pixel_values: torch.Tensor):
        # Input pixel_values expected shape: (batch_size, num_frames, num_channels, height, width)
        # This is the standard for Hugging Face VideoMAE and TimeSformer.
        
        batch_size = pixel_values.shape[0]
        
        # Get backbone outputs
        # For `transformers.TimesformerModel`, `last_hidden_state` is what we need.
        outputs = self.backbone(pixel_values=pixel_values)
        last_hidden_state = outputs.last_hidden_state
        # Shape: (batch_size, total_num_patches, hidden_size)
        # where total_num_patches = num_frames_input_clip * (image_size // patch_size)^2

        # Reshape and average spatial patches to get per-frame features
        # config.hidden_size gives backbone_feature_dim
        # config.patch_size and config.image_size can give num_spatial_patches_per_frame
        
        # Infer num_spatial_patches_per_frame
        num_total_patches = last_hidden_state.shape[1]
        if num_total_patches % self.num_frames_input_clip != 0:
            # This can happen if the model includes a CLS token in last_hidden_state,
            # or if num_frames from data doesn't match what model was trained on.
            # Most HF base TimesformerModel's last_hidden_state is just patch embeddings.
            print(f"Warning: Total patches {num_total_patches} not cleanly divisible by num_clip_frames {self.num_frames_input_clip}. Model output structure might be different than expected.")
            # Fallback: use mean of all patches for sequence, zeros for frames (crude)
            seq_representation = last_hidden_state.mean(dim=1)
            frame_logits = torch.zeros((batch_size, self.num_frames_input_clip), device=pixel_values.device)
        else:
            num_spatial_patches_per_frame = num_total_patches // self.num_frames_input_clip
            
            # Reshape to (batch_size, num_frames, num_spatial_patches_per_frame, hidden_size)
            frame_patch_embeddings = last_hidden_state.view(
                batch_size,
                self.num_frames_input_clip,
                num_spatial_patches_per_frame,
                self.backbone.config.hidden_size # Or backbone_feature_dim
            )
            # Average over spatial patches to get per-frame features
            frame_features = frame_patch_embeddings.mean(dim=2)  # Shape: (batch_size, num_frames, hidden_size)

            # Frame-level predictions (logits)
            frame_logits = self.frame_fc(frame_features).squeeze(-1)  # Shape: (batch_size, num_frames)

            # Sequence-level prediction (logits) - e.g., by averaging frame features
            seq_representation = frame_features.mean(dim=1) # Shape: (batch_size, hidden_size)
        
        seq_logits = self.seq_fc(seq_representation).squeeze(-1) # Shape: (batch_size)

        # Return logits; Sigmoid will be applied in loss function (BCEWithLogitsLoss) or for inference
        return frame_logits, seq_logits