import torch
import torch.nn as nn
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.timesformer.configuration_timesformer import TimesformerConfig

class HFCustomTimeSformer(nn.Module):
    def __init__(self,
                 hf_model_name: str,
                 num_frames_input_clip: int,
                 backbone_feature_dim: int = 768,
                 pretrained: bool = True):
        super().__init__()
        self.num_frames_input_clip = num_frames_input_clip

        if pretrained:
            self.backbone = AutoModel.from_pretrained(hf_model_name)
        else:
            config = TimesformerConfig.from_pretrained(hf_model_name)
            self.backbone = AutoModel.from_config(config)
        
        # Ensure backbone_feature_dim matches the actual hidden size from the loaded model's config
        self.backbone_feature_dim = self.backbone.config.hidden_size

        self.frame_fc = nn.Linear(self.backbone_feature_dim, 1)
        self.seq_fc = nn.Linear(self.backbone_feature_dim, 1)

    def forward(self, pixel_values: torch.Tensor):
        batch_size = pixel_values.shape[0]
        
        outputs = self.backbone(pixel_values=pixel_values)
        last_hidden_state = outputs.last_hidden_state
        # Expected shape of last_hidden_state: (batch_size, 1 + num_frames * num_patches_per_frame, hidden_size)
        # where the first token [:, 0, :] is the CLS token.

        # --- Sequence-level prediction using CLS token ---
        # The CLS token is typically used for sequence-level tasks.
        cls_token_features = last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        seq_logits = self.seq_fc(cls_token_features).squeeze(-1) # (batch_size)

        # --- Frame-level prediction using patch tokens ---
        # Exclude the CLS token to get only patch embeddings
        patch_tokens = last_hidden_state[:, 1:, :]  # (batch_size, num_frames * num_patches_per_frame, hidden_size)
        
        num_actual_patch_tokens = patch_tokens.shape[1]

        if num_actual_patch_tokens % self.num_frames_input_clip != 0:
            print(f"Warning: Actual patch tokens {num_actual_patch_tokens} not cleanly divisible by num_clip_frames {self.num_frames_input_clip}. Frame logits will be zeros.")
            frame_logits = torch.zeros((batch_size, self.num_frames_input_clip), device=pixel_values.device)
        else:
            num_spatial_patches_per_frame = num_actual_patch_tokens // self.num_frames_input_clip
            
            # Reshape to (batch_size, num_frames, num_spatial_patches_per_frame, hidden_size)
            frame_patch_embeddings = patch_tokens.view(
                batch_size,
                self.num_frames_input_clip,
                num_spatial_patches_per_frame,
                self.backbone_feature_dim
            )
            # Average over spatial patches to get per-frame features
            frame_features = frame_patch_embeddings.mean(dim=2)  # Shape: (batch_size, num_frames, hidden_size)

            # Frame-level predictions (logits)
            frame_logits = self.frame_fc(frame_features).squeeze(-1)  # Shape: (batch_size, num_frames)
            
        return frame_logits, seq_logits