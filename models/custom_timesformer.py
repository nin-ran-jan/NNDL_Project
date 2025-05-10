import torch
import torch.nn as nn
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.timesformer.configuration_timesformer import TimesformerConfig

class HFCustomTimeSformer(nn.Module):
    def __init__(self,
                 hf_model_name: str,
                 num_frames_input_clip: int,
                 backbone_feature_dim_config: int = 768, # Configured feature dim, backbone might differ
                 pretrained: bool = True,
                 dropout_rate: float = 0.3, # Added dropout for custom heads
                 freeze_backbone: bool = False # Added option to freeze backbone
                ):
        super().__init__()
        self.num_frames_input_clip = num_frames_input_clip

        if pretrained:
            self.backbone = AutoModel.from_pretrained(hf_model_name)
        else:
            config = TimesformerConfig.from_pretrained(hf_model_name) # Load config
            self.backbone = AutoModel.from_config(config) # Init from config

        self.backbone_actual_feature_dim = self.backbone.config.hidden_size
        
        if freeze_backbone:
            print("Freezing backbone parameters.")
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Custom heads with dropout
        self.dropout = nn.Dropout(dropout_rate)
        self.frame_fc = nn.Linear(self.backbone_actual_feature_dim, 1)
        self.seq_fc = nn.Linear(self.backbone_actual_feature_dim, 1)


    def forward(self, pixel_values: torch.Tensor):
        batch_size = pixel_values.shape[0]
        
        outputs = self.backbone(pixel_values=pixel_values)
        last_hidden_state = outputs.last_hidden_state
        # last_hidden_state shape: (batch_size, 1 (CLS) + num_frames * num_patches_per_frame, hidden_size)
        
        # Sequence-level prediction using CLS token
        cls_token_features = last_hidden_state[:, 0, :]
        seq_representation_after_dropout = self.dropout(cls_token_features)
        seq_logits = self.seq_fc(seq_representation_after_dropout).squeeze(-1)

        # Frame-level prediction using patch tokens
        patch_tokens = last_hidden_state[:, 1:, :]
        num_actual_patch_tokens = patch_tokens.shape[1]

        if num_actual_patch_tokens % self.num_frames_input_clip != 0:
            # This should ideally not happen if CLS token handling is correct
            # and num_frames_input_clip matches how patches are formed.
            print(f"Warning: Actual patch tokens {num_actual_patch_tokens} not cleanly divisible by num_clip_frames {self.num_frames_input_clip}. Frame logits will be zeros.")
            frame_logits = torch.zeros((batch_size, self.num_frames_input_clip), device=pixel_values.device)
        else:
            num_spatial_patches_per_frame = num_actual_patch_tokens // self.num_frames_input_clip
            frame_patch_embeddings = patch_tokens.view(
                batch_size,
                self.num_frames_input_clip,
                num_spatial_patches_per_frame,
                self.backbone_actual_feature_dim
            )
            frame_features = frame_patch_embeddings.mean(dim=2) # Avg spatial patches
            frame_features_after_dropout = self.dropout(frame_features)
            frame_logits = self.frame_fc(frame_features_after_dropout).squeeze(-1)
            
        return frame_logits, seq_logits

    def unfreeze_backbone(self):
        print("Unfreezing backbone parameters.")
        for param in self.backbone.parameters():
            param.requires_grad = True