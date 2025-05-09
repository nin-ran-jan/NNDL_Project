# timesformer_pipeline/models/custom_timesformer.py
import torch
import torch.nn as nn
import torch.hub

class CustomTimeSformer(nn.Module):
    def __init__(self,
                 num_input_clip_frames: int, # e.g., 8 or 16, must match dataset
                 backbone_name: str = "timesformer_base_patch16_224", # From PyTorchVideo
                 pretrained: bool = True,
                 # Specify dimensions for custom heads based on backbone output
                 # For timesformer_base_patch16_224, the feature dim is typically 768
                 backbone_feature_dim: int = 768
                ):
        super().__init__()
        self.num_input_clip_frames = num_input_clip_frames

        # Load pretrained TimeSformer from PyTorchVideo
        try:
            self.backbone = torch.hub.load("facebookresearch/pytorchvideo", model=backbone_name, pretrained=pretrained)
        except Exception as e:
            print(f"Could not load {backbone_name} from pytorchvideo. Ensure you have internet and pytorchvideo installed.")
            print(f"Error: {e}")
            raise

        # --- Strategy to get per-frame features ---
        # We need to modify the backbone or tap into its intermediate layers.
        # The standard TimeSformer head performs temporal pooling then projection.
        # We want features *before* this temporal pooling, for each of the input_clip_frames.

        # For pytorchvideo.models.TimeSformer, the features before the head's projection
        # are typically the CLS token output or an average of patch tokens.
        # To get per-frame features, we need to be a bit more invasive or hope for a specific API.
        # Let's assume a scenario where we can access the sequence of patch embeddings
        # after they've passed through the transformer blocks.
        # The `self.backbone.model.blocks` are the transformer layers.
        # The output of `self.backbone.model.norm` (if it's applied to the sequence)
        # might be `(batch_size, num_patches_total, backbone_feature_dim)`.
        # `num_patches_total` = `num_input_clip_frames * num_spatial_patches_per_frame`.

        # We'll try to "hijack" the forward pass or use components.
        # This is a simplification; true per-frame features from TimeSformer might require
        # averaging patch tokens corresponding to each frame or specific model surgery.

        # For this example, let's make a placeholder assumption that we can get (B, T_clip, D)
        # If not, this part needs significant refinement based on the chosen TimeSformer library API.
        # A common approach: remove the original head.
        original_head = self.backbone.head
        self.backbone.head = nn.Identity() # Remove original classification head

        # If we can ensure the output of self.backbone(video_clips) is now (B, T_clip, D_feat)
        # or (B, N_total_patches, D_feat) which we then need to pool spatially per frame.
        # This is highly dependent on the specific TimeSformer internal structure.

        # Let's define new heads based on `backbone_feature_dim`
        self.frame_fc = nn.Linear(backbone_feature_dim, 1)
        self.seq_fc = nn.Linear(backbone_feature_dim, 1) # Assumes we average frame features for seq rep

    def forward(self, video_clips: torch.Tensor):
        # video_clips shape: (batch_size, channels, num_input_clip_frames, height, width)

        # --- Attempt to get per-frame features from the backbone ---
        # This is the most CRITICAL and model-specific part.
        # The following is a conceptual sketch.
        
        x = video_clips
        # Pass through patch embedding and positional embedding
        x = self.backbone.model.patch_embed(x) # (B, N_total_patches, D)
        
        # For TimeSformer, N_total_patches = T_clip * H_patches * W_patches
        # T_clip is self.num_input_clip_frames
        # H_patches = H / patch_size, W_patches = W / patch_size
        
        # Add positional and temporal embeddings
        cls_token = self.backbone.model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1) # (B, 1 + N_total_patches, D)
        
        x = x + self.backbone.model.pos_embed # Positional embedding
        x = x + self.backbone.model.temporal_embed # Temporal embedding for TimeSformer
        
        x = self.backbone.model.pos_drop(x)

        # Pass through Transformer blocks
        for blk in self.backbone.model.blocks:
            x = blk(x)
        
        x = self.backbone.model.norm(x) # (B, 1 + N_total_patches, D)

        # Now, x contains features for CLS token and all patch tokens.
        # We need to aggregate patch tokens for each frame to get per-frame features.
        # Remove CLS token:
        patch_tokens = x[:, 1:, :] # (B, N_total_patches, D)
        
        # Reshape to separate frame dimension from spatial patch dimension
        # N_total_patches = num_input_clip_frames * num_spatial_patches
        # num_spatial_patches = (H_input / patch_size_H) * (W_input / patch_size_W)
        # Example: For 224x224 input and 16x16 patches, num_spatial_patches = (224/16)*(224/16) = 14*14 = 196
        
        # Get num_spatial_patches (this is model dependent, but for ViT L/14 on 224x224, it's (224/14)^2 = 16^2 = 256)
        # For TimeSformer base_patch16_224, spatial patch size is 16.
        # num_spatial_patches_per_frame = (target_spatial_size_H // 16) * (target_spatial_size_W // 16)
        # Example: (224 // 16) * (224 // 16) = 14 * 14 = 196
        # So, patch_tokens should be (B, num_input_clip_frames * 196, D)

        # We need to know num_spatial_patches to reshape.
        # This is hardcoded based on common TimeSformer structure.
        # A more robust way would be to infer this from model config if possible.
        num_spatial_patches = patch_tokens.shape[1] // self.num_input_clip_frames
        
        if patch_tokens.shape[1] % self.num_input_clip_frames != 0:
            print(f"Warning: Total patches {patch_tokens.shape[1]} not divisible by num_clip_frames {self.num_input_clip_frames}. Frame feature extraction might be incorrect.")
            # Fallback: use CLS token for sequence, and zeros for frames (not ideal)
            cls_features = x[:, 0, :] # (B, D)
            seq_logits = self.seq_fc(cls_features).squeeze(-1)
            seq_probs = torch.sigmoid(seq_logits)
            frame_probs = torch.zeros((x.shape[0], self.num_input_clip_frames), device=x.device)
            return frame_probs, seq_probs

        # Reshape to (B, num_input_clip_frames, num_spatial_patches, D)
        reshaped_patch_tokens = patch_tokens.view(
            patch_tokens.shape[0],
            self.num_input_clip_frames,
            num_spatial_patches,
            patch_tokens.shape[-1]
        )

        # Average spatial patches to get per-frame features
        frame_features = reshaped_patch_tokens.mean(dim=2)  # (B, num_input_clip_frames, D)

        # Frame-level predictions
        frame_logits = self.frame_fc(frame_features).squeeze(-1)  # (B, num_input_clip_frames)
        frame_probs = torch.sigmoid(frame_logits)

        # Sequence-level predictions (e.g., by averaging frame_features or using CLS token)
        # Option 1: Average our derived frame_features
        seq_representation = frame_features.mean(dim=1)  # (B, D)
        # Option 2: Use the CLS token directly if preferred
        # seq_representation = x[:, 0, :] # (B,D) from earlier computation of x

        seq_logits = self.seq_fc(seq_representation).squeeze(-1) # (B)
        seq_probs = torch.sigmoid(seq_logits)

        return frame_probs, seq_probs