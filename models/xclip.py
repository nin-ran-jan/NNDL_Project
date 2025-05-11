import torch
import torch.nn as nn
from transformers.models.x_clip import XCLIPModel, XCLIPConfig
import torch.nn.functional as F


class CustomXCLIPModel(nn.Module):
    def __init__(self,
                model_name: str,
                num_frames: int = 8,
                dropout_rate: float = 0.3,
                pretrained: bool = True,
                freeze_backbone: bool = True,
                freeze_text_model: bool = True,
                freeze_projection: bool = True):
        super().__init__()
        
        self.num_frames = num_frames
        
        if pretrained:
            self.model = XCLIPModel.from_pretrained(model_name)
        else:
            config = XCLIPConfig.from_pretrained(model_name)
            self.model = XCLIPModel(config)
        
        # Get model dimensions
        self.hidden_dim = self.model.config.vision_config.hidden_size
        self.projection_dim = self.model.config.projection_dim
        
        # Freeze components as required
        if freeze_backbone:
            self.freeze_vision_model()
        
        if freeze_text_model:
            self.freeze_text_model()
            
        if freeze_projection:
            self.freeze_projection_layers()
        
        # Custom heads for our task
        self.dropout = nn.Dropout(dropout_rate)
        
        # Frame-level classifier
        self.frame_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )
        
        # Sequence-level classifier from projection space
        self.seq_classifier = nn.Sequential(
            nn.Linear(self.projection_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate/2),
            nn.Linear(256, 1)
        )
        
        # Initialize dummy text inputs with "collision" and "no collision" prompts
        # This will give the model context even though we're not using actual text input
        self._initialize_text_features()
    
    def _initialize_text_features(self):
        # Will be initialized with actual tokenizer in initialize_text_features method
        self.register_buffer("text_features", torch.zeros(2, self.projection_dim))
        self.text_initialized = False
    
    def initialize_text_features(self, tokenizer):
        """Initialize the class token features for "collision" and "no collision" """
        if self.text_initialized:
            return
            
        # Using specific prompts for our dashcam collision task
        prompts = [
            "a dashcam video of a traffic collision accident",
            "a dashcam video of a near-collision dangerous situation",
            "a dashcam video of normal driving with no incidents",
            "a dashcam video showing safe normal driving conditions"
        ]
        
        text_inputs = tokenizer(
            prompts, 
            padding=True, 
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            text_outputs = self.model.get_text_features(**text_inputs)
            # Store positive prompts (first two) and negative prompts (last two)
            positive_embeds = text_outputs[:2].mean(dim=0, keepdim=True)
            negative_embeds = text_outputs[2:].mean(dim=0, keepdim=True)
            text_features = torch.cat([positive_embeds, negative_embeds], dim=0)
            self.text_features.copy_(text_features)
            
        self.text_initialized = True
    
    def freeze_vision_model(self):
        """Freeze the vision encoder parameters"""
        for param in self.model.vision_model.parameters():
            param.requires_grad = False
            
        # Also freeze the multiframe integration transformer
        if hasattr(self.model, 'mit') and self.model.mit is not None:
            for param in self.model.mit.parameters():
                param.requires_grad = False
    
    def freeze_text_model(self):
        """Freeze the text encoder parameters"""
        for param in self.model.text_model.parameters():
            param.requires_grad = False
    
    def freeze_projection_layers(self):
        """Freeze the projection layers"""
        if hasattr(self.model, 'visual_projection'):
            for param in self.model.visual_projection.parameters():
                param.requires_grad = False
            
        if hasattr(self.model, 'text_projection'):
            for param in self.model.text_projection.parameters():
                param.requires_grad = False
    
    def unfreeze_vision_model(self):
        """Unfreeze the vision encoder parameters"""
        for param in self.model.vision_model.parameters():
            param.requires_grad = True
            
        # Also unfreeze the multiframe integration transformer
        if hasattr(self.model, 'mit') and self.model.mit is not None:
            for param in self.model.mit.parameters():
                param.requires_grad = True
    
    def forward(self, pixel_values):
        """
        Forward pass through the model
        
        Args:
            pixel_values: Tensor of shape [batch_size, num_frames, channels, height, width]
            
        Returns:
            frame_logits: Frame-level predictions [batch_size, num_frames]
            seq_logits: Sequence-level prediction [batch_size]
        """
        batch_size = pixel_values.shape[0]
        
        # Get video features from the model
        video_features = self.model.get_video_features(pixel_values=pixel_values)
        
        # For frame-level features, we need to access the vision model outputs
        # We need to restructure the input for this
        pixel_values_flat = pixel_values.reshape(-1, 3, 224, 224)
        
        vision_outputs = self.model.vision_model(pixel_values=pixel_values_flat)
        frame_tokens = vision_outputs.last_hidden_state
        
        # Extract patch embeddings for each frame (excluding CLS token)
        patch_tokens = frame_tokens[:, 1:, :]
        
        # Reshape to [batch_size, num_frames, num_patches, hidden_dim]
        num_patches = patch_tokens.shape[1] 
        patch_tokens = patch_tokens.reshape(batch_size, self.num_frames, num_patches, -1)
        
        # Average over spatial patches for each frame
        frame_features = patch_tokens.mean(dim=2)  # [batch_size, num_frames, hidden_dim]
        
        # Apply frame classifier
        frame_features = self.dropout(frame_features)
        frame_logits = self.frame_classifier(frame_features).squeeze(-1)  # [batch_size, num_frames]
        
        # Apply sequence classifier to video features
        video_features = self.dropout(video_features)
        seq_logits = self.seq_classifier(video_features).squeeze(-1)  # [batch_size]
        
        return frame_logits, seq_logits