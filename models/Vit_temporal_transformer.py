import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class TemporalPositionalEncoding(nn.Module):
    """Enhanced positional encoding with temporal awareness"""
    def __init__(self, d_model, dropout=0.1, max_len=30):
        super(TemporalPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Standard positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Emphasize temporal positions toward the end of the sequence
        temporal_weights = torch.linspace(1.0, 2.0, max_len).unsqueeze(1)
        pe = pe * temporal_weights
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TemporalAttention(nn.Module):
    """Attention mechanism with causal masking and time decay"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(TemporalAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
    def forward(self, query, key, value, key_padding_mask=None):
        # Generate causal mask (each position can only attend to previous positions)
        seq_len = query.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(query.device)
        
        # Apply attention with causal mask
        attn_output, attn_weights = self.mha(
            query, key, value, 
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask
        )
        
        return attn_output, attn_weights

class TemporalFeedForward(nn.Module):
    """Feed-forward network with temporal convolutions"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(TemporalFeedForward, self).__init__()
        self.temporal_conv1 = nn.Conv1d(d_model, d_ff, kernel_size=3, padding=1)
        self.temporal_conv2 = nn.Conv1d(d_ff, d_model, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        residual = x
        
        # Temporal convolutions (batch, d_model, seq_len)
        x = x.transpose(1, 2)
        x = self.temporal_conv1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.temporal_conv2(x)
        x = x.transpose(1, 2)  # back to (batch, seq_len, d_model)
        
        # Add & norm
        x = residual + x
        x = self.norm(x)
        
        return x

class TemporalTransformerEncoderLayer(nn.Module):
    """Enhanced transformer encoder layer with temporal components"""
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TemporalTransformerEncoderLayer, self).__init__()
        self.temporal_attn = TemporalAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.temporal_ffn = TemporalFeedForward(d_model, dim_feedforward, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_key_padding_mask=None):
        # Self-attention block
        residual = src
        src2, _ = self.temporal_attn(src, src, src, key_padding_mask=src_key_padding_mask)
        src = residual + self.dropout(src2)
        src = self.norm1(src)
        
        # Temporal feed-forward block
        src = self.temporal_ffn(src)
        
        return src

class TemporalTransformer(nn.Module):
    """Complete temporal transformer for accident prediction"""
    def __init__(self, feature_dim, model_dim, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(TemporalTransformer, self).__init__()
        self.model_type = 'Temporal Transformer'
        self.feature_dim = feature_dim
        self.model_dim = model_dim
        
        # Feature projection
        self.input_projection = nn.Linear(feature_dim, model_dim)
        
        # Temporal feature extraction
        self.temporal_conv = nn.Conv1d(model_dim, model_dim, kernel_size=5, padding=2)
        
        # Positional encoding
        self.pos_encoder = TemporalPositionalEncoding(model_dim, dropout)
        
        # Temporal transformer layers
        self.temporal_layers = nn.ModuleList([
            TemporalTransformerEncoderLayer(
                model_dim, nhead, dim_feedforward, dropout
            ) for _ in range(num_encoder_layers)
        ])
        
        # Time-to-accident regression head
        self.tta_regression = nn.Linear(model_dim, 1)
        self.tta_activation = nn.Sigmoid()  # Normalize to [0,1]
        
        # Frame-level prediction head
        self.frame_fc = nn.Linear(model_dim, 1)
        self.frame_sigmoid = nn.Sigmoid()
        
        # Sequence-level prediction head
        self.seq_fc = nn.Linear(model_dim, 1)
        self.seq_sigmoid = nn.Sigmoid()
        
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights for better convergence"""
        initrange = 0.1
        
        # Initialize projection layer
        nn.init.xavier_uniform_(self.input_projection.weight)
        if hasattr(self.input_projection, 'bias') and self.input_projection.bias is not None:
            self.input_projection.bias.data.zero_()
            
        # Initialize temporal conv
        nn.init.xavier_uniform_(self.temporal_conv.weight)
        if hasattr(self.temporal_conv, 'bias') and self.temporal_conv.bias is not None:
            self.temporal_conv.bias.data.zero_()
        
        # Initialize prediction heads
        nn.init.xavier_uniform_(self.seq_fc.weight)
        self.seq_fc.bias.data.zero_()
        
        nn.init.xavier_uniform_(self.frame_fc.weight)
        self.frame_fc.bias.data.zero_()
        
        nn.init.xavier_uniform_(self.tta_regression.weight)
        self.tta_regression.bias.data.zero_()
        
    def temporal_feature_extraction(self, x):
        """Extract temporal features using convolution"""
        # x: (batch_size, seq_len, model_dim)
        x_t = x.transpose(1, 2)  # -> (batch_size, model_dim, seq_len)
        x_t = self.temporal_conv(x_t)
        x_t = F.gelu(x_t)
        x_t = x_t.transpose(1, 2)  # -> (batch_size, seq_len, model_dim)
        return x_t
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Forward pass of the temporal transformer.
        
        Args:
            src (Tensor): Input tensor of shape (batch_size, seq_len, feature_dim).
            src_mask (Tensor, optional): Mask for the src sequence (unused in this implementation).
            src_key_padding_mask (Tensor, optional): Padding mask for source sequence.
                                                   Shape: (batch_size, seq_len).
                                                   
        Returns:
            frame_probs (Tensor): Frame-level accident probabilities, shape (batch_size, seq_len).
            seq_probs (Tensor): Sequence-level accident probability, shape (batch_size).
            tta_predictions (Tensor): Time-to-accident predictions, shape (batch_size, seq_len).
        """
        # Project features to model dimension
        projected_src = self.input_projection(src)  # (batch_size, seq_len, model_dim)
        
        # Extract temporal features
        temporal_features = self.temporal_feature_extraction(projected_src)
        
        # Add positional encoding
        pos_encoded = self.pos_encoder(temporal_features)
        
        # Process through temporal transformer layers
        output = pos_encoded
        for layer in self.temporal_layers:
            output = layer(output, src_key_padding_mask=src_key_padding_mask)
            
        # Frame-level predictions
        frame_logits = self.frame_fc(output)
        frame_logits = frame_logits.squeeze(-1)  # (batch_size, seq_len)
        frame_probs = self.frame_sigmoid(frame_logits)
        
        # Time-to-accident predictions
        tta_preds = self.tta_regression(output)
        tta_preds = tta_preds.squeeze(-1)  # (batch_size, seq_len)
        tta_predictions = self.tta_activation(tta_preds)
        
        # Sequence-level representation and prediction
        if src_key_padding_mask is not None:
            # Masked pooling for variable length sequences
            expanded_padding_mask = (~src_key_padding_mask).unsqueeze(-1).float()  # (B, S, 1)
            masked_output = output * expanded_padding_mask
            summed_output = masked_output.sum(dim=1)  # (B, D)
            num_unmasked_elements = expanded_padding_mask.sum(dim=1)  # (B, 1)
            num_unmasked_elements = torch.clamp(num_unmasked_elements, min=1e-9)  # avoid divide by zero
            seq_rep = summed_output / num_unmasked_elements
        else:
            # Simple mean pooling for fixed length sequences
            seq_rep = output.mean(dim=1)
        
        # Final sequence-level prediction
        seq_logits = self.seq_fc(seq_rep)
        seq_logits = seq_logits.squeeze(-1)
        seq_probs = self.seq_sigmoid(seq_logits)
        
        return frame_probs, seq_probs, tta_predictions