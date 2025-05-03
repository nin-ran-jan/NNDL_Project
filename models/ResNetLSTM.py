import torch
import torch.nn as nn

class ResNetLSTM(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_layers=1, dropout=0.3):
        super(ResNetLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        
        self.frame_fc = nn.Linear(2 * hidden_dim, 1)
        self.frame_sigmoid = nn.Sigmoid()
        
        self.seq_fc = nn.Linear(2 * hidden_dim, 1)
        self.seq_sigmoid = nn.Sigmoid()

    def forward(self, x): 
        lstm_out, (h_n, _) = self.lstm(x)

        frame_logits = self.frame_fc(lstm_out).squeeze(-1)
        frame_probs = self.frame_sigmoid(frame_logits)

        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        seq_rep = torch.cat((h_forward, h_backward), dim=1) 

        seq_logits = self.seq_fc(seq_rep).squeeze(-1)
        seq_probs = self.seq_sigmoid(seq_logits)

        return frame_probs, seq_probs
