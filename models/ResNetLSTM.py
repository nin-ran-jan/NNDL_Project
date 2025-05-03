import torch.nn as nn

class ResNetLSTM(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_layers=1, dropout=0.3):
        super(ResNetLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, 1)  
        self.sigmoid = nn.Sigmoid()         

    def forward(self, x):  
        lstm_out, _ = self.lstm(x)         
        logits = self.fc(lstm_out).squeeze(-1)  
        probs = self.sigmoid(logits)            
        return probs
