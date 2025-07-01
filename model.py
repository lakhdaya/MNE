import torch
import torch.nn as nn

class EEG_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(EEG_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)  # out: [batch, time, hidden]
        out = out[:, -1, :]    # take last time step
        out = self.classifier(out)
        return out