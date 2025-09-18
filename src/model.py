import torch
import torch.nn as nn

class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM output
        out, _ = self.lstm(x, (h0, c0))  # out: (batch, seq_len, hidden_size)
        
        # Take only last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# Quick test
if __name__ == "__main__":
    # Example: batch_size=2, seq_len=5 days, features=6 (Open, High, Low, Close, Volume, daily_sentiment)
    batch_size = 2
    seq_len = 5
    input_size = 6
    x = torch.randn(batch_size, seq_len, input_size)
    
    model = StockLSTM(input_size)
    y_pred = model(x)
    print("Output shape:", y_pred.shape)  # Should be (batch_size, 1)
