import torch
import torch.nn as nn
from torch.utils.data import Dataset

# ---------------------------
# Dataset class
# ---------------------------
class StockDataset(Dataset):
    def __init__(self, data, seq_len=5):
        """
        data: pd.DataFrame with columns ['Open','High','Low','Close','Volume','daily_sentiment']
        seq_len: number of days in input sequence
        """
        self.seq_len = seq_len
        self.data = data[['Open','High','Low','Close','Volume','daily_sentiment']].values
        self.targets = data['Close'].values  # predicting next day Close
        self.n_samples = len(data) - seq_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.targets[idx+self.seq_len]  # next day close
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# ---------------------------
# Training function
# ---------------------------
def train_model(model, dataloader, epochs=20, lr=0.001, device="cpu"):
    """
    Train LSTM model on stock data.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(1, epochs+1):
        epoch_loss = 0
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.view(-1,1).to(device)
            
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss/len(dataloader):.4f}")
