import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from src.model import StockLSTM
from src.features import merge_stock_news, add_sentiment_to_news
from src.data_loader import fetch_stock_data, fetch_news

# ---------------------------
# Dataset class (same as train.py)
# ---------------------------
class StockDatasetEval(Dataset):
    def __init__(self, data, seq_len=5):
        self.seq_len = seq_len
        self.data = data[['Open','High','Low','Close','Volume','daily_sentiment']].values
        self.targets = data['Close'].values
        self.n_samples = len(data) - seq_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.targets[idx+self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# ---------------------------
# Evaluation function
# ---------------------------
def evaluate_model(model, dataloader):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            y_pred = model(x_batch)
            predictions.extend(y_pred.squeeze().tolist())
            actuals.extend(y_batch.tolist())
    
    return predictions, actuals

# ---------------------------
# Quick pipeline test
# ---------------------------
if __name__ == "__main__":
    # 1️⃣ Fetch stock data
    ticker = "AAPL"
    stock_df = fetch_stock_data(ticker, period="3mo", interval="1d")
    
    # 2️⃣ Fetch news (replace YOUR_API_KEY)
    api_key = "YOUR_API_KEY"
    news_df = fetch_news(ticker, api_key)
    
    # 3️⃣ Add sentiment
    news_sentiment = add_sentiment_to_news(news_df)
    
    # 4️⃣ Merge features
    merged_df = merge_stock_news(stock_df, news_sentiment)
    
    # 5️⃣ Prepare dataset & dataloader
    seq_len = 5
    dataset = StockDatasetEval(merged_df, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # 6️⃣ Load model
    input_size = 6
    model = StockLSTM(input_size)
    model.load_state_dict(torch.load("stock_lstm.pth"))
    
    # 7️⃣ Evaluate
    preds, actuals = evaluate_model(model, dataloader)
    
    # 8️⃣ Plot results
    plt.figure(figsize=(12,6))
    plt.plot(range(len(actuals)), actuals, label="Actual Close")
    plt.plot(range(len(preds)), preds, label="Predicted Close")
    plt.xlabel("Time Steps")
    plt.ylabel("Stock Close Price")
    plt.title(f"{ticker} Stock Price Prediction")
    plt.legend()
    plt.show()
