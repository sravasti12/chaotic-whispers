"""
Main orchestrator for Chaotic Whispers:
1️⃣ Fetch stock + news
2️⃣ Add FinBERT sentiment
3️⃣ Merge features
4️⃣ Train LSTM
5️⃣ Evaluate & plot predictions
"""

import torch
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt

# 1️⃣ Import your modules
from src.data_loader import fetch_stock_data, fetch_news
from src.features import add_sentiment_to_news, merge_stock_news
from src.model import StockLSTM
from src.train import StockDataset, train_model, train_model  # make sure train_model is the fixed version
from src.evaluate import evaluate_model  # optional, if you have it

# ---------------------------
# User settings
# ---------------------------
TICKER = "AAPL"
SEQ_LEN = 5
BATCH_SIZE = 16
EPOCHS = 10
API_KEY = "YOUR_API_KEY"  # Replace with your NewsAPI key
DEVICE = torch.device("cpu")  # or "cuda" if GPU available

# ---------------------------
# Step 1: Fetch stock data
# ---------------------------
print(f"Fetching stock data for {TICKER}...")
stock_df = fetch_stock_data(TICKER, period="3mo", interval="1d")

# ---------------------------
# Step 2: Fetch news safely
# ---------------------------
print(f"Fetching news for {TICKER}...")
news_df = fetch_news(TICKER, API_KEY)

if news_df is None or news_df.empty:
    print("No news articles found. Using neutral sentiment.")
    news_sentiment = pd.DataFrame(columns=['publishedAt', 'title', 'sentiment_score'])
else:
    news_sentiment = add_sentiment_to_news(news_df)
    if news_sentiment.empty:
        print("No valid news titles found. Using neutral sentiment.")
        news_sentiment = pd.DataFrame(columns=['publishedAt', 'title', 'sentiment_score'])

# ---------------------------
# Step 3: Merge stock + sentiment
# ---------------------------
merged_df = merge_stock_news(stock_df, news_sentiment)

# ---------------------------
# Step 4: Prepare dataset & dataloader
# ---------------------------
dataset = StockDataset(merged_df, seq_len=SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---------------------------
# Step 5: Initialize & train model
# ---------------------------
input_size = 6  # Open, High, Low, Close, Volume, daily_sentiment
model = StockLSTM(input_size)
model.to(DEVICE)

print("Training LSTM model...")
train_model(model, dataloader, epochs=EPOCHS)

# Save model
torch.save(model.state_dict(), "stock_lstm.pth")
print("Model saved as stock_lstm.pth")

# ---------------------------
# Step 6: Evaluate & plot (optional)
# ---------------------------
if 'evaluate_model' in globals():
    print("Evaluating model...")
    dataset_eval = StockDataset(merged_df, seq_len=SEQ_LEN)
    dataloader_eval = DataLoader(dataset_eval, batch_size=BATCH_SIZE, shuffle=False)

    preds, actuals = evaluate_model(model, dataloader_eval)

    plt.figure(figsize=(12,6))
    plt.plot(range(len(actuals)), actuals, label="Actual Close")
    plt.plot(range(len(preds)), preds, label="Predicted Close")
    plt.xlabel("Time Steps")
    plt.ylabel("Stock Close Price")
    plt.title(f"{TICKER} Stock Price Prediction")
    plt.legend()
    plt.show()
