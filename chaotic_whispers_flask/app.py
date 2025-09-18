import sys
import os

# 🔧 Allow Flask to find the "src" folder one level up
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, render_template, request
import torch
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import io, base64

# ✅ Import your existing modules
from src.data_loader import fetch_stock_data, fetch_news
from src.features import add_sentiment_to_news, merge_stock_news
from src.model import StockLSTM
from src.train import StockDataset
from src.evaluate import evaluate_model

# ---------------------------
# Flask setup
# ---------------------------
app = Flask(__name__)
DEVICE = torch.device("cpu")   # use "cuda" if GPU available


@app.route("/", methods=["GET", "POST"])
def index():
    plot_url = None
    ticker = None

    if request.method == "POST":
        ticker = request.form.get("ticker", "AAPL").upper()

        # 1️⃣ Fetch stock data
        stock_df = fetch_stock_data(ticker, period="3mo", interval="1d")

        # 2️⃣ Fetch news
        news_df = fetch_news(ticker, "YOUR_API_KEY")  # replace with your NewsAPI key
        if news_df is None or news_df.empty:
            news_sentiment = pd.DataFrame(columns=['publishedAt', 'title', 'sentiment_score'])
        else:
            news_sentiment = add_sentiment_to_news(news_df)
            if news_sentiment.empty:
                news_sentiment = pd.DataFrame(columns=['publishedAt', 'title', 'sentiment_score'])

        # 3️⃣ Merge stock + sentiment
        merged_df = merge_stock_news(stock_df, news_sentiment)

        # 4️⃣ Load trained model
        input_size = 6  # OHLCV + sentiment
        model = StockLSTM(input_size)
        model.load_state_dict(torch.load("models/stock_lstm.pth", map_location=DEVICE))
        model.to(DEVICE)
        model.eval()

        # 5️⃣ Prepare dataset
        dataset = StockDataset(merged_df, seq_len=5)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

        # 6️⃣ Predict
        preds, actuals = evaluate_model(model, dataloader)

        # 7️⃣ Plot results
        plt.figure(figsize=(10,5))
        plt.plot(range(len(actuals)), actuals, label="Actual")
        plt.plot(range(len(preds)), preds, label="Predicted")
        plt.title(f"{ticker} Stock Prediction")
        plt.xlabel("Time Steps")
        plt.ylabel("Price")
        plt.legend()

        # Save plot to memory
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode()

    return render_template("index.html", plot_url=plot_url, ticker=ticker)


# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)

