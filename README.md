# Chaotic Whispers 🚀

**Stock Price Prediction using LSTM + FinBERT Sentiment Analysis**  
A full-stack Python project combining historical stock data and financial news sentiment to predict short-term stock trends. Includes a Flask web interface for real-time predictions.

---

## Features
- Predict stock price trends using **LSTM** on OHLCV (Open, High, Low, Close, Volume) data.
- Analyze financial news sentiment with **FinBERT** and incorporate it into predictions.
- Full **pipeline**: fetch stock + news data → preprocess → merge features → train/evaluate model.
- Simple **Flask web interface**: input a stock ticker → get predicted prices and sentiment insights.

---

## Folder Structure
chaotic-whispers/
│
├── src/
│ ├── data_loader.py # Fetch stock data & news
│ ├── features.py # Merge stock + sentiment features
│ ├── model.py # LSTM model definition
│ ├── sentiment.py # FinBERT sentiment pipeline
│ ├── dataset.py # Dataset & DataLoader
│ ├── train.py # Training script
│ └── evaluate.py # Evaluation script
│
├── chaotic_whispers_flask/ # Flask app
│ ├── templates/
│ └── static/
│
├── models/ # Trained model weights (.pth)
├── notebooks/ # Optional: Jupyter notebooks
├── requirements.txt
├── README.md
└── .gitignore

