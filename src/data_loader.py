import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta

def fetch_stock_data(ticker, period="6mo", interval="1d"):
    """
    Fetch OHLCV stock data using yfinance.
    """
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
    
    # Reset index to make 'Date' a column
    df = df.reset_index()
    
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[1]=='' else col[1] for col in df.columns]

    # Ensure consistent column names
    expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in expected_cols:
        if col not in df.columns:
            # If missing, create column with zeros (won't break pipeline)
            df[col] = 0.0

    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Keep only expected columns + Date
    df = df[['Date'] + expected_cols]
    
    return df



def fetch_news(ticker, api_key, from_date=None, to_date=None, page_size=50):
    """
    Fetch news headlines for a stock using NewsAPI.

    Args:
        ticker (str): Stock symbol, e.g., "AAPL"
        api_key (str): NewsAPI key
        from_date (str): YYYY-MM-DD
        to_date (str): YYYY-MM-DD
        page_size (int): Number of articles

    Returns:
        pd.DataFrame: Columns=['publishedAt', 'title', 'description']
    """
    if from_date is None:
        from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    if to_date is None:
        to_date = datetime.now().strftime('%Y-%m-%d')

    url = (
        f"https://newsapi.org/v2/everything?q={ticker}&from={from_date}&to={to_date}"
        f"&language=en&sortBy=publishedAt&pageSize={page_size}&apiKey={api_key}"
    )
    response = requests.get(url).json()
    articles = response.get("articles", [])
    
    news_data = []
    for a in articles:
        news_data.append({
            "publishedAt": a["publishedAt"],
            "title": a["title"],
            "description": a["description"]
        })
    return pd.DataFrame(news_data)

# Quick test
if __name__ == "__main__":
    ticker = "AAPL"
    stock_df = fetch_stock_data(ticker)
    print("Stock data:")
    print(stock_df.head())

    # Replace YOUR_API_KEY with your actual NewsAPI key
    api_key = "79525e30fd7840c89b800b2507040660"
    news_df = fetch_news(ticker, api_key)
    print("\nNews headlines:")
    print(news_df.head())
