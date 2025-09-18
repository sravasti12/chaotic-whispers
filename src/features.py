import pandas as pd
from src.sentiment import get_sentiment

def add_sentiment_to_news(news_df):
    """
    Add FinBERT sentiment score to news DataFrame.
    Returns DataFrame with columns: ['publishedAt', 'title', 'sentiment_score']
    """
    import pandas as pd
    from src.sentiment import get_sentiment

    # If news_df is empty, return empty DataFrame with expected columns
    if news_df is None or news_df.empty:
        return pd.DataFrame(columns=['publishedAt', 'title', 'sentiment_score'])

    # Check if 'title' column exists
    if 'title' not in news_df.columns:
        # Try common alternatives
        if 'headline' in news_df.columns:
            news_df['title'] = news_df['headline']
        else:
            # If no title, skip
            news_df['title'] = ""

    sentiments = []
    for title in news_df['title']:
        if title.strip() == "":
            label, score = "neutral", 0.0
        else:
            label, score = get_sentiment(title)
        sentiments.append(score)

    news_df['sentiment_score'] = sentiments
    return news_df[['publishedAt', 'title', 'sentiment_score']]


def merge_stock_news(stock_df, news_df):
    """
    Merge stock OHLCV data with news sentiment features.
    """
    # Ensure 'Date' is a column
    if 'Date' not in stock_df.columns:
        stock_df = stock_df.reset_index()

    # Convert to date only
    stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.date

    if news_df is None or news_df.empty:
        stock_df['daily_sentiment'] = 0.0
        return stock_df

    news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt']).dt.date

    # Aggregate sentiment per day
    daily_sentiment = news_df.groupby('publishedAt')['sentiment_score'].mean().reset_index()
    daily_sentiment.rename(columns={'publishedAt': 'Date', 'sentiment_score': 'daily_sentiment'}, inplace=True)

    merged_df = pd.merge(stock_df, daily_sentiment, on='Date', how='left')
    merged_df['daily_sentiment'] = merged_df['daily_sentiment'].fillna(0.0)
    return merged_df



# Quick test
if __name__ == "__main__":
    # Example stock data
    stock_df = pd.DataFrame({
        "Date": pd.to_datetime(["2025-09-18", "2025-09-19"]),
        "Open": [150, 152],
        "High": [155, 153],
        "Low": [149, 151],
        "Close": [154, 152],
        "Volume": [1000000, 1200000]
    })

    # Example news data
    news_df = pd.DataFrame({
        "publishedAt": ["2025-09-18", "2025-09-18"],
        "title": ["Company reports record profits", "Stock market reacts positively"],
        "description": ["Details about earnings", "Investors optimistic"]
    })

    news_sentiment = add_sentiment_to_news(news_df)
    merged = merge_stock_news(stock_df, news_sentiment)
    print(merged)
