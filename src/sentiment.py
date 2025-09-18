from transformers import pipeline

# Load FinBERT sentiment model
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")

def get_sentiment(text):
    """
    Runs FinBERT sentiment analysis on input text.
    Returns: label (positive/negative/neutral), score
    """
    result = finbert(text)[0]
    return result["label"], result["score"]

# Quick test
if __name__ == "__main__":
    sample = "The company posted higher-than-expected profits and the stock soared."
    label, score = get_sentiment(sample)
    print(f"Text: {sample}")
    print(f"Sentiment: {label} (score: {score:.4f})")
