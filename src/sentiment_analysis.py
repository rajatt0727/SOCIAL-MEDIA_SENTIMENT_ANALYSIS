# src/sentiment_analysis.py
from transformers import pipeline
import pandas as pd

# Load cleaned tweets
df = pd.read_csv('data/cleaned_tweets.csv')

# Specify the sentiment-analysis model and revision
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
revision = "af0f99b"  # Specific revision of the model

# Load sentiment-analysis pipeline with specified model
sentiment_pipeline = pipeline('sentiment-analysis', model=model_name, revision=revision)

# Apply sentiment analysis
df['Sentiment'] = df['Cleaned_Content'].apply(lambda x: sentiment_pipeline(x)[0]['label'])
df.to_csv('data/sentiment_tweets.csv', index=False)
print(df[['Content', 'Cleaned_Content', 'Sentiment']].head())
