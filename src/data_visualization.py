# src/data_visualization.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load sentiment data
df = pd.read_csv('data/sentiment_tweets.csv')

# Plot sentiment distribution
sns.countplot(x='Sentiment', data=df)
plt.title('Sentiment Distribution of Social Media Posts')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Plot sentiment over time
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
sentiment_time_series = df.resample('D').Sentiment.value_counts().unstack().fillna(0)
sentiment_time_series.plot(kind='line', figsize=(14, 7))
plt.title('Sentiment Over Time')
plt.xlabel('Date')
plt.ylabel('Count')
plt.show()
