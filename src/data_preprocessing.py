# src/data_preprocessing.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    words = word_tokenize(text)
    words = [word for word in words if word.lower() not in stopwords.words('english')]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Load tweets
df = pd.read_csv('data/synthetic_tweets.csv')
df['Cleaned_Content'] = df['Content'].apply(preprocess_text)
df.to_csv('data/cleaned_tweets.csv', index=False)
print(df[['Content', 'Cleaned_Content']].head())
