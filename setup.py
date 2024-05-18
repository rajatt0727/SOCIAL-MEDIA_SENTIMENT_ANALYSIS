# setup.py
import nltk
import seaborn as sns
import matplotlib.pyplot as plt

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Install necessary libraries
import os
os.system('pip install pandas')
os.system('pip install numpy')
os.system('pip install scikit-learn')
os.system('pip install transformers')
os.system('pip install torch')
os.system('pip install datasets')  # Add this line to install the datasets library
os.system('pip install faker')

print("All libraries are successfully installed and initialized.")
