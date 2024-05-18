# src/model_refinement.py
import logging
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import torch
import pandas as pd

# Configure logging
logging.basicConfig(filename='logs/model_training.log', level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

# Load the dataset
df = pd.read_csv('data/sentiment_tweets.csv')
dataset = Dataset.from_pandas(df[['Cleaned_Content', 'Sentiment']])

# Check unique values in the "Sentiment" column
unique_sentiments = df['Sentiment'].unique()
print(f"Unique sentiment values: {unique_sentiments}")
logger.info(f"Unique sentiment values: {unique_sentiments}")

# Prepare the dataset
def preprocess_function(examples):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer(examples["Cleaned_Content"], truncation=True, padding=True)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
encoded_dataset = dataset.map(preprocess_function, batched=True)

# Define label mapping
label_mapping = {"POSITIVE": 1, "NEGATIVE": 0}

# Ensure all sentiment values are in the label mapping
def map_labels(examples):
    try:
        return {"label": [label_mapping[label] for label in examples["Sentiment"]]}
    except KeyError as e:
        logger.error(f"KeyError: {e} in examples {examples}")
        raise

encoded_dataset = encoded_dataset.map(map_labels, batched=True)

# Load model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# Training arguments
training_args = TrainingArguments(
    output_dir='results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir='logs',
    logging_steps=10,
    save_steps=10,
    evaluation_strategy="steps",
    save_total_limit=2
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset,
    eval_dataset=encoded_dataset
)

# Train the model
logger.info("Starting model training")
trainer.train()
logger.info("Model training completed")

# Save the model
model.save_pretrained('results/refined_model')
tokenizer.save_pretrained('results/refined_model')
logger.info("Model saved to results/refined_model")

# Save training logs
logs = trainer.state.log_history
pd.DataFrame(logs).to_csv('logs/training_logs.csv', index=False)
logger.info("Training logs saved to logs/training_logs.csv")
