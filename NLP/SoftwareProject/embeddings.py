import json
import numpy as np
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel


def convert_embeddings_to_json(df, column_name):
    df[column_name] = df[column_name].apply(lambda x: json.dumps(x.tolist()))


# Load the preprocessed data
print('Loading the preprocessed data...')
amazon_train = pd.read_csv('amazon_product_reviews/amazon_reviews_train_preprocessed.csv')
amazon_test = pd.read_csv('amazon_product_reviews/amazon_reviews_test_preprocessed.csv')
laroseda_train_df = pd.read_csv('laroseda/laroseda_train_preprocessed.csv')
laroseda_test_df = pd.read_csv('laroseda/laroseda_test_preprocessed.csv')

# Initialize tokenizer and model
print('Initializing tokenizer and model...')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
model = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased')

# Check if CUDA is available and move model to GPU if possible
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)


# Function to get embeddings
def get_embeddings(text_list):
    embeddings = []
    for text in text_list:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Move tensor to CPU before converting to numpy
        embeddings.append(embedding)
    return np.vstack(embeddings)


# Generate embeddings for Amazon reviews
print('Generating embeddings for Amazon reviews...')
amazon_train_embeddings = get_embeddings(amazon_train['reviewText'].tolist())
amazon_test_embeddings = get_embeddings(amazon_test['reviewText'].tolist())

# Add embeddings to the DataFrame as JSON strings
amazon_train['embeddings'] = list(map(json.dumps, amazon_train_embeddings.tolist()))
amazon_test['embeddings'] = list(map(json.dumps, amazon_test_embeddings.tolist()))

# Generate embeddings for Laroseda
print('Generating embeddings for Laroseda reviews...')
laroseda_train_embeddings = get_embeddings(laroseda_train_df['content'].tolist())
laroseda_test_embeddings = get_embeddings(laroseda_test_df['content'].tolist())

# Add embeddings to the DataFrame as JSON strings
laroseda_train_df['embeddings'] = list(map(json.dumps, laroseda_train_embeddings.tolist()))
laroseda_test_df['embeddings'] = list(map(json.dumps, laroseda_test_embeddings.tolist()))

# Save the DataFrames with embeddings to CSV files
amazon_train.to_csv('amazon_product_reviews/amazon_reviews_train_with_embeddings.csv', index=False)
amazon_test.to_csv('amazon_product_reviews/amazon_reviews_test_with_embeddings.csv', index=False)
laroseda_train_df.to_csv('laroseda/laroseda_train_with_embeddings.csv', index=False)
laroseda_test_df.to_csv('laroseda/laroseda_test_with_embeddings.csv', index=False)
