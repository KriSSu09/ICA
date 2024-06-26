import json
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the datasets
amazon_reviews = pd.read_csv('amazon_product_reviews/amazon_reviews.csv')
with open('laroseda/laroseda_train.json', 'r') as f:
    laroseda_train = json.load(f)
with open('laroseda/laroseda_test.json', 'r') as f:
    laroseda_test = json.load(f)

# Create DataFrames for Laroseda
print('Creating DataFrames for Laroseda...')
laroseda_train_df = pd.DataFrame(laroseda_train['reviews'])
laroseda_test_df = pd.DataFrame(laroseda_test['reviews'])

# Preprocess the data
print('Preprocessing the data...')
amazon_reviews = amazon_reviews[['reviewText', 'overall']].dropna()
laroseda_train_df = laroseda_train_df[['content', 'starRating']].dropna()
laroseda_test_df = laroseda_test_df[['content', 'starRating']].dropna()

# Convert ratings to int
amazon_reviews['overall'] = amazon_reviews['overall'].astype(int)
laroseda_train_df['starRating'] = laroseda_train_df['starRating'].astype(int)
laroseda_test_df['starRating'] = laroseda_test_df['starRating'].astype(int)

# Split the Amazon dataset into training and test sets
amazon_train, amazon_test = train_test_split(amazon_reviews, test_size=0.5, random_state=42)

# Save the preprocessed data
print('Saving the preprocessed data...')
amazon_train.to_csv('amazon_product_reviews/amazon_reviews_train_preprocessed.csv', index=False)
amazon_test.to_csv('amazon_product_reviews/amazon_reviews_test_preprocessed.csv', index=False)
laroseda_train_df.to_csv('laroseda/laroseda_train_preprocessed.csv', index=False)
laroseda_test_df.to_csv('laroseda/laroseda_test_preprocessed.csv', index=False)
