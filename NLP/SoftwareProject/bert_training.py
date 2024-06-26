import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

from review_dataset import ReviewDataset

# Load the preprocessed datasets
print('Loading the preprocessed data...')
amazon_train = pd.read_csv('amazon_product_reviews/amazon_reviews_train_preprocessed.csv')
laroseda_train_df = pd.read_csv('laroseda/laroseda_train_preprocessed.csv')

# Initialize tokenizer
distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')

# Tokenize and create datasets
MAX_LEN = 128
BATCH_SIZE = 64

amazon_train_dataset = ReviewDataset(
    reviews=amazon_train.reviewText.to_numpy(),
    labels=amazon_train.overall.to_numpy(),
    tokenizer=distilbert_tokenizer,
    max_len=MAX_LEN
)

laroseda_train_dataset = ReviewDataset(
    reviews=laroseda_train_df.content.to_numpy(),
    labels=laroseda_train_df.starRating.to_numpy(),
    tokenizer=distilbert_tokenizer,
    max_len=MAX_LEN
)

combined_train_dataset = ReviewDataset(
    reviews=pd.concat([amazon_train.reviewText, laroseda_train_df.content]).to_numpy(),
    labels=pd.concat([amazon_train.overall, laroseda_train_df.starRating]).to_numpy(),
    tokenizer=distilbert_tokenizer,
    max_len=MAX_LEN
)

amazon_train_loader = DataLoader(amazon_train_dataset, sampler=RandomSampler(amazon_train_dataset),
                                 batch_size=BATCH_SIZE)

laroseda_train_loader = DataLoader(laroseda_train_dataset, sampler=RandomSampler(laroseda_train_dataset),
                                   batch_size=BATCH_SIZE)

combined_train_loader = DataLoader(combined_train_dataset, sampler=RandomSampler(combined_train_dataset),
                                   batch_size=BATCH_SIZE)


# Function to train the model
def train_model(model, data_loader, epochs=3):
    global device
    optimizer = AdamW(model.parameters(), lr=2e-5)

    model = model.train()
    for epoch in range(epochs):
        for d in data_loader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            labels = d['labels'].to(device)

            model.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()


# Load pretrained DistilBERT models
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print('Loading pretrained models...')
amazon_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-multilingual-cased', num_labels=5)
laroseda_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-multilingual-cased', num_labels=5)
combined_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-multilingual-cased', num_labels=5)

# Move models to device
amazon_model = amazon_model.to(device)
laroseda_model = laroseda_model.to(device)
combined_model = combined_model.to(device)

# Train models
print('Training Amazon model...')
train_model(amazon_model, amazon_train_loader)

print('Training Laroseda model...')
train_model(laroseda_model, laroseda_train_loader)

print('Training Combined model...')
train_model(combined_model, combined_train_loader)

# Save the trained models to disk
print("Saving trained models to disk...")
amazon_model.save_pretrained('models/amazon_model')
laroseda_model.save_pretrained('models/laroseda_model')
combined_model.save_pretrained('models/combined_model')
