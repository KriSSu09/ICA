import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, SequentialSampler
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

from review_dataset import ReviewDataset

# Load the preprocessed datasets
print('Loading the preprocessed data...')
amazon_test = pd.read_csv('amazon_product_reviews/amazon_reviews_test_preprocessed.csv')
laroseda_test_df = pd.read_csv('laroseda/laroseda_test_preprocessed.csv')

# Initialize tokenizer
distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')

# Tokenize and create datasets
MAX_LEN = 128
BATCH_SIZE = 64

amazon_test_dataset = ReviewDataset(
    reviews=amazon_test.reviewText.to_numpy(),
    labels=amazon_test.overall.to_numpy(),
    tokenizer=distilbert_tokenizer,
    max_len=MAX_LEN
)

laroseda_test_dataset = ReviewDataset(
    reviews=laroseda_test_df.content.to_numpy(),
    labels=laroseda_test_df.starRating.to_numpy(),
    tokenizer=distilbert_tokenizer,
    max_len=MAX_LEN
)

amazon_test_loader = DataLoader(amazon_test_dataset, sampler=SequentialSampler(amazon_test_dataset),
                                batch_size=BATCH_SIZE)

laroseda_test_loader = DataLoader(laroseda_test_dataset, sampler=SequentialSampler(laroseda_test_dataset),
                                  batch_size=BATCH_SIZE)

print('Loading pretrained models...')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
amazon_model = DistilBertForSequenceClassification.from_pretrained('models/amazon_model', num_labels=5)
laroseda_model = DistilBertForSequenceClassification.from_pretrained('models/laroseda_model', num_labels=5)
combined_model = DistilBertForSequenceClassification.from_pretrained('models/combined_model', num_labels=5)
untrained_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-multilingual-cased',
                                                                      num_labels=5)

# Move models to device
amazon_model = amazon_model.to(device)
laroseda_model = laroseda_model.to(device)
combined_model = combined_model.to(device)
untrained_model = untrained_model.to(device)


# Inference function
def predict(model, data_loader):
    model = model.eval()
    reviews = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            labels = d['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)

            reviews.extend(d['review_text'])
            predictions.extend(preds)
            prediction_probs.extend(logits)
            real_values.extend(labels)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return reviews, predictions, prediction_probs, real_values


# Perform inference
print("Predicting on Amazon Reviews with Amazon model...")
amazon_reviews_amazon_model, amazon_preds_amazon_model, amazon_probs_amazon_model, amazon_labels_amazon_model = predict(
    amazon_model, amazon_test_loader)
print("Predicting on Amazon Reviews with Laroseda model...")
amazon_reviews_laroseda_model, amazon_preds_laroseda_model, amazon_probs_laroseda_model, amazon_labels_laroseda_model = predict(
    laroseda_model, amazon_test_loader)
print("Predicting on Amazon Reviews with Combined model...")
amazon_reviews_combined_model, amazon_preds_combined_model, amazon_probs_combined_model, amazon_labels_combined_model = predict(
    combined_model, amazon_test_loader)
print("Predicting on Amazon Reviews with Untrained model...")
amazon_reviews_untrained_model, amazon_preds_untrained_model, amazon_probs_untrained_model, amazon_labels_untrained_model = predict(
    untrained_model, amazon_test_loader)

print("Predicting on Laroseda Reviews with Amazon model...")
laroseda_reviews_amazon_model, laroseda_preds_amazon_model, laroseda_probs_amazon_model, laroseda_labels_amazon_model = predict(
    amazon_model, laroseda_test_loader)
print("Predicting on Laroseda Reviews with Laroseda model...")
laroseda_reviews_laroseda_model, laroseda_preds_laroseda_model, laroseda_probs_laroseda_model, laroseda_labels_laroseda_model = predict(
    laroseda_model, laroseda_test_loader)
print("Predicting on Laroseda Reviews with Combined model...")
laroseda_reviews_combined_model, laroseda_preds_combined_model, laroseda_probs_combined_model, laroseda_labels_combined_model = predict(
    combined_model, laroseda_test_loader)
print("Predicting on Laroseda Reviews with Untrained model...")
laroseda_reviews_untrained_model, laroseda_preds_untrained_model, laroseda_probs_untrained_model, laroseda_labels_untrained_model = predict(
    untrained_model, laroseda_test_loader)


# Evaluate predictions
def evaluate(predictions, labels):
    predictions = predictions.numpy()
    labels = labels.numpy()
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    return accuracy, precision, recall, f1


def print_results(dataset_name, model_name, results):
    accuracy, precision, recall, f1 = results
    print(f"{dataset_name} with {model_name} results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print()


# Evaluate and print results for Amazon reviews
amazon_results_amazon_model = evaluate(amazon_preds_amazon_model, amazon_labels_amazon_model)
print_results("Amazon Reviews", "Amazon model", amazon_results_amazon_model)

amazon_results_laroseda_model = evaluate(amazon_preds_laroseda_model, amazon_labels_laroseda_model)
print_results("Amazon Reviews", "Laroseda model", amazon_results_laroseda_model)

amazon_results_combined_model = evaluate(amazon_preds_combined_model, amazon_labels_combined_model)
print_results("Amazon Reviews", "Combined model", amazon_results_combined_model)

amazon_results_untrained_model = evaluate(amazon_preds_untrained_model, amazon_labels_untrained_model)
print_results("Amazon Reviews", "Untrained model", amazon_results_untrained_model)

# Evaluate and print results for Laroseda reviews
laroseda_results_amazon_model = evaluate(laroseda_preds_amazon_model, laroseda_labels_amazon_model)
print_results("Laroseda Reviews", "Amazon model", laroseda_results_amazon_model)

laroseda_results_laroseda_model = evaluate(laroseda_preds_laroseda_model, laroseda_labels_laroseda_model)
print_results("Laroseda Reviews", "Laroseda model", laroseda_results_laroseda_model)

laroseda_results_combined_model = evaluate(laroseda_preds_combined_model, laroseda_labels_combined_model)
print_results("Laroseda Reviews", "Combined model", laroseda_results_combined_model)

laroseda_results_untrained_model = evaluate(laroseda_preds_untrained_model, laroseda_labels_untrained_model)
print_results("Laroseda Reviews", "Untrained model", laroseda_results_untrained_model)
