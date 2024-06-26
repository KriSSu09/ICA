import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json


def convert_json_to_embeddings(df, column_name):
    df[column_name] = df[column_name].apply(lambda x: np.array(json.loads(x)))


# Load the embedding data
print('Loading the embedding data...')
laroseda_train_df = pd.read_csv('laroseda/laroseda_train_with_embeddings.csv')
laroseda_test_df = pd.read_csv('laroseda/laroseda_test_with_embeddings.csv')
amazon_train = pd.read_csv('amazon_product_reviews/amazon_reviews_train_with_embeddings.csv')
amazon_test = pd.read_csv('amazon_product_reviews/amazon_reviews_test_with_embeddings.csv')

# Convert JSON strings back to numpy arrays
convert_json_to_embeddings(laroseda_train_df, 'embeddings')
convert_json_to_embeddings(laroseda_test_df, 'embeddings')
convert_json_to_embeddings(amazon_train, 'embeddings')
convert_json_to_embeddings(amazon_test, 'embeddings')

# Train a classifier on Laroseda
print('Training laroseda classifier...')
laroseda_clf = LogisticRegression(max_iter=1000)
laroseda_clf.fit(np.vstack(laroseda_train_df['embeddings']), laroseda_train_df['starRating'].astype(int))

# Train a classifier on Amazon
print('Training amazon classifier...')
amazon_clf = LogisticRegression(max_iter=1000)
amazon_clf.fit(np.vstack(amazon_train['embeddings']), amazon_train['overall'].astype(int))

# Combine the training data
combined_train_embeddings = np.vstack(
    (np.vstack(laroseda_train_df['embeddings']), np.vstack(amazon_train['embeddings'])))
combined_train_labels = np.concatenate(
    (laroseda_train_df['starRating'].astype(int), amazon_train['overall'].astype(int)))

# Train a classifier on the combined dataset
print('Training combined classifier...')
combined_clf = LogisticRegression(max_iter=1000)
combined_clf.fit(combined_train_embeddings, combined_train_labels)


# Test classifiers on both datasets
def evaluate_classifier(clf, embeddings, labels):
    predictions = clf.predict(embeddings)
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    return accuracy, precision, recall, f1


# Evaluate on Amazon test set
print('Evaluating amazon classifier...')
amazon_test_labels = amazon_test['overall'].astype(int)
amazon_on_amazon_results = evaluate_classifier(amazon_clf, np.vstack(amazon_test['embeddings']), amazon_test_labels)
laroseda_on_amazon_results = evaluate_classifier(laroseda_clf, np.vstack(amazon_test['embeddings']), amazon_test_labels)
combined_on_amazon_results = evaluate_classifier(combined_clf, np.vstack(amazon_test['embeddings']), amazon_test_labels)

# Evaluate on Laroseda test set
print('Evaluating laroseda classifier...')
laroseda_test_labels = laroseda_test_df['starRating'].astype(int)
amazon_on_laroseda_results = evaluate_classifier(amazon_clf, np.vstack(laroseda_test_df['embeddings']),
                                                 laroseda_test_labels)
laroseda_on_laroseda_results = evaluate_classifier(laroseda_clf, np.vstack(laroseda_test_df['embeddings']),
                                                   laroseda_test_labels)
combined_on_laroseda_results = evaluate_classifier(combined_clf, np.vstack(laroseda_test_df['embeddings']),
                                                   laroseda_test_labels)


# Print the results
def print_results(dataset_name, results):
    accuracy, precision, recall, f1 = results
    print(f"{dataset_name} results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print()


print_results("Amazon classifier on Amazon test set", amazon_on_amazon_results)
print_results("Laroseda classifier on Amazon test set", laroseda_on_amazon_results)
print_results("Combined classifier on Amazon test set", combined_on_amazon_results)
print_results("Amazon classifier on Laroseda test set", amazon_on_laroseda_results)
print_results("Laroseda classifier on Laroseda test set", laroseda_on_laroseda_results)
print_results("Combined classifier on Laroseda test set", combined_on_laroseda_results)
