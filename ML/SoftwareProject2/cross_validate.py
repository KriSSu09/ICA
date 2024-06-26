import numpy as np
from performance_metrics import accuracy, precision, recall, f1_score, auc, auprc


def cross_validate(model, X, y, cv=5):
    if cv > len(y):
        raise ValueError("Number of folds cannot be greater than the number of instances.")

    indices = np.arange(len(y))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    fold_size = len(y) // cv
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'auc': [], 'auprc': []}

    for i in range(cv):
        test_indices = slice(i * fold_size, (i + 1) * fold_size)

        X_train = np.concatenate([X[:i * fold_size], X[(i + 1) * fold_size:]])
        y_train = np.concatenate([y[:i * fold_size], y[(i + 1) * fold_size:]])
        X_test = X[test_indices]
        y_test = convert_labels_to_binary(y[test_indices])

        model.fit(X_train, y_train)
        y_pred = convert_labels_to_binary(model.predict(X_test))
        y_scores = model.predict_proba(X_test)

        # Calculate and record performance metrics
        metrics['accuracy'].append(accuracy(y_test, y_pred))
        metrics['precision'].append(precision(y_test, y_pred))
        metrics['recall'].append(recall(y_test, y_pred))
        metrics['f1_score'].append(f1_score(y_test, y_pred))
        metrics['auc'].append(auc(y_test, y_scores[:, 1]))
        metrics['auprc'].append(auprc(y_test, y_scores[:, 1]))

    return metrics


def convert_labels_to_binary(y):
    """
    Convert labels from 'Yes'/'No' to 1/0.
    """
    return np.array([1 if label == 'Yes' else 0 for label in y])
