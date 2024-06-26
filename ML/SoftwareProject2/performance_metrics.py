import numpy as np


def accuracy(y_true, y_pred):
    """
    Calculate the accuracy of predictions.
    """
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    return correct_predictions / total_predictions


def precision(y_true, y_pred):
    """
    Calculate the precision of predictions.
    """
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positives = np.sum(y_pred == 1)
    return true_positives / predicted_positives if predicted_positives != 0 else 0


def recall(y_true, y_pred):
    """
    Calculate the recall of predictions.
    """
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    actual_positives = np.sum(y_true == 1)
    return true_positives / actual_positives if actual_positives != 0 else 0


def f1_score(y_true, y_pred):
    """
    Calculate the F1 score of predictions.
    """
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    if prec + rec == 0:
        return 0
    return 2 * (prec * rec) / (prec + rec)


def calculate_roc(y_true, y_scores):
    """
    Calculate the ROC curve points.

    Returns:
    list: FPR and TPR values at different thresholds.
    """
    tprs = []
    fprs = []

    for threshold in np.unique(y_scores):
        # Apply threshold
        y_pred = np.array(y_scores >= threshold).astype(int)

        # Calculate TPR and FPR
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        TN = np.sum((y_true == 0) & (y_pred == 0))

        TPR = TP / (TP + FN) if TP + FN != 0 else 0
        FPR = FP / (FP + TN) if FP + TN != 0 else 0

        tprs.append(TPR)
        fprs.append(FPR)

    return fprs, tprs


def auc(y_true, y_scores):
    """
    Calculate the AUC.
    """
    fprs, tprs = calculate_roc(y_true, y_scores)

    # Sort the FPRs and corresponding TPRs
    sorted_indices = np.argsort(fprs)
    sorted_fprs = np.array(fprs)[sorted_indices]
    sorted_tprs = np.array(tprs)[sorted_indices]

    # Calculate the area using the trapezoidal rule
    return np.trapz(sorted_tprs, sorted_fprs)


def calculate_precision_recall_curve(y_true, y_scores):
    """
    Calculate the precision-recall curve points.

    Returns:
    list: Recall and precision values at different thresholds.
    """
    # Sort scores and corresponding true values
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_scores_sorted = y_scores[sorted_indices]

    precisions = []
    recalls = []

    for threshold in np.unique(y_scores_sorted):
        # Apply threshold
        y_pred = (y_scores_sorted >= threshold).astype(int)

        # Calculate Precision and Recall
        TP = np.sum((y_true_sorted == 1) & (y_pred == 1))
        FP = np.sum((y_true_sorted == 0) & (y_pred == 1))
        FN = np.sum((y_true_sorted == 1) & (y_pred == 0))
        precision = TP / (TP + FP) if TP + FP != 0 else 0
        recall = TP / (TP + FN) if TP + FN != 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    return recalls, precisions


def auprc(y_true, y_scores):
    """
    Calculate the AUPRC.
    """
    recalls, precisions = calculate_precision_recall_curve(y_true, y_scores)

    # Sort the recalls and corresponding precisions
    sorted_indices = np.argsort(recalls)
    sorted_recalls = np.array(recalls)[sorted_indices]
    sorted_precisions = np.array(precisions)[sorted_indices]

    # Calculate the area using the trapezoidal rule
    return np.trapz(sorted_precisions, sorted_recalls)
