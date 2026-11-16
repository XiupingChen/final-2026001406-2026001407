import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score)


def validation_threshold(y, probability):
    """Max validation F1; ties choose the largest threshold. Never uses test labels."""
    precision, recall, thresholds = precision_recall_curve(y, probability)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2*precision[:-1]*recall[:-1] / np.maximum(precision[:-1]+recall[:-1], 1e-15)
    return float(thresholds[np.flatnonzero(f1 >= f1.max()-1e-14)[-1]])


def evaluate(y, probability, threshold):
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Invalid predicted probability")
    selected = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y, selected, labels=[0, 1]).ravel()
    return {"roc_auc": float(roc_auc_score(y, probability)),
            "pr_auc_ap": float(average_precision_score(y, probability)),
            "recall_05": float(recall_score(y, probability >= 0.5, zero_division=0)),
            "recall_valid_f1": float(recall_score(y, selected, zero_division=0)),
            "precision_valid_f1": float(precision_score(y, selected, zero_division=0)),
            "f1_test_at_valid_threshold": float(f1_score(y, selected, zero_division=0)),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "brier": float(brier_score_loss(y, probability)), "threshold": threshold}
