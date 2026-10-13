from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix

def compute_metrics(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "quadratic_kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")) if len(labels) > 1 else 1.0,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }
