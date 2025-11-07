# ==========================================================
#  FINAL PROJECT – DATA MINING
#  Dataset: Breast Cancer Wisconsin (Diagnostic)
#  Source: UCI Machine Learning Repository
#  Positive class = Malignant
#  Negative class = Benign
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, auc, brier_score_loss

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# -------------------------------------------------------------
# Load Dataset + Inform User
# -------------------------------------------------------------
print("==========================================================")
print("USING DATASET: Breast Cancer Wisconsin (Diagnostic) – UCI")
print("Loaded via sklearn.datasets.load_breast_cancer()")
print("----------------------------------------------------------")
print("Original labels from sklearn:")
print("  0 = malignant (cancer)")
print("  1 = benign (non-cancer)")
print("Re-mapping so POSITIVE CLASS = 1 = malignant")
print("----------------------------------------------------------")

data = load_breast_cancer()
X = data.data
y_orig = data.target

# Map malignant → 1 (positive)
y = (y_orig == 0).astype(int)

print(f"Dataset shape: {X.shape[0]} samples × {X.shape[1]} features")
print("==========================================================\n")

# -------------------------------------------------------------
# Metrics Helpers
# -------------------------------------------------------------
def safe_div(a, b):
    return a / b if b != 0 else np.nan

def compute_counts(y_true, y_pred):
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    P = TP + FN
    N = TN + FP
    return TP, TN, FP, FN, P, N

def compute_skill_scores(TP, TN, FP, FN):
    P = TP + FN
    N = TN + FP
    TPR = safe_div(TP, P)
    FPR = safe_div(FP, N)
    TSS = TPR - FPR

    denom = (TP + FN)*(FN + TN) + (TP + FP)*(FP + TN)
    HSS = safe_div(2*(TP*TN - FP*FN), denom)
    return TSS, HSS

def compute_fold_metrics(y_true, y_pred, y_prob):
    TP, TN, FP, FN, P, N = compute_counts(y_true, y_pred)

    TPR = safe_div(TP, P)
    TNR = safe_div(TN, N)
    FPR = safe_div(FP, N)
    FNR = safe_div(FN, P)

    accuracy = safe_div(TP + TN, P + N)
    balanced_accuracy = np.nanmean([TPR, TNR])

    precision = safe_div(TP, TP + FP)
    recall = TPR
    f1 = safe_div(2 * precision * recall, precision + recall)
    error_rate = 1 - accuracy

    TSS, HSS = compute_skill_scores(TP, TN, FP, FN)

    bs = brier_score_loss(y_true, y_prob, pos_label=1)

    p_bar = np.mean(y_true)
    bs_ref = p_bar * (1 - p_bar)
    bss = 1 - safe_div(bs, bs_ref)

    return {
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "P": P, "N": N,
        "TPR": TPR, "TNR": TNR, "FPR": FPR, "FNR": FNR,
        "Accuracy": accuracy,
        "BalancedAccuracy": balanced_accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ErrorRate": error_rate,
        "TSS": TSS,
        "HSS": HSS,
        "BS": bs,
        "BSS": bss
    }

# -------------------------------------------------------------
# Models
# -------------------------------------------------------------
models = {
    "LogisticRegression": Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000))
    ]),
    "SVM_RBF": Pipeline([
        ("scale", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True))
    ]),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
    "MLP": Pipeline([
        ("scale", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(64,32), max_iter=2000, random_state=42))
    ]),
}

# -------------------------------------------------------------
# 10-Fold Cross Validation
# -------------------------------------------------------------
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
results_all_models = {}
roc_curves = {}

print("Starting 10-fold Stratified Cross-Validation...\n")

for model_name, model in models.items():
    print(f"Training model: {model_name}")
    fold_results = []
    y_true_all = []
    y_prob_all = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred.astype(float)

        metrics = compute_fold_metrics(y_test, y_pred, y_prob)
        metrics["Fold"] = fold
        fold_results.append(metrics)

        y_true_all.append(y_test)
        y_prob_all.append(y_prob)

    df = pd.DataFrame(fold_results).sort_values("Fold")

    avg = df.mean(numeric_only=True)
    avg["Fold"] = "Average"
    df = pd.concat([df, avg.to_frame().T], ignore_index=True)

    results_all_models[model_name] = df

    y_true_all = np.concatenate(y_true_all)
    y_prob_all = np.concatenate(y_prob_all)

    fpr, tpr, _ = roc_curve(y_true_all, y_prob_all)
    model_auc = auc(fpr, tpr)
    roc_curves[model_name] = (fpr, tpr, model_auc)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {model_auc:.4f}")
    plt.plot([0,1], [0,1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.show()

print("\nAll models completed.\n")

# -------------------------------------------------------------
# Summary table
# -------------------------------------------------------------
summary_rows = []

for model_name, df in results_all_models.items():
    avg = df[df["Fold"] == "Average"].iloc[0].copy()
    avg["Model"] = model_name
    summary_rows.append(avg)

summary_df = pd.DataFrame(summary_rows)
print("===== SUMMARY OF MODEL AVERAGES =====")
print(summary_df[[
    "Model", "Accuracy", "BalancedAccuracy", "Precision",
    "Recall", "F1", "ErrorRate", "TSS", "HSS", "BS", "BSS"
]])
