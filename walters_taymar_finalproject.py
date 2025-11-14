# ==========================================================
#  FINAL PROJECT – DATA MINING
#  Dataset: Breast Cancer Wisconsin (Diagnostic)
#  Source: UCI Machine Learning Repository
#
#  Positive class (1) = Malignant
#  Negative class (0) = Benign
#
#  This script:
#    • Runs 10-fold Stratified CV on 4 models
#    • Computes all metrics (TP/TN/FP/FN + TSS/HSS + BS/BSS)
#    • Saves results EXACTLY where this script lives
#    • Saves per-fold CSVs + summary CSV
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

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
#  ALWAYS SAVE FILES IN THE SAME FOLDER AS THIS SCRIPT
# -------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))



# -------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------
def safe_div(a, b):
    return a / b if b != 0 else np.nan


def compute_counts(y_true, y_pred):
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    P = TP + FN
    N = TN + FP
    return TP, TN, FP, FN, P, N


def compute_skill_scores(TP, TN, FP, FN):
    P = TP + FN
    N = TN + FP

    TPR = safe_div(TP, P)
    FPR = safe_div(FP, N)
    TSS = TPR - FPR

    denom = (TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)
    HSS = safe_div(2 * (TP * TN - FP * FN), denom)

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
# Main Experiment Runner
# -------------------------------------------------------------
def run_experiment(SAVE_ROC=False, SAVE_FOLDER="results_dm_final"):

    # Always save results inside the script's directory
    SAVE_FOLDER = os.path.join(SCRIPT_DIR, SAVE_FOLDER)
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    print(f"\nSaving all output to: {SAVE_FOLDER}\n")

    # -----------------------------------------
    # Load dataset
    # -----------------------------------------
    print("==========================================================")
    print("USING DATASET: Breast Cancer Wisconsin (Diagnostic)")
    print("----------------------------------------------------------")

    data = load_breast_cancer()
    X = data.data
    y_orig = data.target

    # Remap sklearn labels → project labels
    y = (y_orig == 0).astype(int)

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print("Positive class = malignant = 1")
    print("Negative class = benign = 0")
    print("==========================================================\n")

    # -----------------------------------------
    # Define models
    # -----------------------------------------
    models = {
        "LogisticRegression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000))
        ]),
        "SVM_RBF": Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(kernel='rbf', probability=True))
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=42
        ),
        "MLP": Pipeline([
            ("scale", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=2000,
                random_state=42
            ))
        ])
    }

    # -----------------------------------------
    # 10-fold CV
    # -----------------------------------------
    print("Starting 10-fold Stratified Cross-Validation...\n")

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    summary_rows = []

    for model_name, model in models.items():
        print("--------------------------------------------------")
        print(f" Training model: {model_name}")
        print("--------------------------------------------------")

        fold_results = []
        y_true_all = []
        y_prob_all = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = y_pred.astype(float)

            metrics = compute_fold_metrics(y_test, y_pred, y_prob)
            metrics["Fold"] = fold_idx
            fold_results.append(metrics)

            y_true_all.append(y_test)
            y_prob_all.append(y_prob)

            print(f"  Fold {fold_idx:2d} | Acc={metrics['Accuracy']:.4f} | F1={metrics['F1']:.4f}")

        # Save per-fold metrics
        df_folds = pd.DataFrame(fold_results).sort_values("Fold")

        avg = df_folds.mean(numeric_only=True)
        avg["Fold"] = "Average"
        df_folds = pd.concat([df_folds, avg.to_frame().T], ignore_index=True)

        csv_path = os.path.join(SAVE_FOLDER, f"{model_name}_10fold_metrics.csv")
        df_folds.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

        # ---------------------------------------------
        # ROC Curve
        # ---------------------------------------------
        y_true_all = np.concatenate(y_true_all)
        y_prob_all = np.concatenate(y_prob_all)

        fpr, tpr, _ = roc_curve(y_true_all, y_prob_all)
        model_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {model_auc:.4f}")
        plt.plot([0, 1], [0, 1], "--")
        plt.title(f"ROC Curve – {model_name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.tight_layout()

        if SAVE_ROC:
            roc_path = os.path.join(SAVE_FOLDER, f"ROC_{model_name}.png")
            plt.savefig(roc_path, dpi=120)
            print(f"Saved ROC: {roc_path}")

        plt.show()

        avg_row = avg.copy()
        avg_row["Model"] = model_name
        summary_rows.append(avg_row)

    # -----------------------------------------
    # Summary table
    # -----------------------------------------
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(SAVE_FOLDER, "Models_Summary_Averages.csv")
    summary_df.to_csv(summary_csv, index=False)

    print("\n===== SUMMARY OF MODEL AVERAGES (10-FOLD CV) =====")
    print(summary_df[[
        "Model",
        "Accuracy", "BalancedAccuracy",
        "Precision", "Recall", "F1",
        "ErrorRate",
        "TSS", "HSS",
        "BS", "BSS"
    ]])
    print("==================================================")
    print(f"Saved summary → {summary_csv}")

    return summary_df



# -------------------------------------------------------------
# Script Entry Point
# -------------------------------------------------------------
if __name__ == "__main__":
    SAVE_ROC = False  # toggle if you want PNGs saved
    run_experiment(SAVE_ROC=SAVE_ROC)