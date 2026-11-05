
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ==========================================================
# 1. PROJECT PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR / "data" / "processed" / "normalized_labels.csv"
)

PREDICTIONS_PATH = (
    BASE_DIR / "outputs" / "evaluation" / "model_predictions.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs" / "evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2. LABEL DEFINITIONS
# ==========================================================

TRUE_LABELS = {
    "TRUE"
}

FALSE_LABELS = {
    "FALSE",
    "MOSTLY_FALSE",
    "MISLEADING",
    "MANIPULATED"
}

ABSTENTION_LABELS = {
    "INSUFFICIENT_EVIDENCE",
    "NAN",
    "NONE",
    ""
}


# ==========================================================
# 3. NORMALIZATION FUNCTIONS
# ==========================================================

def normalize_id(value):

    value = str(value).strip().upper()

    value = value.replace("_", "-")
    value = value.replace(" ", "")

    return value


def normalize_label(value):

    if pd.isna(value):
        return ""

    return str(value).strip().upper()


def convert_to_binary(label):

    label = normalize_label(label)

    if label in TRUE_LABELS:
        return "TRUE"

    if label in FALSE_LABELS:
        return "FALSE"

    return None


# ==========================================================
# 4. LOAD DATA
# ==========================================================

def load_data():

    print("\nLoading dataset and predictions...")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Predictions not found: {PREDICTIONS_PATH}"
        )

    dataset = pd.read_csv(DATASET_PATH)

    predictions = pd.read_csv(PREDICTIONS_PATH)

    print("\nDataset columns:")
    print(dataset.columns.tolist())

    print("\nPrediction columns:")
    print(predictions.columns.tolist())

    return dataset, predictions


# ==========================================================
# 5. IDENTIFY REQUIRED COLUMNS
# ==========================================================

def find_column(dataframe, possible_names):

    for name in possible_names:

        if name in dataframe.columns:
            return name

    raise ValueError(
        f"Could not find any of these columns: {possible_names}"
    )


# ==========================================================
# 6. PREPARE DATA
# ==========================================================

def prepare_data(dataset, predictions):

    ground_truth_id_column = find_column(
        dataset,
        ["Sample_ID", "Sample ID", "SampleId"]
    )

    ground_truth_label_column = find_column(
        dataset,
        ["Normalized_Label", "Claim_Label", "Label"]
    )

    prediction_id_column = find_column(
        predictions,
        ["Sample_ID", "Sample ID", "SampleId"]
    )

    prediction_label_column = find_column(
        predictions,
        ["Predicted_Label", "Prediction", "Predicted"]
    )

    dataset = dataset.copy()
    predictions = predictions.copy()

    # Normalize IDs.

    dataset["Normalized_ID"] = (
        dataset[ground_truth_id_column].apply(normalize_id)
    )

    predictions["Normalized_ID"] = (
        predictions[prediction_id_column].apply(normalize_id)
    )

    # Normalize labels.

    dataset["Ground_Truth_Label"] = (
        dataset[ground_truth_label_column].apply(normalize_label)
    )

    predictions["Predicted_Label_Clean"] = (
        predictions[prediction_label_column].apply(normalize_label)
    )

    # Convert multiclass labels into binary labels.

    dataset["Ground_Truth_Binary"] = (
        dataset["Ground_Truth_Label"].apply(convert_to_binary)
    )

    predictions["Predicted_Binary"] = (
        predictions["Predicted_Label_Clean"].apply(convert_to_binary)
    )

    # Remove duplicate prediction IDs, keeping the latest row.

    predictions = predictions.drop_duplicates(
        subset=["Normalized_ID"],
        keep="last"
    )

    # Merge by normalized sample ID.

    merged = dataset.merge(
        predictions[
            [
                "Normalized_ID",
                "Predicted_Label_Clean",
                "Predicted_Binary"
            ]
        ],
        on="Normalized_ID",
        how="left",
        indicator=True
    )

    return merged, dataset, predictions


# ==========================================================
# 7. DATASET SUMMARY
# ==========================================================

def display_dataset_summary(dataset):

    print("\n========== DATASET SUMMARY ==========")

    print("Total dataset samples:", len(dataset))

    print("\nOriginal label distribution:")

    print(
        dataset["Ground_Truth_Label"].value_counts(
            dropna=False
        )
    )

    print("\nBinary ground-truth distribution:")

    print(
        dataset["Ground_Truth_Binary"].value_counts(
            dropna=False
        )
    )


# ==========================================================
# 8. PREDICTION SUMMARY
# ==========================================================

def display_prediction_summary(merged):

    total_samples = len(merged)

    matched = merged["_merge"] == "both"

    missing = merged["_merge"] == "left_only"

    abstentions = (
        merged["Predicted_Label_Clean"]
        .isin(ABSTENTION_LABELS)
    )

    valid_predictions = (
        merged["Predicted_Binary"].notna()
        & matched
    )

    evaluated_count = int(valid_predictions.sum())

    abstention_count = int(
        (matched & abstentions).sum()
    )

    missing_count = int(missing.sum())

    coverage = (
        evaluated_count / total_samples * 100
        if total_samples else 0
    )

    print("\n========== PREDICTION SUMMARY ==========")

    print("Total samples:", total_samples)
    print("Matched prediction records:", int(matched.sum()))
    print("Evaluated predictions:", evaluated_count)
    print("Abstentions:", abstention_count)
    print("Missing predictions:", missing_count)

    print(f"Prediction coverage: {coverage:.2f}%")

    return valid_predictions, coverage


# ==========================================================
# 9. CLASSIFICATION EVALUATION
# ==========================================================

def evaluate_model(merged, valid_predictions, coverage):

    evaluated = merged.loc[valid_predictions].copy()

    print("\n========== OVERALL METRICS ==========")

    print("Total samples:", len(merged))
    print("Evaluated samples:", len(evaluated))
    print(f"Coverage: {coverage:.2f}%")

    if evaluated.empty:

        print("\nNo valid predictions available for evaluation.")

        return None

    y_true = evaluated["Ground_Truth_Binary"]
    y_pred = evaluated["Predicted_Binary"]

    accuracy = accuracy_score(y_true, y_pred)

    macro_precision = precision_score(
        y_true,
        y_pred,
        labels=["TRUE", "FALSE"],
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        labels=["TRUE", "FALSE"],
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=["TRUE", "FALSE"],
        average="macro",
        zero_division=0
    )

    weighted_precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    weighted_recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Weighted Recall: {weighted_recall:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    # Classification report.

    report = classification_report(
        y_true,
        y_pred,
        labels=["TRUE", "FALSE"],
        target_names=["TRUE", "FALSE"],
        zero_division=0
    )

    print("\n========== CLASSIFICATION REPORT ==========")
    print(report)

    # Confusion matrix.

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=["TRUE", "FALSE"]
    )

    print("\n========== CONFUSION MATRIX ==========")
    print(cm)

    # Save confusion matrix image.

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["TRUE", "FALSE"]
    )

    display.plot(values_format="d")

    plt.title("Misinformation Verification Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=300
    )

    plt.close()

    # Save classification report.

    with open(
        OUTPUT_DIR / "classification_report.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

    # Save metrics.

    metrics = {

        "Total_Samples": len(merged),

        "Evaluated_Samples": len(evaluated),

        "Abstentions": int(
            (
                merged["_merge"].eq("both")
                & merged["Predicted_Label_Clean"].isin(
                    ABSTENTION_LABELS
                )
            ).sum()
        ),

        "Missing_Predictions": int(
            merged["_merge"].eq("left_only").sum()
        ),

        "Prediction_Coverage_Percent": coverage,

        "Accuracy": accuracy,

        "Macro_Precision": macro_precision,

        "Macro_Recall": macro_recall,

        "Macro_F1": macro_f1,

        "Weighted_Precision": weighted_precision,

        "Weighted_Recall": weighted_recall,

        "Weighted_F1": weighted_f1
    }

    pd.DataFrame([metrics]).to_csv(
        OUTPUT_DIR / "evaluation_metrics.csv",
        index=False
    )

    # Save sample-wise evaluation.

    merged.to_csv(
        OUTPUT_DIR / "evaluation_results.csv",
        index=False
    )

    print("\nEvaluation files saved in:")
    print(OUTPUT_DIR)

    print("\n========== EVALUATION COMPLETED ==========")

    return metrics


# ==========================================================
# 10. MAIN
# ==========================================================

def main():

    dataset, predictions = load_data()

    merged, dataset, predictions = prepare_data(
        dataset,
        predictions
    )

    display_dataset_summary(dataset)

    valid_predictions, coverage = display_prediction_summary(
        merged
    )

    evaluate_model(
        merged,
        valid_predictions,
        coverage
    )


if __name__ == "__main__":

    main()
