from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

def classification_report_with_supports(
    test_labels_enc, 
    test_preds, 
    label_encoder, 
    train_labels_enc
):
    """
    Generate a classification report DataFrame that includes:
      - precision, recall, f1-score, support
      - training support for each label
      - rows for labels that appear in training but not in test
    
    Args:
        test_labels_enc (array-like): Encoded test labels
        test_preds (array-like): Predictions on the test set (encoded)
        label_encoder (LabelEncoder): The fitted LabelEncoder object
        train_labels_enc (array-like): Encoded training labels
    Returns:
        pd.DataFrame: Classification report with additional columns
    """
    unique_labels, label_counts = np.unique(test_labels_enc, return_counts=True)
    sorted_indices = np.argsort(label_counts)[::-1]
    sorted_labels = unique_labels[sorted_indices]
    report = classification_report(test_labels_enc, test_preds, output_dict=True)
    decoded_labels = label_encoder.inverse_transform(sorted_labels)
    report_filtered = {
        decoded_label: report[str(int(encoded_label))]
        for decoded_label, encoded_label in zip(decoded_labels, sorted_labels)
        if str(int(encoded_label)) in report
    }
    df_report = pd.DataFrame(report_filtered).T
    df_report = df_report[['precision', 'recall', 'f1-score', 'support']]
    train_unique_labels, train_label_counts = np.unique(train_labels_enc, return_counts=True)
    train_support_dict = {
        label_encoder.inverse_transform([lbl])[0]: count
        for lbl, count in zip(train_unique_labels, train_label_counts)
    }
    df_report["train_support"] = df_report.index.map(train_support_dict).fillna(0).astype(int)
    train_only_labels = set(train_support_dict.keys()) - set(df_report.index)

    for label in train_only_labels:
        df_report.loc[label] = {
            "precision": np.nan,
            "recall": np.nan,
            "f1-score": np.nan,
            "support": np.nan,  # No support in test set
            "train_support": train_support_dict[label],
        }

    #Sort by test support and then training support
    df_report = df_report.sort_values(by=["support", "train_support"], ascending=[False, False])
    return df_report