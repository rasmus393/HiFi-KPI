import json
import pickle
import torch, gc
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.Modelling.Text_classifier.convert_to_depth import multi_level_collapse
from src.Modelling.Text_classifier.bert_based_embeddings import make_embeddings
from src.Modelling.Text_classifier.report import classification_report_with_supports
from src.Modelling.Text_classifier.model import run_classifier
from datasets import load_dataset
import argparse
from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser(description="Choose taxonomy")
parser.add_argument(
    "--taxonomy",
    choices=["calculation", "presentation"],
    default="presentation",
    help="Select the taxonomy to use (default: presentation)."
)

args = parser.parse_args()
dataset = load_dataset("AAU-NLP/HiFi-KPI")
train_data = dataset["train"]
validation_data = dataset["validation"]
test_data = dataset["test"]

taxonomy_file = (
    "calculationMasterTaxonomy.jsonl"
    if args.taxonomy == "calculation"
    else "presentationMasterTaxonomy.jsonl"
)

file_path = hf_hub_download(
    repo_id="AAU-NLP/HiFi-KPI",
    filename=taxonomy_file,
    repo_type="dataset"
)

taxonomy_df = pd.read_json(file_path, orient='records', lines=True)

multi_collap = multi_level_collapse(taxonomy_df, train_data, validation_data, test_data, iterations=10)
train_embeddings, train_labels, val_embeddings, val_labels, test_embeddings, test_labels = make_embeddings(train_data,validation_data,test_data, embedding_model="google/embeddinggemma-300m")
input_dim = train_embeddings.shape[1]
print(f"Input dim: {input_dim}")
encoders = []
for i in range(10):
    all_labels = multi_collap[0][i] + multi_collap[1][i] + multi_collap[2][i]
    encoder = LabelEncoder()
    encoder.fit(all_labels)
    encoders.append(encoder)

all_reports = []
test_acc_level = []
test_f1s_level = []

for i in range(0, len(multi_collap[3]), 2):
    child_path_dict = multi_collap[3][i]

    print(f"Index: {i}")
    cur_label_encoder = encoders[i]
    test_accs = []
    test_f1s = []
    reports_for_level_i = []
    for run in range(1):
        cur_train = cur_label_encoder.transform(multi_collap[0][i])
        cur_val = cur_label_encoder.transform(multi_collap[1][i])
        test = cur_label_encoder.transform(multi_collap[2][i])
        model = run_classifier(train_embeddings,cur_train, val_embeddings, cur_val, cur_label_encoder, input_dim=input_dim, batch_size=128, device="cuda")
        
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        val_embeddings = torch.as_tensor(val_embeddings, dtype=torch.float32, device=device)
        test_embeddings = torch.as_tensor(test_embeddings, dtype=torch.float32, device=device)

        def batched_inference(model, embeddings, batch_size=64):
            model.eval()
            all_preds = []
            with torch.no_grad():
                for start_idx in range(0, embeddings.size(0), batch_size):
                    end_idx = start_idx + batch_size
                    batch = embeddings[start_idx:end_idx]
                    outputs = model(batch)
                    preds = torch.argmax(outputs, dim=1)
                    all_preds.append(preds)
            return torch.cat(all_preds).cpu().numpy()
        batch_size = 2048
        val_preds = batched_inference(model, val_embeddings, batch_size)
        test_preds = batched_inference(model, test_embeddings, batch_size)

        # Calculate and print metrics for the validation set
        val_acc = accuracy_score(cur_val, val_preds)
        val_f1 = f1_score(cur_val, val_preds, average='macro')
        print(f"[VAL] Accuracy: {val_acc:.4f}, F1 (macro): {val_f1:.4f}")

        # Calculate and print metrics for the test set
        test_acc = accuracy_score(test, test_preds)
        test_f1 = f1_score(test, test_preds, average='macro')
        test_precision = precision_score(test, test_preds, average='macro')
        test_recall = recall_score(test, test_preds, average='macro')
        test_micro_f1 = f1_score(test, test_preds, average='micro')

        test_accs.append(test_acc)
        test_f1s.append(test_f1)

        print(f"[TEST] Accuracy: {test_acc:.4f}, F1 (macro): {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, Micro F1: {test_micro_f1:.4f}")
        true_labels_str = cur_label_encoder.inverse_transform(test)
        predicted_labels_str = cur_label_encoder.inverse_transform(test_preds)
        test_texts = [item['text'] for item in test_data]

        output_df = pd.DataFrame({
            'text': test_texts,
            'true_label': true_labels_str,
            'predicted_label': predicted_labels_str
        })

        output_filename = f"Final_reports/cal_predictions_Level_{i+1}.csv"
        output_df.to_csv(output_filename, index=False)
        del model 
        gc.collect()
        torch.cuda.empty_cache()

        df_report = classification_report_with_supports(
            test_labels_enc=test,
            test_preds=test_preds,
            label_encoder=encoders[i],
            train_labels_enc=encoders[i].transform(multi_collap[0][i])
        )
        # Store the report DataFrame for this iteration
        reports_for_level_i.append(df_report)
    test_acc_level.append(test_accs)
    test_f1s_level.append(test_f1s)
    all_reports.append(reports_for_level_i)

print("test_accs", test_acc_level)
print("test f1s", test_f1s_level)

for level_idx, reports_for_specific_level in enumerate(all_reports):
    original_i_value = 1 + level_idx * 2
    for single_report_df in reports_for_specific_level:
        filename = f"Final_reports/cal_Level_{original_i_value}_clean_train_gemma_embeddings.csv"
        single_report_df.to_csv(filename)
