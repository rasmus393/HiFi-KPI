import torch
from transformers import AutoModelForTokenClassification
from functools import partial
from datasets import DatasetDict
from sklearn.metrics import classification_report
import pandas as pd
import json
from accelerate import Accelerator
from transformers import BertTokenizerFast

from src.Preprocessing.misc import create_dataloaders
from src.Preprocessing.no_BIO import process_dataset, tokenize_and_align_labels, create_ner_dataset_from_label2id

model_name = "AAU-NLP/Pre-FLANG-BERT-SL1000"
safe_model_name = model_name.replace("/", "_")
batch_size = 192

with open(r'Granularity1_test.json', 'r', encoding='utf-8') as file:
    test = json.load(file)
processed_data_test = process_dataset(test)

datasets = DatasetDict({
    "test": processed_data_test
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizerFast.from_pretrained(model_name)

model = AutoModelForTokenClassification.from_pretrained(model_name)
model.to(device)

id2label = model.config.id2label
label2id = model.config.label2id

tokenized_datasets = {}
for split_name in ["test"]:
    processed_dataset = create_ner_dataset_from_label2id(datasets[split_name], label2id)

    required_columns = ["tokens", "ner_tags"]
    tokenized_datasets[split_name] = processed_dataset.map(
        partial(tokenize_and_align_labels, tokenizer),
        batched=True,
        remove_columns=processed_dataset.column_names,
    )

accelerator = Accelerator()
model = accelerator.prepare(model)
_, _, test_dataloader = create_dataloaders(tokenized_datasets, batch_size, tokenizer)

test_dataloader = accelerator.prepare(test_dataloader)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        attention_mask = attention_mask.detach().cpu().numpy()

        # Collect only valid predictions/labels
        for p, l, am in zip(preds, labels, attention_mask):
            valid_mask = (l != -100) & (am == 1)
            all_preds.extend(p[valid_mask].tolist())
            all_labels.extend(l[valid_mask].tolist())

unique_labels = sorted(set(all_labels + all_preds))
label_list = [id2label[i] for i in unique_labels]

report_dict = classification_report(
    all_labels,
    all_preds,
    target_names=label_list,
    output_dict=True
)

print("Accuracy on test set:", report_dict["accuracy"])
print("Weighted F1-score on test set:", report_dict["weighted avg"]["f1-score"])
print("Macro F1-score on test set:", report_dict["macro avg"]["f1-score"])
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(f"classification_report_{model_name}.csv", index=True)

