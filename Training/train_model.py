from transformers import AutoTokenizer
from src.Preprocessing.misc import create_dataloaders
from src.Modelling.model import init_model, run_train_loop
from functools import partial
from datasets import DatasetDict
from src.Preprocessing.no_BIO import process_dataset, create_ner_dataset_processed_data, tokenize_and_align_labels, create_ner_dataset_from_label2id
from datasets import load_dataset
model_name = "SALT-NLP/FLANG-BERT"

label_count = 999
batch_size = 96
learning_rate = 1e-5
patience = 2
num_train_epochs = 50
outputdir = "FLANG_1000_no_BIO_final_cleaned"

dataset = load_dataset("AAU-NLP/HiFi-KPI")
train = dataset['train']
validation = dataset['validation']
test = dataset['test']

processed_data = process_dataset(train)
processed_data_val = process_dataset(validation)
processed_data_test = process_dataset(test)

datasets = DatasetDict({
    "train": processed_data,
    "validation": processed_data_val,
    "test": processed_data_test,
})

tokenized_datasets = {}
tokenizer = AutoTokenizer.from_pretrained(model_name)
train_dataset, label2id, id2label = create_ner_dataset_processed_data(datasets["train"], label_count)

tokenized_datasets["train"] = train_dataset.map(
    partial(tokenize_and_align_labels, tokenizer),
    batched=True,
    remove_columns=train_dataset.column_names,
)

for split_name in ["validation", "test"]:
    processed_dataset = create_ner_dataset_from_label2id(datasets[split_name], label2id)
    tokenized_datasets[split_name] = processed_dataset.map(
        partial(tokenize_and_align_labels, tokenizer),
        batched=True,
        remove_columns=processed_dataset.column_names,
    )

train_dataloader, eval_dataloader, test_dataloader = create_dataloaders(tokenized_datasets, batch_size, tokenizer)
model, optimizer, accelerator, train_dataloader, eval_dataloader = init_model(model_name, train_dataloader, eval_dataloader, id2label,label2id, learning_rate)
run_train_loop(model, accelerator, tokenizer, optimizer, train_dataloader, eval_dataloader, id2label, label2id, num_train_epochs, patience, outputdir)