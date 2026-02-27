from transformers import  DataCollatorForTokenClassification
import json
from functools import partial
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader

def create_dataloaders(dataset, batch_size, tokenizer):
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    train_dataloader = None
    eval_dataloader = None
    test_dataloader = None

    # Create DataLoader for the train subset if it exists
    if "train" in dataset:
        train_dataloader = DataLoader(
            dataset["train"],
            shuffle=True,
            collate_fn=data_collator,
            batch_size=batch_size,
        )

    # Create DataLoader for the validation subset if it exists
    if "validation" in dataset:
        eval_dataloader = DataLoader(
            dataset["validation"],
            collate_fn=data_collator,
            batch_size=batch_size,
        )

    # Create DataLoader for the test subset if it exists
    if "test" in dataset:
        test_dataloader = DataLoader(
            dataset["test"],
            collate_fn=data_collator,
            batch_size=batch_size,
        )

    return train_dataloader, eval_dataloader, test_dataloader

def generate_dataset_from_json(train_location, validation_location, tokenizer, label_count, test_location=None):
    # Load training dataset
    with open(train_location, 'r', encoding='utf-8') as file:
        train_dataset = json.load(file)
    
    # Load validation dataset
    with open(validation_location, 'r', encoding='utf-8') as file:
        validation_dataset = json.load(file)
    
    # Process training dataset
    train_dataset, label2id, id2label = create_ner_dataset(train_dataset, label_count)
    
    # Process validation dataset
    validation_dataset, label2id, id2label = create_ner_test(validation_dataset, label2id, id2label)
    
    # Initialize the dataset dictionary with train and validation datasets
    dataset = DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset
    })
    
    # If a test location is specified, load and process the test dataset
    if test_location:
        with open(test_location, 'r', encoding='utf-8') as file:
            test_dataset = json.load(file)
        
        test_dataset, _, _ = create_ner_test(test_dataset, label2id, id2label)
        dataset["test"] = test_dataset  # Add the test dataset to the dataset dictionary

    # Tokenize the datasets
    tokenized_datasets = dataset.map(
        partial(tokenize_and_align_labels, tokenizer),  # Pre-fill the tokenizer
        batched=True,
        remove_columns=dataset['train'].column_names,
    )
    
    return label2id, id2label, tokenized_datasets

def create_label_mappings(dataset):
    ner_feature = dataset["train"].features["ner_tags"]
    label_names = ner_feature.feature.names
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id
