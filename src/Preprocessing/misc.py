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
