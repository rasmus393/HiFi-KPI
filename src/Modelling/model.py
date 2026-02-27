from transformers import AutoModelForTokenClassification
from torch.optim import AdamW
from accelerate import Accelerator
from src.metrics import postprocess
from transformers import get_scheduler
from tqdm.auto import tqdm
import numpy as np
from itertools import islice
import evaluate
import torch
import logging
from datetime import datetime
import os

def init_model(model_name, train_dataloader, eval_dataloader, id2label,label2id, learning_rate): 
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    return model, optimizer, accelerator, train_dataloader, eval_dataloader

def evaluate_and_report(model, dataloader, accelerator, label_names, metric):
    """Evaluate the model and return evaluation results."""
    model.eval()
    all_predictions = []
    all_labels = []

    # Evaluate on the provided dataloader
    for batch in dataloader:
        with torch.no_grad():
            outputs = model(**batch)

        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]

        predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

        predictions_gathered = accelerator.gather(predictions)
        labels_gathered = accelerator.gather(labels)

        true_predictions, true_labels = postprocess(predictions_gathered, labels_gathered, list(label_names))
        metric.add_batch(predictions=true_predictions, references=true_labels)

        all_predictions.extend(true_predictions)
        all_labels.extend(true_labels)

    results = metric.compute()
    return results

def setup_logger(model_name, batch_size, learning_rate):
    """Set up logging configuration"""
    model_name_clean = model_name.replace('/', '_')
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"logs/log_{model_name_clean}_batch{batch_size}_lr{learning_rate}_{timestamp}.log"
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def run_train_loop(model, accelerator, tokenizer, optimizer,
                    train_dataloader, eval_dataloader, id2label,
                    label2id, epochs, patience, output_dir, batches_to_evaluate=1000):
    
    # Extract batch size and learning rate from optimizer
    for batch in train_dataloader:
        batch_size = batch['input_ids'].shape[0] 
        break
    lr = optimizer.param_groups[0]['lr']
    
    # Setup logger
    logger = setup_logger(output_dir.split('_')[0], batch_size, lr)
    logger.info(f"Starting training with configuration:")
    logger.info(f"Model: {output_dir.split('_')[0]}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {lr}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Patience: {patience}")
    logger.info(f"Output directory: {output_dir}")
    
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    progress_bar = tqdm(range(epochs), desc="Training", position=0)
    
    best_f1 = -np.inf
    patience_counter = 0
    
    label_names = list(id2label.values())
    metric = evaluate.load("seqeval")
    
    for epoch in range(epochs):
        # Training
        model.train()
        batch_progress = tqdm(train_dataloader,
                            desc=f"Epoch {epoch+1}",
                            position=1,
                            leave=False,
                            total=len(train_dataloader))
        
        for batch_num, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            
            if batch_num % batches_to_evaluate == 0:
                partial_results = evaluate_and_report(
                    model, islice(eval_dataloader, 20), 
                    accelerator, label_names, metric
                )
                metrics = {
                    key: partial_results[f"overall_{key}"]
                    for key in ["precision", "recall", "f1", "accuracy"]
                }
                logger.info(f"Epoch {epoch+1}, Batch {batch_num} metrics: {metrics}")
                
                model.train()
        
        # Full evaluation at the end of the epoch
        results = evaluate_and_report(model, eval_dataloader, accelerator, label_names, metric)
        epoch_f1 = results["overall_f1"]
        
        metrics_str = {
            key: f"{results[f'overall_{key}']:.4f}"
            for key in ["precision", "recall", "f1", "accuracy"]
        }
        
        logger.info(f"Epoch {epoch + 1} complete. Metrics: {metrics_str}")
        progress_bar.set_postfix(metrics=metrics_str)
        progress_bar.update(1)
        
        # Early stopping check
        if epoch_f1 > best_f1:
            best_f1 = epoch_f1
            patience_counter = 0
            logger.info(f"New best F1: {best_f1:.4f} - Saving model...")
            
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(output_dir)
            model.save_pretrained(output_dir)
        else:
            patience_counter += 1
            logger.info(f"No improvement in F1 score for {patience_counter} epochs.")
            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break
        
        batch_progress.close()
    
    progress_bar.close()
    logger.info("Training completed!")