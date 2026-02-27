import spacy
from collections import Counter
from datasets import Dataset

def create_bio_tags(text, entities):
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    doc = nlp(text)
    tokens = [token.text for token in doc]
    bio_tags = ["O"] * len(tokens)

    for entity in entities:
        start_token = None
        end_token = None
        for i, token in enumerate(doc):
            if token.idx == entity["Start character"]:
                start_token = i
            if token.idx + len(token) == entity["End character"]:
                end_token = i

        if start_token is not None and end_token is not None:
            bio_tags[start_token] = f"{entity['Label']}"
            for i in range(start_token + 1, end_token + 1):
                # I-token is most likely only present due to edge cases in data e.g. a weird span white space tagged or similar.
                print("no I-token")
    
    return tokens, bio_tags

def process_dataset(data):
    processed_data = []
    for entry in data:
        tokens, labels = create_bio_tags(entry["text"], entry["entities"])
        processed_data.append({
            'tokens': tokens,
            'labels': labels
        })
    hf_dataset = Dataset.from_list(processed_data)
    
    return hf_dataset

def align_labels_with_tokens(labels, word_ids):
    previous_word_id = None
    new_labels = []
    for word_id in word_ids:
        if word_id is None:
            new_labels.append(-100)
        elif word_id != previous_word_id:
            new_labels.append(labels[word_id])
        else:
            new_labels.append(-100)
        previous_word_id = word_id
    return new_labels

def tokenize_and_align_labels(tokenizer,  examples ):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        max_length=512,
        truncation=True,
        is_split_into_words=True,
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

def create_ner_dataset_processed_data(processed_data, num_of_tags):
    tag_counts = Counter()
    for elems in processed_data:
        for elem1 in elems['labels']:
                tag_counts[elem1] += 1

    most_common_tags = [tag for tag, _ in tag_counts.most_common() if tag != 'O'][:num_of_tags]
    label_names = ['O'] + [f'{tag}' for tag in most_common_tags] + ["specialOOS"]

    label2id = {label: idx for idx, label in enumerate(label_names)}
    id2label = {idx: label for label, idx in label2id.items()}

    tokens = []
    labels = []

    for entry in processed_data:
        
        tokens.append(entry['tokens'])
        entry_labels = []
        
        for label in entry['labels']:
            if label == 'O':
                entry_labels.append(label2id['O'])
            elif label in label_names:
                entry_labels.append(label2id[label])
            else:
                prefix = label[:2] if label.startswith(('B-', 'I-')) else ''
                special_label = f"{prefix}specialOOS"
                entry_labels.append(label2id[special_label])
        
        labels.append(entry_labels)

    dataset = Dataset.from_dict({
        'tokens': tokens,
        'ner_tags': labels
    })

    return dataset, label2id, id2label

def create_ner_dataset_from_label2id(processed_data, label2id):
    tokens = []
    labels = []

    for entry in processed_data:
        tokens.append(entry['tokens'])
        entry_labels = []
        
        for label in entry['labels']:
            if label in label2id:
                entry_labels.append(label2id[label])
            else:
                # we remove the prefix to be agnostic to if it is BIO schema or just direct prediction.
                prefix = label[:2] if label.startswith(('B-', 'I-')) else ''
                special_label = f"{prefix}specialOOS"
                entry_labels.append(label2id[special_label])

        labels.append(entry_labels)

    dataset = Dataset.from_dict({
        'tokens': tokens,
        'ner_tags': labels
    })

    return dataset
