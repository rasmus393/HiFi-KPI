from sentence_transformers import SentenceTransformer

def extract_data(data):
    """
    Extracts sentences and their corresponding first entity labels.
    If no entity exists, assigns 'UNKNOWN'.
    """
    sentences = []
    labels = []

    for item in data:
        sentences.append(item["text"])
        label = item["entities"][0]["Label"] if item["entities"] else "UNKNOWN"
        labels.append(label)

    return sentences, labels

def make_embeddings(train_data,validation_data,test_data, embedding_model):
    train_sentences, train_labels = extract_data(train_data)
    val_sentences, val_labels = extract_data(validation_data)
    test_sentences, test_labels = extract_data(test_data)
    model = SentenceTransformer(embedding_model)

    train_embeddings = model.encode(train_sentences, batch_size=256, show_progress_bar=True)
    val_embeddings = model.encode(val_sentences, batch_size=256, show_progress_bar=True)
    test_embeddings = model.encode(test_sentences, batch_size=256, show_progress_bar=True)

    return train_embeddings, train_labels, val_embeddings, val_labels, test_embeddings, test_labels





