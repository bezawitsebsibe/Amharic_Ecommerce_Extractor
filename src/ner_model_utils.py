from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import evaluate

def get_dataset_splits(tokenizer):
    # Load dataset
    dataset = load_dataset('text', data_files={'train': 'data/amharic_conll_data.txt'}, split='train')

    def tokenize_and_align_labels(example):
        tokens = tokenizer(example['text'], truncation=True, padding='max_length')
        tokens["labels"] = [0] * len(tokens["input_ids"])  # Dummy labels, replace with real labels as needed
        return tokens

    tokenized = dataset.map(tokenize_and_align_labels)
    return tokenized.train_test_split(test_size=0.2).values()

# Initialize the metric once
metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    preds = predictions.argmax(-1)

    acc = accuracy_score(labels.flatten(), preds.flatten())
    f1 = f1_score(labels.flatten(), preds.flatten(), average='weighted')
    precision = precision_score(labels.flatten(), preds.flatten(), average='weighted')
    recall = recall_score(labels.flatten(), preds.flatten(), average='weighted')
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
