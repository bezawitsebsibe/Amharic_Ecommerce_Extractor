from datasets import Dataset
from transformers import AutoTokenizer
import pandas as pd

def read_conll_data(file_path):
    tokens, labels = [], []
    current_tokens, current_labels = [], []

    with open(file_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_tokens:
                    tokens.append(current_tokens)
                    labels.append(current_labels)
                    current_tokens, current_labels = [], []
            else:
                splits = line.split()
                if len(splits) == 2:
                    token, label = splits
                    current_tokens.append(token)
                    current_labels.append(label)

    return tokens, labels

def convert_to_dataset(tokens, labels):
    data = [{'tokens': t, 'ner_tags': l} for t, l in zip(tokens, labels)]
    return Dataset.from_list(data)

def tokenize_and_align_labels(examples, tokenizer, label2id):
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True)
    labels = []

    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            else:
                label_ids.append(label2id[label[word_idx]])  # could be -100 for subwords
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs
