from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from seqeval.metrics import classification_report
from ner_model_utils import read_conll_data, convert_to_dataset, tokenize_and_align_labels
import numpy as np
import torch

# ======== Load and Prepare Dataset ========
model_checkpoint = "Davlan/afro-xlmr-mini"
  # or "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Labels used in your CoNLL file
label_list = [
    "O",
    "B-Product", "I-Product",
    "B-LOC", "I-LOC",
    "B-PRICE", "I-PRICE",
]

label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}


tokens, tags = read_conll_data("data/amharic_conll_data.txt")
dataset = convert_to_dataset(tokens, tags)

# Split data
dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset['train']
val_dataset = dataset['test']

# Tokenize and align
tokenized_train = train_dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer, label2id), batched=True)
tokenized_val = val_dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer, label2id), batched=True)

# ======== Load Model ========
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# ======== Training Setup ========
args = TrainingArguments(
    output_dir="outputs/fine_tuned_model",
    #evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    save_strategy="epoch",
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# ======== Trainer ========
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    pred_labels = [[id2label[p] for (p, l) in zip(prediction, label) if l != -100]
                   for prediction, label in zip(predictions, labels)]
    
    report = classification_report(true_labels, pred_labels, output_dict=True)
    return {
        "f1": report["weighted avg"]["f1-score"],
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"]
    }

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ======== Train ========
trainer.train()

# ======== Save Model ========
trainer.save_model("outputs/fine_tuned_model")
