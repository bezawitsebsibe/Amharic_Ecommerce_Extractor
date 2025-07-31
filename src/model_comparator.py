import os
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
)
from datasets import load_dataset
import evaluate

from ner_model_utils import get_dataset_splits, compute_metrics
import numpy as np

MODEL_LIST = {
    "xlm-roberta": "Davlan/afro-xlmr-mini",
    "distilbert": "Davlan/distilmBERT-base-multilingual-cased-ner-hrl",
    "mbert": "bert-base-multilingual-cased"
}

def train_model(model_name, model_ckpt, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = AutoModelForTokenClassification.from_pretrained(model_ckpt, num_labels=8)  # Adjust num_labels as needed

    train_dataset, val_dataset = get_dataset_splits(tokenizer)

    # NOTE: Your transformers version 4.54.0 does NOT support evaluation_strategy param.
    # So we remove it to avoid the error.
    training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir=f"{output_dir}/logs",
    logging_steps=10,
    save_strategy="epoch",
)



    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

    trainer.train()
    results = trainer.evaluate()
    return results

def main():
    results = {}
    for name, checkpoint in MODEL_LIST.items():
        print(f"\n--- Training {name} ---")
        out_path = f"outputs/fine_tuned_models/{name}"
        os.makedirs(out_path, exist_ok=True)
        metrics = train_model(name, checkpoint, out_path)
        results[name] = metrics

    print("\n=== Model Comparison Results ===")
    for model_name, metrics in results.items():
        print(f"{model_name}: {metrics}")

if __name__ == "__main__":
    main()
