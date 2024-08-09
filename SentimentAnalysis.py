import os
import random
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from datasets import load_dataset

from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


def _load_data():
    emotion_dataset = load_dataset('dair-ai/emotion')

    idx2lbl = {idx: lbl for idx, lbl in enumerate(emotion_dataset['train'].features['label'].names)}
    lbl2idx = {lbl: idx for idx, lbl in idx2lbl.items()}

    tokenized_dataset = emotion_dataset.map(_tokenize_seqs, batched=True)
    tokenized_dataset = tokenized_dataset.rename_column('label', 'labels')

    return tokenized_dataset, idx2lbl, lbl2idx


def _get_tokenizer():
    return AutoTokenizer.from_pretrained('bert-base-uncased')


def _tokenize_seqs(examples):
    tokenizer = _get_tokenizer()
    return tokenizer(examples['text'], truncation=True, max_length=512)


def _load_model(idx2lbl, lbl2idx):
    return AutoModelForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=len(idx2lbl),
        id2label=idx2lbl,
        label2id=lbl2idx,
    )


def compute_metrics(eval_preds):
    preds, labels = eval_preds.predictions, eval_preds.label_ids

    preds = np.argmax(preds, axis=1)

    f1 = f1_score(labels, preds, average='weighted')
    return {'f1': f1}


def _training(model, tokenized_dataset, tokenizer):
    training_args = TrainingArguments(
        output_dir='./logs/run1',
        per_device_train_batch_size=4,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        weight_decay=1e-3,
        num_train_epochs=3,
        evaluation_strategy='epoch',
        logging_strategy='steps',
        logging_steps=len(tokenized_dataset['train']) / 4,
        save_strategy='epoch',
        save_total_limit=1,
        seed=42,
        fp16=True,
        dataloader_num_workers=2,
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        tokenizer=tokenizer,
    )

    trainer.train()

    return trainer


def _eval(trainer, tokenized_dataset):
    test_predictions = trainer.predict(tokenized_dataset['test'])

    print(f'Test F1 score: {test_predictions.metrics["test_f1"]}')

    predictions = np.argmax(test_predictions.predictions, axis=1)
    true_labels = tokenized_dataset["test"]["labels"]

    unique_labels = sorted(set(true_labels))
    correct_counts = {label: 0 for label in unique_labels}
    total_counts = {label: 0 for label in unique_labels}

    for true_label, predicted_label in zip(true_labels, predictions):
        total_counts[true_label] += 1
        if true_label == predicted_label:
            correct_counts[true_label] += 1

    labels = []
    correct_percentages = []
    incorrect_percentages = []

    for label in unique_labels:
        total = total_counts[label]
        correct = correct_counts[label]
        correct_pct = (correct / total) * 100 if total > 0 else 0
        incorrect_pct = 100 - correct_pct
        labels.append(label)
        correct_percentages.append(correct_pct)
        incorrect_percentages.append(incorrect_pct)

    plt.bar(labels, correct_percentages, label='Correct', color='green')
    plt.bar(labels, incorrect_percentages, bottom=correct_percentages, label='Incorrect', color='red')

    plt.xlabel("Labels")
    plt.ylabel("Percentage")
    plt.title("Percentages of Correct and Incorrect Predictions per Label")
    plt.legend(title="Prediction Type")

    plt.savefig("correct_incorrect_percentages.png")


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    print(f'Using device: {device}')
    tokenized_dataset, idx2lbl, lbl2idx = _load_data()
    model = _load_model(idx2lbl, lbl2idx)
    trainer = _training(model, tokenized_dataset, _get_tokenizer())
    _eval(trainer, tokenized_dataset)

    trainer.save_model('ft_bert_emotion')


if __name__ == "__main__":
    main()
