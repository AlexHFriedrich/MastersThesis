import numpy as np
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


def load_model(path):
    model = AutoModelForSequenceClassification.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer


def tokenize(tokenizer, text):
    return tokenizer(text, truncation=True, max_length=512)


def call_model(model, tokenizer, text):
    tokenized_text = tokenize(tokenizer, text)
    print(tokenized_text["input_ids"])
    return model(tokenized_text["input_ids"].to_array(), tokenized_text["attention_mask"], tokenized_text["token_type_ids"])
